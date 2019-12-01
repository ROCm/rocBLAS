/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "../blas1/rocblas_copy.hpp"
#include "../blas1/rocblas_scal.hpp"
#include "../blas2/rocblas_gemv.hpp"
#include "../blas3/Tensile/gemm.hpp"
#include "dcld.hpp"
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U, typename V>
__global__ void set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                                     rocblas_int    n,
                                                     U              alpha_device_host,
                                                     rocblas_stride stride_alpha,
                                                     V              Aa,
                                                     ptrdiff_t      shifta,
                                                     rocblas_int    lda,
                                                     rocblas_int    strideA)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);

    if(tx < m && ty < n && alpha == 0)
    {
        T* A = load_ptr_batch(Aa, hipBlockIdx_z, shifta, strideA);

        A[tx + lda * ty] = 0;
    }
}

template <typename T, typename U, typename V>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const U*       alpha,
                                                      rocblas_stride stride_alpha,
                                                      V*             A,
                                                      rocblas_int    offsetA,
                                                      rocblas_int    lda,
                                                      rocblas_int    strideA,
                                                      rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 16;
    static constexpr int GEMV_DIM_Y = 16;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 grid(blocksX, blocksY, batch_count);
    dim3 threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(set_matrix_zero_if_alpha_zero_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           alpha,
                           stride_alpha,
                           A,
                           offsetA,
                           lda,
                           strideA);
    else
        hipLaunchKernelGGL(set_matrix_zero_if_alpha_zero_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           *alpha,
                           stride_alpha,
                           A,
                           offsetA,
                           lda,
                           strideA);
    return rocblas_status_success;
}

template <rocblas_int RB, rocblas_int CB, typename T>
rocblas_status rocblas_trmm_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation transa,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     const T*          a,
                                     rocblas_int       lda,
                                     T*                c,
                                     rocblas_int       ldc,
                                     T*                workspace)
{

    //
    // Level 3 Blas routine.
    //
    // -- Written on 8-February-1989.
    //    Jack Dongarra, Argonne National Laboratory.
    //    iain Duff, AERE Harwell.
    //    Jeremy Du Croz, Numerical Algorithms Group Ltd.
    //    Sven Hammarling, Numerical Algorithms Group Ltd.
    //
    // -- Rewritten in December-1993.
    //    GEMM-Based Level 3 BLAS.
    //    Per Ling, institute of information Processing,
    //    University of Umea, Sweden.
    //
    // -- Rewritten for gemm based trmm for rocBLAS
    //
    T zero = 0.0;
    T one  = 1.0;
    //
    //    And when alpha.eq.zero.
    //
    if(rocblas_pointer_mode_host == handle->pointer_mode && 0 == *alpha)
    {
        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
            (set_matrix_zero_if_alpha_zero_template<T>)(handle, m, n, alpha, 0, c, 0, ldc, 0, 1));
        return rocblas_status_success;
    }
    else if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // set matrix to zero and continue calculation. This will give
        // the same functionality as Legacy BLAS. alpha is on device and
        // it should not be copied from device to host because this is
        // an asynchronous function and the copy would make it synchronous.
        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
            (set_matrix_zero_if_alpha_zero_template<T>)(handle, m, n, alpha, 0, c, 0, ldc, 0, 1));
    }

    // grid size for rocblas_copy_template
    constexpr rocblas_int NB = 256;

    // assign space for dt1 and dt2
    rocblas_int rb = RB, cb = CB;
    rocblas_int ldt1 = rb, ldt2 = cb;
    T*          dt1 = workspace;
    T*          dt2 = workspace + rb * cb;

    rocblas_int    offd = rocblas_diagonal_unit == diag ? 1 : 0;
    rocblas_int    isec, jsec, tsec;
    rocblas_status status = rocblas_status_success;

    if(side == rocblas_side_left)
    {
        if(uplo == rocblas_fill_upper)
        {
            if(transa == rocblas_operation_none)
            {
                //
                //              Form  C := alpha*A*C. Left, Upper, No transpose.
                //
                bool cldc = dcld(ldc);
                for(int ii = 1; ii <= m; ii += cb)
                {
                    isec = cb < m - ii + 1 ? cb : m - ii + 1;
                    //
                    //                  T2 := A', the transpose of a upper unit or non-unit
                    //                  triangular diagonal block of A is copied to the
                    //                  lower triangular part of T2.
                    //
                    for(int i = ii + offd; i <= ii + isec - 1; i++)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_copy_template<NB, T>)(handle,
                                                           i - ii + 1 - offd,
                                                           &a[ii - 1 + (i - 1) * lda],
                                                           0,
                                                           1,
                                                           0,
                                                           &dt2[i - ii],
                                                           0,
                                                           cb,
                                                           0,
                                                           1));
                    }
                    for(int jj = 1; jj <= n; jj += rb)
                    {
                        jsec = rb < n - jj + 1 ? rb : n - jj + 1;
                        //
                        //                      T1 := C', the transpose of a rectangular block
                        //                      of C is copied to T1.
                        //
                        if(cldc)
                        {
                            for(int j = jj; j <= jj + jsec - 1; j++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   isec,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   &dt1[j - jj],
                                                                   0,
                                                                   rb,
                                                                   0,
                                                                   1));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &c[i - 1 + (jj - 1) * ldc],
                                                                   0,
                                                                   ldc,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                        }
                        //
                        //                      T1 := alpha*T1*T2 + delta*T1, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether T2 stores a unit or non-unit triangular
                        //                      block. Gamma and tsec are used to compensate for
                        //                      a deficiency in DGEMV that appears if the second
                        //                      dimension (tsec) is zero.
                        //
                        for(int i = ii; i <= ii + isec - 1; i++)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &dt2[i - ii + (i - ii) * ldt2],
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = ii + isec - 1 - i;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               jsec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               &dt1[(i - ii + 1) * ldt1],
                                                               0,
                                                               rb,
                                                               0,
                                                               &dt2[i - ii + 1 + (i - ii) * ldt2],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &dt1[(i - ii) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &dt1[j - jj],
                                                               0,
                                                               rb,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                    }
                    //
                    //                  C := alpha*A*C + C, general matrix multiply
                    //                  involving a rectangular block of A.
                    //
                    if(ii + isec <= m)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_none,
                                                                  rocblas_operation_none,
                                                                  isec,
                                                                  n,
                                                                  m - ii - isec + 1,
                                                                  alpha,
                                                                  (const T*)(&a[ii - 1
                                                                                + (ii + isec - 1)
                                                                                      * lda]),
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  (const T*)(&c[ii + isec - 1]),
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  &one,
                                                                  &c[ii - 1],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
            else
            {
                //
                //             Form  C := alpha*A'*C. Left, Upper, Transpose.
                //
                bool cldc = dcld(ldc);
                for(int ii = m - ((m - 1) % cb); ii >= 1; ii -= cb)
                {
                    isec = cb < m - ii + 1 ? cb : m - ii + 1;
                    for(int jj = 1; jj <= n; jj += rb)
                    {
                        jsec = rb < n - jj + 1 ? rb : n - jj + 1;
                        //
                        //                      T1 := C', the transpose of a rectangular block
                        //                      of C is copied to T1.
                        //
                        if(cldc)
                        {
                            for(int j = jj; j <= jj + jsec - 1; j++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   isec,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   &dt1[j - jj],
                                                                   0,
                                                                   rb,
                                                                   0,
                                                                   1));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &c[i - 1 + (jj - 1) * ldc],
                                                                   0,
                                                                   ldc,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                        }
                        //
                        //                      T1 := alpha*T1*A + delta*T1, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether A is a unit or non-unit triangular
                        //                      matrix. Gamma and tsec are used to compensate
                        //                      for a deficiency in DGEMV that appears if the
                        //                      second dimension (tsec) is zero.
                        //
                        for(int i = ii + isec - 1; i >= ii; i--)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &a[i - 1 + (i - 1) * lda],
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = ii + isec - 1 - i;
                            tsec = i - ii;
                            if(0 == tsec)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               jsec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               0,
                                                               rb,
                                                               0,
                                                               &a[ii - 1 + (i - 1) * lda],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &dt1[(i - ii) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &dt1[j - jj],
                                                               0,
                                                               rb,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                    }
                    //
                    //                   C := alpha*A'*C + C, general matrix multiply
                    //                   involving the transpose of a rectangular block
                    //                   of A.
                    //
                    if(ii > 1)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_transpose,
                                                                  rocblas_operation_none,
                                                                  isec,
                                                                  n,
                                                                  ii - 1,
                                                                  alpha,
                                                                  (const T*)&a[(ii - 1) * lda],
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  (const T*)c,
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  &one,
                                                                  &c[ii - 1],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
        }
        else
        {
            if(transa == rocblas_operation_none)
            {
                //
                //             Form  C := alpha*A*C. Left, Lower, No transpose.
                //
                bool cldc = dcld(ldc);
                for(int ix = m; ix >= 1; ix -= cb)
                {
                    rocblas_int ii = 1 > ix - cb + 1 ? 1 : ix - cb + 1;
                    isec           = ix - ii + 1;
                    //
                    //                   T2 := A', the transpose of a lower unit or non-unit
                    //                   triangular diagonal block of A is copied to the
                    //                   upper triangular part of T2.
                    //
                    for(int i = ii; i <= ii + isec - 1 - offd; i++)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_copy_template<NB, T>)(handle,
                                                           ii + isec - i - offd,
                                                           &a[i + offd - 1 + (i - 1) * lda],
                                                           0,
                                                           1,
                                                           0,
                                                           &dt2[i - ii + (i - ii + offd) * ldt2],
                                                           0,
                                                           cb,
                                                           0,
                                                           1));
                    }
                    for(int jj = 1; jj <= n; jj += rb)
                    {
                        jsec = rb < n - jj + 1 ? rb : n - jj + 1;
                        //
                        //                      T1 := C', the transpose of a rectangular block
                        //                      of C is copied to T1.
                        //
                        if(cldc)
                        {
                            for(int j = jj; j <= jj + jsec - 1; j++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   isec,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   &dt1[j - jj],
                                                                   0,
                                                                   rb,
                                                                   0,
                                                                   1));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &c[i - 1 + (jj - 1) * ldc],
                                                                   0,
                                                                   ldc,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                        }
                        //
                        //                      T1 := alpha*T1*T2 + delta*T1, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether T2 stores a unit or non-unit triangular
                        //                      block. Gamma and tsec are used to compensate for
                        //                      a deficiency in DGEMV that appears if the second
                        //                      dimension (tsec) is zero.
                        //
                        for(int i = ii + isec - 1; i >= ii; i--)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &dt2[i - ii + (i - ii) * ldt2],
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = i - ii;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               jsec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               0,
                                                               rb,
                                                               0,
                                                               &dt2[(i - ii) * ldt2],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &dt1[(i - ii) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &dt1[j - jj],
                                                               0,
                                                               rb,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                    }
                    //
                    //                   C := alpha*A'*C + C, general matrix multiply
                    //                   involving a rectangular block of A.
                    //
                    if(ii > 1)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_none,
                                                                  rocblas_operation_none,
                                                                  isec,
                                                                  n,
                                                                  ii - 1,
                                                                  alpha,
                                                                  (const T*)&a[ii - 1],
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  (const T*)c,
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  &one,
                                                                  &c[ii - 1],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
            else
            {
                //
                //              Form  C := alpha*A'*C. Left, Lower, Transpose.
                //
                bool cldc = dcld(ldc);
                for(int ix = ((m - 1) % cb) + 1; ix <= m; ix += cb)
                {
                    rocblas_int ii = 1 > ix - cb + 1 ? 1 : ix - cb + 1;
                    isec           = ix - ii + 1;
                    for(int jj = 1; jj <= n; jj += rb)
                    {
                        jsec = rb < n - jj + 1 ? rb : n - jj + 1;
                        //
                        //                      T1 := C', the transpose of a rectangular block
                        //                      of C is copied to T1.
                        //
                        if(cldc)
                        {
                            for(int j = jj; j <= jj + jsec - 1; j++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   isec,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   &dt1[j - jj],
                                                                   0,
                                                                   rb,
                                                                   0,
                                                                   1));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &c[i - 1 + (jj - 1) * ldc],
                                                                   0,
                                                                   ldc,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                        }
                        //
                        //                      T1 := alpha*T1*A + delta*T1, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether A is a unit or non-unit triangular
                        //                      matrix. Gamma and tsec are used to compensate
                        //                      for a deficiency in DGEMV that appears if the
                        //                      second dimension (tsec) is zero.
                        //
                        for(int i = ii; i <= ii + isec - 1; i++)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   &a[i - 1 + (i - 1) * lda],
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = ii + isec - 1 - i;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   0,
                                                                   &dt1[(i - ii) * ldt1],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               jsec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               &dt1[(i - ii + 1) * ldt1],
                                                               0,
                                                               rb,
                                                               0,
                                                               &a[i + (i - 1) * lda],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &dt1[(i - ii) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &dt1[j - jj],
                                                               0,
                                                               rb,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                    }
                    //
                    //                  C := alpha*A'*C + C, general matrix multiply
                    //                  involving the transpose of a rectangular block
                    //                  of A.
                    //
                    if(ii + isec <= m)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_transpose,
                                                                  rocblas_operation_none,
                                                                  isec,
                                                                  n,
                                                                  m - ii - isec + 1,
                                                                  alpha,
                                                                  (const T*)&a[ii + isec - 1
                                                                               + (ii - 1) * lda],
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  (const T*)&c[ii + isec - 1],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  &one,
                                                                  &c[ii - 1],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
        }
    }
    else
    {
        if(uplo == rocblas_fill_upper)
        {
            if(transa == rocblas_operation_none)
            {
                //
                //              Form  C := alpha*C*A. Right, Upper, No transpose.
                //
                for(int jj = n - (n - 1) % cb; jj >= 1; jj -= cb)
                {
                    jsec = cb < n - jj + 1 ? cb : n - jj + 1;
                    for(int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m - ii + 1 ? rb : m - ii + 1;
                        //
                        //                      T1 := C, a rectangular block of C is copied
                        //                      to T1.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               &dt1[(j - jj) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                        //
                        //                      C := alpha*T1*A + delta*C, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether A is a unit or non-unit triangular
                        //                      matrix. Gamma and tsec are used to compensate
                        //                      for a deficiency in DGEmV that appears if the
                        //                      second dimension (tsec) is zero.
                        //
                        for(int j = jj + jsec - 1; j >= jj; j--)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   &a[j - 1 + (j - 1) * lda],
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = j - jj;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               isec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               0,
                                                               rb,
                                                               0,
                                                               &a[jj - 1 + (j - 1) * lda],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                    }
                    //
                    //                  C := alpha*C*A + C, general matrix multiply
                    //                  involving a rectangular block of A.
                    //
                    if(jj > 1)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_none,
                                                                  rocblas_operation_none,
                                                                  m,
                                                                  jsec,
                                                                  jj - 1,
                                                                  alpha,
                                                                  (const T*)c,
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  (const T*)&a[(jj - 1) * lda],
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  &one,
                                                                  &c[(jj - 1) * ldc],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
            else
            {
                //
                //              Form  C := alpha*C*A'. Right, Upper, Transpose.
                //
                for(int jj = 1; jj <= n; jj += cb)
                {
                    jsec = cb < n - jj + 1 ? cb : n - jj + 1;
                    //
                    //                  T2 := A', the transpose of a upper unit or non-unit
                    //                  triangular diagonal block of A is copied to the
                    //                  lower triangular part of T2.
                    //
                    for(int j = jj + offd; j <= jj + jsec - 1; j++)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_copy_template<NB, T>)(handle,
                                                           j - jj + 1 - offd,
                                                           &a[jj - 1 + (j - 1) * lda],
                                                           0,
                                                           1,
                                                           0,
                                                           &dt2[j - jj],
                                                           0,
                                                           cb,
                                                           0,
                                                           1));
                    }
                    for(int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m - ii + 1 ? rb : m - ii + 1;
                        //
                        //                      T1 := C, a rectangular block of C is copied
                        //                      to T1.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               &dt1[(j - jj) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                        //
                        //                      C := alpha*T1*T2 + delta*C, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether T2 is a unit or non-unit triangular
                        //                      matrix. Gamma and tsec are used to compensate
                        //                      for a deficiency in DGEmV that appears if the
                        //                      second dimension (tsec) is zero.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   &dt2[j - jj + (j - jj) * ldt2],
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = jj + jsec - 1 - j;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               isec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               &dt1[(j - jj + 1) * ldt1],
                                                               0,
                                                               rb,
                                                               0,
                                                               &dt2[j - jj + 1 + (j - jj) * ldt2],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                    }
                    //
                    //                  C := alpha*C*A' + C, general matrix multiply
                    //                  involving the transpose of a rectangular block
                    //                  of A.
                    //
                    if(jj + jsec <= n)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false,
                                                   false>)(handle,
                                                           rocblas_operation_none,
                                                           rocblas_operation_transpose,
                                                           m,
                                                           jsec,
                                                           n - jj - jsec + 1,
                                                           alpha,
                                                           (const T*)&c[(jj + jsec - 1) * ldc],
                                                           0,
                                                           ldc,
                                                           0,
                                                           (const T*)&a[jj - 1
                                                                        + (jj + jsec - 1) * lda],
                                                           0,
                                                           lda,
                                                           0,
                                                           &one,
                                                           &c[(jj - 1) * ldc],
                                                           0,
                                                           ldc,
                                                           0,
                                                           1));
                    }
                }
            }
        }
        else
        {
            if(transa == rocblas_operation_none)
            {
                //
                //              Form  C := alpha*C*A. Right, Lower, No transpose.
                //
                for(int jx = ((n - 1) % cb) + 1; jx <= n; jx += cb)
                {
                    rocblas_int jj = 1 > jx - cb + 1 ? 1 : jx - cb + 1;
                    jsec           = jx - jj + 1;
                    for(int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m - ii + 1 ? rb : m - ii + 1;
                        //
                        //                      T1 := C, a rectangular block of C is copied
                        //                      to T1.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               &dt1[(j - jj) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                        //
                        //                      C := alpha*T1*A + delta*C, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether A is a unit or non-unit triangular
                        //                      matrix. Gamma and tsec are used to compensate
                        //                      for a deficiency in DGEmV that appears if the
                        //                      second dimension (tsec) is zero.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   &a[j - 1 + (j - 1) * lda],
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = jj + jsec - 1 - j;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               isec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               &dt1[(j - jj + 1) * ldt1],
                                                               0,
                                                               rb,
                                                               0,
                                                               &a[j + (j - 1) * lda],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                    }
                    //
                    //                   C := alpha*C*A + C, general matrix multiply
                    //                   involving a rectangular block of A.
                    //
                    if(jj + jsec <= n)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_none,
                                                                  rocblas_operation_none,
                                                                  m,
                                                                  jsec,
                                                                  n - jj - jsec + 1,
                                                                  alpha,
                                                                  (const T*)&c[(jj + jsec - 1)
                                                                               * ldc],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  (const T*)&a[jj + jsec - 1
                                                                               + (jj - 1) * lda],
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  &one,
                                                                  &c[(jj - 1) * ldc],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
            else
            {
                //
                //              Form  C := alpha*C*A'. Right, Lower, Transpose.
                //
                for(int jx = n; jx >= 1; jx -= cb)
                {
                    rocblas_int jj = 1 > jx - cb + 1 ? 1 : jx - cb + 1;
                    jsec           = jx - jj + 1;
                    //
                    //                  T2 := A', the transpose of a lower unit or non-unit
                    //                  triangular diagonal block of A is copied to the
                    //                  upper triangular part of T2.
                    //
                    for(int j = jj; j <= jj + jsec - 1 - offd; j++)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_copy_template<NB, T>)(handle,
                                                           jj + jsec - j - offd,
                                                           &a[j + offd - 1 + (j - 1) * lda],
                                                           0,
                                                           1,
                                                           0,
                                                           &dt2[j - jj + (j - jj + offd) * ldt2],
                                                           0,
                                                           cb,
                                                           0,
                                                           1));
                    }
                    for(int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m - ii + 1 ? rb : m - ii + 1;
                        //
                        //                      T1 := C, a rectangular block of C is copied
                        //                      to T1.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<NB, T>)(handle,
                                                               isec,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               &dt1[(j - jj) * ldt1],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                        }
                        //
                        //                      C := alpha*T1*T2 + delta*C, triangular matrix
                        //                      multiply where the value of delta depends on
                        //                      whether T2 is a unit or non-unit triangular
                        //                      matrix. Gamma and tsec are used to compensate
                        //                      for a deficiency in DGEmV that appears if the
                        //                      second dimension (tsec) is zero.
                        //
                        for(int j = jj + jsec - 1; j >= jj; j--)
                        {
                            if(diag == rocblas_diagonal_non_unit)
                            {
                                auto saved_pointer_mode
                                    = handle->push_pointer_mode(rocblas_pointer_mode_device);
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   &dt2[j - jj + (j - jj) * ldt2],
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            tsec = j - jj;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   &c[ii - 1 + (j - 1) * ldc],
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   1));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<T>)(handle,
                                                               rocblas_operation_none,
                                                               isec,
                                                               tsec,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               0,
                                                               rb,
                                                               0,
                                                               &dt2[(j - jj) * ldt2],
                                                               0,
                                                               1,
                                                               0,
                                                               alpha,
                                                               0,
                                                               &c[ii - 1 + (j - 1) * ldc],
                                                               0,
                                                               1,
                                                               0,
                                                               1));
                            }
                        }
                    }
                    //
                    //                  C := alpha*C*A' + C, general matrix multiply involving the transpose of a rectangular block of A.
                    //
                    if(jj > 1)
                    {
                        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_none,
                                                                  rocblas_operation_transpose,
                                                                  m,
                                                                  jsec,
                                                                  jj - 1,
                                                                  alpha,
                                                                  (const T*)c,
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  (const T*)&a[jj - 1],
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  &one,
                                                                  &c[(jj - 1) * ldc],
                                                                  0,
                                                                  ldc,
                                                                  0,
                                                                  1));
                    }
                }
            }
        }
    }

    return rocblas_status_success;
}
