/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "../blas1/rocblas_copy.hpp"
#include "../blas1/rocblas_scal.hpp"
#include "../blas2/rocblas_gemv.hpp"
#include "../blas3/Tensile/gemm.hpp"
#include "dcld.hpp"
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

template <typename TScal, typename TPtr>
__global__ void set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                                     rocblas_int    n,
                                                     TScal          alpha_device_host,
                                                     rocblas_stride stride_alpha,
                                                     TPtr           Aa,
                                                     ptrdiff_t      offsetA,
                                                     rocblas_int    lda,
                                                     rocblas_int    strideA)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);

    if(tx < m && ty < n && alpha == 0)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offsetA, strideA);

        A[tx + lda * ty] = 0;
    }
}

template <typename TScal, typename TPtr>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TPtr           A,
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
        hipLaunchKernelGGL(set_matrix_zero_if_alpha_zero_kernel,
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
        hipLaunchKernelGGL(set_matrix_zero_if_alpha_zero_kernel,
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

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <rocblas_int RB,
          rocblas_int CB,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_trmm_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation transa,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal             alpha,
                                     TConstPtr         a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TPtr              c,
                                     rocblas_int       ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count,
                                     TPtr              workspace,
                                     rocblas_stride    stride_w)
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
        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(set_matrix_zero_if_alpha_zero_template(
            handle, m, n, alpha, 0, c, 0, ldc, 0, batch_count));
        return rocblas_status_success;
    }
    else if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // set matrix to zero and continue calculation. This will give
        // the same functionality as Legacy BLAS. alpha is on device and
        // it should not be copied from device to host because this is
        // an asynchronous function and the copy would make it synchronous.
        PRINT_AND_RETURN_IF_ROCBLAS_ERROR(set_matrix_zero_if_alpha_zero_template(
            handle, m, n, alpha, 0, c, 0, ldc, 0, batch_count));
    }

    // grid size for rocblas_copy_template
    constexpr rocblas_int NB = 256;

    // assign space for dt1 and dt2
    rocblas_int rb = RB, cb = CB;
    rocblas_int ldt1 = rb, ldt2 = cb;
    TPtr        dt1 = workspace;
    TPtr        dt2 = workspace + rb * cb;

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
                            (rocblas_copy_template<false, NB>)(handle,
                                                               i - ii + 1 - offd,
                                                               a,
                                                               ii - 1 + (i - 1) * lda,
                                                               1,
                                                               stride_a,
                                                               dt2,
                                                               i - ii,
                                                               cb,
                                                               stride_w,
                                                               batch_count));
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
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       isec,
                                                                       c,
                                                                       ii - 1 + (j - 1) * ldc,
                                                                       1,
                                                                       stride_c,
                                                                       dt1,
                                                                       j - jj,
                                                                       rb,
                                                                       stride_w,
                                                                       batch_count));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       jsec,
                                                                       c,
                                                                       i - 1 + (jj - 1) * ldc,
                                                                       ldc,
                                                                       stride_c,
                                                                       dt1,
                                                                       (i - ii) * ldt1,
                                                                       1,
                                                                       stride_w,
                                                                       batch_count));
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
                                                                   dt2,
                                                                   i - ii + (i - ii) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            tsec = ii + isec - 1 - i;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   1,
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               (i - ii + 1) * ldt1,
                                                               rb,
                                                               stride_w,
                                                               dt2,
                                                               i - ii + 1 + (i - ii) * ldt2,
                                                               1,
                                                               stride_w,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               (i - ii) * ldt1,
                                                               1,
                                                               stride_w,
                                                               batch_count));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   dt1,
                                                                   j - jj,
                                                                   rb,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   batch_count));
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
                                                                  a,
                                                                  ii - 1 + (ii + isec - 1) * lda,
                                                                  lda,
                                                                  stride_a,
                                                                  (TConstPtr)c,
                                                                  ii + isec - 1,
                                                                  ldc,
                                                                  stride_c,
                                                                  &one,
                                                                  c,
                                                                  ii - 1,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                    //
                    //                   T2 := A or T2 := conjg( A ), a unit or non-unit
                    //                   upper triangular diagonal block of A is copied to
                    //                   the upper triangular part of T2.
                    //
                    for(int j = ii + offd; j <= ii + isec - 1; j++)
                    {
                        if(transa == rocblas_operation_conjugate_transpose)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<true, NB>)(handle,
                                                                  j - ii + 1 - offd,
                                                                  a,
                                                                  ii - 1 + (j - 1) * lda,
                                                                  1,
                                                                  stride_a,
                                                                  dt2,
                                                                  (j - ii) * ldt2,
                                                                  1,
                                                                  stride_w,
                                                                  batch_count));
                        }
                        else
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   j - ii + 1 - offd,
                                                                   a,
                                                                   ii - 1 + (j - 1) * lda,
                                                                   1,
                                                                   stride_a,
                                                                   dt2,
                                                                   (j - ii) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                        }
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
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       isec,
                                                                       c,
                                                                       ii - 1 + (j - 1) * ldc,
                                                                       1,
                                                                       stride_c,
                                                                       dt1,
                                                                       j - jj,
                                                                       rb,
                                                                       stride_w,
                                                                       batch_count));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       jsec,
                                                                       c,
                                                                       i - 1 + (jj - 1) * ldc,
                                                                       ldc,
                                                                       stride_c,
                                                                       dt1,
                                                                       (i - ii) * ldt1,
                                                                       1,
                                                                       stride_w,
                                                                       batch_count));
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
                                                                   dt2,
                                                                   i - ii + (i - ii) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               stride_w,
                                                               dt2,
                                                               (i - ii) * ldt2,
                                                               1,
                                                               stride_w,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               (i - ii) * ldt1,
                                                               1,
                                                               stride_w,
                                                               batch_count));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   dt1,
                                                                   j - jj,
                                                                   rb,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   batch_count));
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
                                                                  transa,
                                                                  rocblas_operation_none,
                                                                  isec,
                                                                  n,
                                                                  ii - 1,
                                                                  alpha,
                                                                  a,
                                                                  (ii - 1) * lda,
                                                                  lda,
                                                                  stride_a,
                                                                  (TConstPtr)c,
                                                                  0,
                                                                  ldc,
                                                                  stride_c,
                                                                  &one,
                                                                  c,
                                                                  ii - 1,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                            (rocblas_copy_template<false, NB>)(handle,
                                                               ii + isec - i - offd,
                                                               a,
                                                               i + offd - 1 + (i - 1) * lda,
                                                               1,
                                                               stride_a,
                                                               dt2,
                                                               i - ii + (i - ii + offd) * ldt2,
                                                               cb,
                                                               stride_w,
                                                               batch_count));
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
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       isec,
                                                                       c,
                                                                       ii - 1 + (j - 1) * ldc,
                                                                       1,
                                                                       stride_c,
                                                                       dt1,
                                                                       j - jj,
                                                                       rb,
                                                                       stride_w,
                                                                       batch_count));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       jsec,
                                                                       c,
                                                                       i - 1 + (jj - 1) * ldc,
                                                                       ldc,
                                                                       stride_c,
                                                                       dt1,
                                                                       (i - ii) * ldt1,
                                                                       1,
                                                                       stride_w,
                                                                       batch_count));
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
                                                                   dt2,
                                                                   i - ii + (i - ii) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            tsec = i - ii;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   0,
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               stride_w,
                                                               dt2,
                                                               (i - ii) * ldt2,
                                                               1,
                                                               stride_w,
                                                               alpha,
                                                               0,
                                                               dt1,
                                                               (i - ii) * ldt1,
                                                               1,
                                                               stride_w,
                                                               batch_count));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   &dt1[j - jj],
                                                                   0,
                                                                   rb,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   batch_count));
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
                                                                  a,
                                                                  ii - 1,
                                                                  lda,
                                                                  stride_a,
                                                                  (TConstPtr)c,
                                                                  0,
                                                                  ldc,
                                                                  stride_c,
                                                                  &one,
                                                                  c,
                                                                  ii - 1,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                    //
                    //    T2 := A or T2 := conjg( A ), a unit or non-unit
                    //    lower triangular diagonal block of A is copied to
                    //    the lower triangular part of T2.
                    //
                    for(int j = ii; j <= ii + isec - 1 - offd; j++)
                    {
                        if(transa == rocblas_operation_conjugate_transpose)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<true, NB>)(handle,
                                                                  ii + isec - j - offd,
                                                                  a,
                                                                  j + offd - 1 + (j - 1) * lda,
                                                                  1,
                                                                  stride_a,
                                                                  dt2,
                                                                  j - ii + offd + (j - ii) * ldt2,
                                                                  1,
                                                                  stride_w,
                                                                  batch_count));
                        }
                        else
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   ii + isec - j - offd,
                                                                   a,
                                                                   j + offd - 1 + (j - 1) * lda,
                                                                   1,
                                                                   stride_a,
                                                                   dt2,
                                                                   j - ii + offd + (j - ii) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                        }
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
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       isec,
                                                                       c,
                                                                       ii - 1 + (j - 1) * ldc,
                                                                       1,
                                                                       stride_c,
                                                                       dt1,
                                                                       j - jj,
                                                                       rb,
                                                                       stride_w,
                                                                       batch_count));
                            }
                        }
                        else
                        {
                            for(int i = ii; i <= ii + isec - 1; i++)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_copy_template<false, NB>)(handle,
                                                                       jsec,
                                                                       c,
                                                                       i - 1 + (jj - 1) * ldc,
                                                                       ldc,
                                                                       stride_c,
                                                                       dt1,
                                                                       (i - ii) * ldt1,
                                                                       1,
                                                                       stride_w,
                                                                       batch_count));
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
                                    (rocblas_scal_template<
                                        NB,
                                        T>)(handle,
                                            jsec,
                                            //                                                                 &a[i - 1 + (i - 1) * lda],
                                            dt2,
                                            i - ii + (i - ii) * ldt2,
                                            1,
                                            stride_w,
                                            dt1,
                                            (i - ii) * ldt1,
                                            1,
                                            stride_w,
                                            batch_count));
                            }
                            tsec = ii + isec - 1 - i;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   jsec,
                                                                   alpha,
                                                                   0,
                                                                   dt1,
                                                                   (i - ii) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            else
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_gemv_template<
                                        T>)(handle,
                                            rocblas_operation_none,
                                            jsec,
                                            tsec,
                                            alpha,
                                            0,
                                            dt1,
                                            (i - ii + 1) * ldt1,
                                            rb,
                                            stride_w,
                                            //                                                             &a[i + (i - 1) * lda],
                                            dt2,
                                            i - ii + 1 + (i - ii) * ldt2,
                                            1,
                                            stride_w,
                                            alpha,
                                            0,
                                            dt1,
                                            (i - ii) * ldt1,
                                            1,
                                            stride_w,
                                            batch_count));
                            }
                        }
                        //
                        //                      C := T1', the transpose of T1 is copied back
                        //                      to C.
                        //
                        for(int j = jj; j <= jj + jsec - 1; j++)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   dt1,
                                                                   j - jj,
                                                                   rb,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   batch_count));
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
                                                                  transa,
                                                                  rocblas_operation_none,
                                                                  isec,
                                                                  n,
                                                                  m - ii - isec + 1,
                                                                  alpha,
                                                                  a,
                                                                  ii + isec - 1 + (ii - 1) * lda,
                                                                  lda,
                                                                  stride_a,
                                                                  (TConstPtr)c,
                                                                  ii + isec - 1,
                                                                  ldc,
                                                                  stride_c,
                                                                  &one,
                                                                  &c[ii - 1],
                                                                  0,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   dt1,
                                                                   (j - jj) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                                   a,
                                                                   j - 1 + (j - 1) * lda,
                                                                   1,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            tsec = j - jj;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               (TConstPtr)dt1,
                                                               0,
                                                               rb,
                                                               stride_w,
                                                               a,
                                                               jj - 1 + (j - 1) * lda,
                                                               1,
                                                               stride_a,
                                                               alpha,
                                                               0,
                                                               c,
                                                               ii - 1 + (j - 1) * ldc,
                                                               1,
                                                               stride_c,
                                                               batch_count));
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
                                                                  (TConstPtr)c,
                                                                  0,
                                                                  ldc,
                                                                  stride_c,
                                                                  a,
                                                                  (jj - 1) * lda,
                                                                  lda,
                                                                  stride_a,
                                                                  &one,
                                                                  c,
                                                                  (jj - 1) * ldc,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                        if(transa == rocblas_operation_conjugate_transpose)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<true, NB>)(handle,
                                                                  j - jj + 1 - offd,
                                                                  a,
                                                                  jj - 1 + (j - 1) * lda,
                                                                  1,
                                                                  stride_a,
                                                                  dt2,
                                                                  j - jj,
                                                                  cb,
                                                                  stride_w,
                                                                  batch_count));
                        }
                        else
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   j - jj + 1 - offd,
                                                                   a,
                                                                   jj - 1 + (j - 1) * lda,
                                                                   1,
                                                                   stride_a,
                                                                   dt2,
                                                                   j - jj,
                                                                   cb,
                                                                   stride_w,
                                                                   batch_count));
                        }
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
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   dt1,
                                                                   (j - jj) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                                   dt2,
                                                                   j - jj + (j - jj) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            tsec = jj + jsec - 1 - j;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               (j - jj + 1) * ldt1,
                                                               rb,
                                                               stride_w,
                                                               dt2,
                                                               j - jj + 1 + (j - jj) * ldt2,
                                                               1,
                                                               stride_w,
                                                               alpha,
                                                               0,
                                                               c,
                                                               ii - 1 + (j - 1) * ldc,
                                                               1,
                                                               stride_c,
                                                               batch_count));
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
                            (rocblas_gemm_template<false, false>)(handle,
                                                                  rocblas_operation_none,
                                                                  transa,
                                                                  m,
                                                                  jsec,
                                                                  n - jj - jsec + 1,
                                                                  alpha,
                                                                  (TConstPtr)c,
                                                                  (jj + jsec - 1) * ldc,
                                                                  ldc,
                                                                  stride_c,
                                                                  a,
                                                                  jj - 1 + (jj + jsec - 1) * lda,
                                                                  lda,
                                                                  stride_a,
                                                                  &one,
                                                                  c,
                                                                  (jj - 1) * ldc,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   dt1,
                                                                   (j - jj) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                                   a,
                                                                   j - 1 + (j - 1) * lda,
                                                                   1,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            tsec = jj + jsec - 1 - j;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               (TConstPtr)dt1,
                                                               (j - jj + 1) * ldt1,
                                                               rb,
                                                               stride_w,
                                                               a,
                                                               j + (j - 1) * lda,
                                                               1,
                                                               stride_a,
                                                               alpha,
                                                               0,
                                                               c,
                                                               ii - 1 + (j - 1) * ldc,
                                                               1,
                                                               stride_c,
                                                               batch_count));
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
                                                                  (TConstPtr)c,
                                                                  (jj + jsec - 1) * ldc,
                                                                  ldc,
                                                                  stride_c,
                                                                  a,
                                                                  jj + jsec - 1 + (jj - 1) * lda,
                                                                  lda,
                                                                  stride_a,
                                                                  &one,
                                                                  c,
                                                                  (jj - 1) * ldc,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
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
                    // noconj
                    for(int j = jj; j <= jj + jsec - 1 - offd; j++)
                    {
                        if(transa == rocblas_operation_conjugate_transpose)
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<true, NB>)(handle,
                                                                  jj + jsec - j - offd,
                                                                  a,
                                                                  j + offd - 1 + (j - 1) * lda,
                                                                  1,
                                                                  stride_a,
                                                                  dt2,
                                                                  j - jj + (j - jj + offd) * ldt2,
                                                                  cb,
                                                                  stride_w,
                                                                  batch_count));
                        }
                        else
                        {
                            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   jj + jsec - j - offd,
                                                                   a,
                                                                   j + offd - 1 + (j - 1) * lda,
                                                                   1,
                                                                   stride_a,
                                                                   dt2,
                                                                   j - jj + (j - jj + offd) * ldt2,
                                                                   cb,
                                                                   stride_w,
                                                                   batch_count));
                        }
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
                                (rocblas_copy_template<false, NB>)(handle,
                                                                   isec,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_c,
                                                                   dt1,
                                                                   (j - jj) * ldt1,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                                   dt2,
                                                                   j - jj + (j - jj) * ldt2,
                                                                   1,
                                                                   stride_w,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
                            }
                            tsec = j - jj;
                            if(tsec == 0)
                            {
                                PRINT_AND_RETURN_IF_ROCBLAS_ERROR(
                                    (rocblas_scal_template<NB, T>)(handle,
                                                                   isec,
                                                                   alpha,
                                                                   0,
                                                                   c,
                                                                   ii - 1 + (j - 1) * ldc,
                                                                   1,
                                                                   stride_w,
                                                                   batch_count));
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
                                                               stride_w,
                                                               dt2,
                                                               (j - jj) * ldt2,
                                                               1,
                                                               stride_w,
                                                               alpha,
                                                               0,
                                                               c,
                                                               ii - 1 + (j - 1) * ldc,
                                                               1,
                                                               stride_c,
                                                               batch_count));
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
                                                                  transa,
                                                                  m,
                                                                  jsec,
                                                                  jj - 1,
                                                                  alpha,
                                                                  (TConstPtr)c,
                                                                  0,
                                                                  ldc,
                                                                  stride_c,
                                                                  a,
                                                                  jj - 1,
                                                                  lda,
                                                                  stride_a,
                                                                  &one,
                                                                  c,
                                                                  (jj - 1) * ldc,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count));
                    }
                }
            }
        }
    }

    return rocblas_status_success;
}
