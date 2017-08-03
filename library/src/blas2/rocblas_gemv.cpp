/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
 
#include "status.h"
#include "definitions.h"
#include "gemv_device.h"

template<typename T, const rocblas_int NB_X, const rocblas_int NB_Y>
__global__ void
gemvn_kernel_host_pointer(hipLaunchParm lp,
    rocblas_int m, rocblas_int n,
    const T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ x, rocblas_int incx,
    const T beta,
    T       *y, rocblas_int incy)
{
    gemvn_device<T, NB_X, NB_Y>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename T, const rocblas_int NB_X, const rocblas_int NB_Y>
__global__ void
gemvn_kernel_device_pointer(hipLaunchParm lp,
    rocblas_int m, rocblas_int n,
    const T* alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ x, rocblas_int incx,
    const T* beta,
    T       *y, rocblas_int incy)
{
    gemvn_device<T, NB_X, NB_Y>(m, n, *alpha, A, lda, x, incx, *beta, y, incy);
}

template<typename T, const rocblas_int NB_X>
__global__ void
gemvc_kernel_host_pointer(hipLaunchParm lp,
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    const T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ x, rocblas_int incx,
    const T beta,
    T       * y, rocblas_int incy)
{
    gemvc_device<T, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename T, const rocblas_int NB_X>
__global__ void
gemvc_kernel_device_pointer(hipLaunchParm lp,
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    const T* alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ x, rocblas_int incx,
    const T* beta,
    T       * y, rocblas_int incy)
{
    gemvc_device<T, NB_X>(m, n, *alpha, A, lda, x, incx, *beta, y, incy);
}

/*! \brief BLAS Level 2 API

    \details
    xGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    trans     rocblas_operation
    @param[in]
    m         rocblas_int
    @param[in]
    n         rocblas_int
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[in]
    beta      specifies the scalar beta.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template<typename T>
rocblas_status
rocblas_gemv_template(rocblas_handle handle,
    rocblas_operation transA, rocblas_int m, rocblas_int n,
    const T *alpha,
    const T *A, rocblas_int lda,
    const T *x, rocblas_int incx,
    const T *beta,
    T * y, rocblas_int incy)
{
    if (nullptr == A)
        return rocblas_status_invalid_pointer;
    else if (nullptr == x)
        return rocblas_status_invalid_pointer;
    else if (nullptr == y)
        return rocblas_status_invalid_pointer;
    else if (nullptr == beta)
        return rocblas_status_invalid_pointer;
    else if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if (m < 0)
        return rocblas_status_invalid_size;
    else if (n < 0)
        return rocblas_status_invalid_size;
    else if (lda < m || lda < 1)
        return rocblas_status_invalid_size;
    else if (0 == incx)
        return rocblas_status_invalid_size;
    else if (0 == incy)
        return rocblas_status_invalid_size;

//  TODO: remove this restriction. See reference implemention
//  if (incx < 0 || incy < 0)
//      return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */

    if (0 == m || 0 == n)
        return rocblas_status_success;

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    if ( transA == rocblas_operation_none ) {
        #define  GEMVN_DIM_X 64 //
        #define  GEMVN_DIM_Y 16 //GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        rocblas_int blocks = (m-1)/(GEMVN_DIM_X * 4) + 1;

        dim3 gemvn_grid( blocks, 1, 1 );
        dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y, 1 );

        if( rocblas_pointer_to_mode((void*)alpha) == rocblas_pointer_mode_device && rocblas_pointer_to_mode((void*)beta) == rocblas_pointer_mode_device ){
            hipLaunchKernel(HIP_KERNEL_NAME(gemvn_kernel_device_pointer<T, GEMVN_DIM_X, GEMVN_DIM_Y>), dim3(gemvn_grid), dim3(gemvn_threads), 0, rocblas_stream,
                                            m, n, alpha, A, lda, x, incx, beta, y, incy);
        }
        else{
            if ( 0.0 == *alpha && 1.0 == *beta) return rocblas_status_success;
            T h_alpha_scalar = *alpha; T h_beta_scalar = *beta;
            hipLaunchKernel(HIP_KERNEL_NAME(gemvn_kernel_host_pointer<T, GEMVN_DIM_X, GEMVN_DIM_Y>), dim3(gemvn_grid), dim3(gemvn_threads), 0, rocblas_stream,
                                            m, n, h_alpha_scalar, A, lda, x, incx, h_beta_scalar, y, incy);
        }
        #undef GEMVN_DIM_X
        #undef GEMVN_DIM_Y
    }
    else {

        //number of columns on the y-dim of the grid, using gemvc because gemvt(transpose) is a instance of gemvc (conjugate)
        dim3 gemvc_grid( n, 1, 1 );
        dim3 gemvc_threads( 256, 1, 1 );

        if( rocblas_pointer_to_mode((void*)alpha) == rocblas_pointer_mode_device &&   rocblas_pointer_to_mode((void*)beta) == rocblas_pointer_mode_device ){
            hipLaunchKernel(HIP_KERNEL_NAME(gemvc_kernel_device_pointer<T, 256>), dim3(gemvc_grid), dim3(gemvc_threads), 0, rocblas_stream,
                                            transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
        }
        else{
            if ( 0.0 == *alpha && 1.0 == *beta) return rocblas_status_success;
            T h_alpha_scalar = *alpha; T h_beta_scalar = *beta;
            hipLaunchKernel(HIP_KERNEL_NAME(gemvc_kernel_host_pointer<T, 256>), dim3(gemvc_grid), dim3(gemvc_threads), 0, rocblas_stream,
                                            transA, m, n, h_alpha_scalar, A, lda, x, incx, h_beta_scalar, y, incy);
        }
    }
    return rocblas_status_success;
}




/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */



extern "C"
rocblas_status
rocblas_sgemv(rocblas_handle handle,
             rocblas_operation transA, rocblas_int m, rocblas_int n,
             const float *alpha,
             const float *A, rocblas_int lda,
             const float *x, rocblas_int incx,
             const float *beta,
             float *y, rocblas_int incy){

    return   rocblas_gemv_template<float>(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);

}

extern "C"
rocblas_status
rocblas_dgemv(rocblas_handle handle,
             rocblas_operation transA, rocblas_int m, rocblas_int n,
             const double *alpha,
             const double *A, rocblas_int lda,
             const double *x, rocblas_int incx,
             const double *beta,
             double *y, rocblas_int incy){

    return   rocblas_gemv_template<double>(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);

}
