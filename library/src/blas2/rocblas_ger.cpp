/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "ger_device.h"
#include "handle.h"

template <typename T>
__global__ void ger_kernel_host_pointer(hipLaunchParm lp,
                                        rocblas_int m,
                                        rocblas_int n,
                                        const T alpha,
                                        const T* __restrict__ x,
                                        rocblas_int incx,
                                        const T* __restrict__ y,
                                        rocblas_int incy,
                                        T* A,
                                        rocblas_int lda)
{
    ger_device<T>(m, n, alpha, x, incx, y, incy, A, lda);
}

template <typename T>
__global__ void ger_kernel_device_pointer(hipLaunchParm lp,
                                          rocblas_int m,
                                          rocblas_int n,
                                          const T* alpha,
                                          const T* __restrict__ x,
                                          rocblas_int incx,
                                          const T* __restrict__ y,
                                          rocblas_int incy,
                                          T* A,
                                          rocblas_int lda)
{
    ger_device<T>(m, n, *alpha, x, incx, y, incy, A, lda);
}

/*! \brief BLAS Level 2 API

    \details
    xGER performs the matrix-vector operations

        A := A + alpha*x*y**T

    where alpha is a scalars, x and y are vectors, and A is an
    m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    m         rocblas_int
              m > 0
    @param[in]
    n         rocblas_int
              n > 0
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
              incx != 0
    @param[in]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.
              incy != 0
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
              lda >= m && lda > 0

    ********************************************************************/

template <typename T>
rocblas_status rocblas_ger_template(rocblas_handle handle,
                                    rocblas_int m,
                                    rocblas_int n,
                                    const T* alpha,
                                    const T* x,
                                    rocblas_int incx,
                                    const T* y,
                                    rocblas_int incy,
                                    T* A,
                                    rocblas_int lda)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_function(handle,
                     replaceX<T>("rocblas_Xger"),
                     m,
                     n,
                     *alpha,
                     (const void*&)x,
                     incx,
                     (const void*&)y,
                     incy,
                     (const void*&)A,
                     lda);
    }
    else
    {
        log_function(handle,
                     replaceX<T>("rocblas_Xger"),
                     m,
                     n,
                     (const void*&)alpha,
                     (const void*&)x,
                     incx,
                     (const void*&)y,
                     incy,
                     (const void*&)A,
                     lda);
    }

    if(nullptr == alpha)
        return rocblas_status_invalid_pointer;
    else if(nullptr == x)
        return rocblas_status_invalid_pointer;
    else if(nullptr == y)
        return rocblas_status_invalid_pointer;
    else if(nullptr == A)
        return rocblas_status_invalid_pointer;

    if(m < 0)
        return rocblas_status_invalid_size;
    else if(n < 0)
        return rocblas_status_invalid_size;
    else if(0 == incx)
        return rocblas_status_invalid_size;
    else if(0 == incy)
        return rocblas_status_invalid_size;
    else if(lda < m || lda < 1)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if(m == 0 || n == 0 || alpha == 0)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->rocblas_stream;

#define GEMV_DIM_X 128
#define GEMV_DIM_Y 8
    rocblas_int blocksX = ((m - 1) / GEMV_DIM_X) + 1;
    rocblas_int blocksY = ((n - 1) / GEMV_DIM_Y) + 1;

    dim3 ger_grid(blocksX, blocksY, 1);
    dim3 ger_threads(GEMV_DIM_X, GEMV_DIM_Y, 1);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernel(HIP_KERNEL_NAME(ger_kernel_device_pointer<T>),
                        dim3(ger_grid),
                        dim3(ger_threads),
                        0,
                        rocblas_stream,
                        m,
                        n,
                        alpha,
                        x,
                        incx,
                        y,
                        incy,
                        A,
                        lda);
    }
    else
    {
        T h_alpha_scalar = *alpha;
        hipLaunchKernel(HIP_KERNEL_NAME(ger_kernel_host_pointer<T>),
                        dim3(ger_grid),
                        dim3(ger_threads),
                        0,
                        rocblas_stream,
                        m,
                        n,
                        h_alpha_scalar,
                        x,
                        incx,
                        y,
                        incy,
                        A,
                        lda);
    }
#undef GEMV_DIM_X
#undef GEMV_DIM_Y

    return rocblas_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_sger(rocblas_handle handle,
                                       rocblas_int m,
                                       rocblas_int n,
                                       const float* alpha,
                                       const float* x,
                                       rocblas_int incx,
                                       const float* y,
                                       rocblas_int incy,
                                       float* A,
                                       rocblas_int lda)
{
    return rocblas_ger_template<float>(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

extern "C" rocblas_status rocblas_dger(rocblas_handle handle,
                                       rocblas_int m,
                                       rocblas_int n,
                                       const double* alpha,
                                       const double* x,
                                       rocblas_int incx,
                                       const double* y,
                                       rocblas_int incy,
                                       double* A,
                                       rocblas_int lda)
{
    return rocblas_ger_template<double>(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
