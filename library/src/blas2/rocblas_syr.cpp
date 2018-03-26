/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "syr_device.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

template <typename T>
__global__ void syr_kernel_host_pointer(rocblas_fill uplo,
                                        rocblas_int n,
                                        const T alpha,
                                        const T* __restrict__ x,
                                        rocblas_int incx,
                                        T* A,
                                        rocblas_int lda)
{
    syr_device<T>(uplo, n, alpha, x, incx, A, lda);
}

template <typename T>
__global__ void syr_kernel_device_pointer(rocblas_fill uplo,
                                          rocblas_int n,
                                          const T* alpha,
                                          const T* __restrict__ x,
                                          rocblas_int incx,
                                          T* A,
                                          rocblas_int lda)
{
    syr_device<T>(uplo, n, *alpha, x, incx, A, lda);
}

/*! \brief BLAS Level 2 API

    \details
    xSYR performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalars, x and y are vectors, and A is a
    symmetric n by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.
    @param[in]
    n         rocblas_int
              n >= 0
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
              incx != 0
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
              lda >= n && lda >= 1

    ********************************************************************/

template <typename T>
rocblas_status rocblas_syr_template(rocblas_handle handle,
                                    rocblas_fill uplo,
                                    rocblas_int n,
                                    const T* alpha,
                                    const T* x,
                                    rocblas_int incx,
                                    T* A,
                                    rocblas_int lda)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xsyr"),
                  uplo,
                  n,
                  *alpha,
                  (const void*&)x,
                  incx,
                  (const void*&)A,
                  lda);

        std::string uplo_letter = rocblas_fill_letter(uplo);

        log_bench(handle,
                  "./rocblas-bench -f syr -r",
                  replaceX<T>("X"),
                  "--uplo",
                  uplo_letter,
                  "-n",
                  n,
                  "--alpha",
                  *alpha,
                  "--incx",
                  incx,
                  "--lda",
                  lda);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xsyr"),
                  uplo,
                  n,
                  (const void*&)alpha,
                  (const void*&)x,
                  incx,
                  (const void*&)A,
                  lda);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if(nullptr == alpha)
        return rocblas_status_invalid_pointer;
    else if(nullptr == x)
        return rocblas_status_invalid_pointer;
    else if(nullptr == A)
        return rocblas_status_invalid_pointer;

    if(n < 0)
        return rocblas_status_invalid_size;
    else if(0 == incx)
        return rocblas_status_invalid_size;
    else if(lda < n || lda < 1)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if(n == 0 || alpha == 0)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->rocblas_stream;

#define GEMV_DIM_X 128
#define GEMV_DIM_Y 8
    rocblas_int blocksX = ((n - 1) / GEMV_DIM_X) + 1;
    rocblas_int blocksY = ((n - 1) / GEMV_DIM_Y) + 1;

    dim3 syr_grid(blocksX, blocksY, 1);
    dim3 syr_threads(GEMV_DIM_X, GEMV_DIM_Y, 1);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL((syr_kernel_device_pointer<T>),
                           dim3(syr_grid),
                           dim3(syr_threads),
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           alpha,
                           x,
                           incx,
                           A,
                           lda);
    }
    else
    {
        T h_alpha_scalar = *alpha;
        hipLaunchKernelGGL((syr_kernel_host_pointer<T>),
                           dim3(syr_grid),
                           dim3(syr_threads),
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           h_alpha_scalar,
                           x,
                           incx,
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

extern "C" rocblas_status rocblas_ssyr(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_int n,
                                       const float* alpha,
                                       const float* x,
                                       rocblas_int incx,
                                       float* A,
                                       rocblas_int lda)
{
    return rocblas_syr_template<float>(handle, uplo, n, alpha, x, incx, A, lda);
}

extern "C" rocblas_status rocblas_dsyr(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_int n,
                                       const double* alpha,
                                       const double* x,
                                       rocblas_int incx,
                                       double* A,
                                       rocblas_int lda)
{
    return rocblas_syr_template<double>(handle, uplo, n, alpha, x, incx, A, lda);
}
