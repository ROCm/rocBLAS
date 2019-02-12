/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

template <typename T, typename U>
__global__ void syr_kernel(rocblas_fill uplo,
                           rocblas_int n,
                           U alpha_device_host,
                           const T* __restrict__ x,
                           rocblas_int incx,
                           T* A,
                           rocblas_int lda)
{
    auto alpha     = load_scalar(alpha_device_host);
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(uplo == rocblas_fill_lower ? tx < n && ty <= tx : ty < n && tx <= ty)
        A[tx + lda * ty] += alpha * x[tx * incx] * x[ty * incx];
}

template <typename>
constexpr char rocblas_syr_name[] = "unknown";
template <>
constexpr char rocblas_syr_name<float>[] = "rocblas_ssyr";
template <>
constexpr char rocblas_syr_name<double>[] = "rocblas_dsyr";

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
rocblas_status rocblas_syr(rocblas_handle handle,
                           rocblas_fill uplo,
                           rocblas_int n,
                           const T* alpha,
                           const T* x,
                           rocblas_int incx,
                           T* A,
                           rocblas_int lda)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!alpha)
        return rocblas_status_invalid_pointer;
    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto uplo_letter = rocblas_fill_letter(uplo);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_syr_name<T>, uplo, n, *alpha, x, incx, A, lda);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f syr -r",
                          rocblas_precision_string<T>,
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
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_syr_name<T>, uplo, n, alpha, x, incx, A, lda);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(
                handle, rocblas_syr_name<T>, "uplo", uplo_letter, "N", n, "incx", incx, "lda", lda);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    if(!x || !A)
        return rocblas_status_invalid_pointer;
    if(n < 0 || !incx || lda < n || lda < 1)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if(!n)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int blocksX             = (n - 1) / GEMV_DIM_X + 1;
    rocblas_int blocksY             = (n - 1) / GEMV_DIM_Y + 1;

    dim3 syr_grid(blocksX, blocksY);
    dim3 syr_threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(incx < 0)
        x += size_t(-incx) * (n - 1);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(
            syr_kernel, syr_grid, syr_threads, 0, rocblas_stream, uplo, n, alpha, x, incx, A, lda);
    else
        hipLaunchKernelGGL(
            syr_kernel, syr_grid, syr_threads, 0, rocblas_stream, uplo, n, *alpha, x, incx, A, lda);

    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_ssyr(rocblas_handle handle,
                            rocblas_fill uplo,
                            rocblas_int n,
                            const float* alpha,
                            const float* x,
                            rocblas_int incx,
                            float* A,
                            rocblas_int lda)
{
    return rocblas_syr(handle, uplo, n, alpha, x, incx, A, lda);
}

rocblas_status rocblas_dsyr(rocblas_handle handle,
                            rocblas_fill uplo,
                            rocblas_int n,
                            const double* alpha,
                            const double* x,
                            rocblas_int incx,
                            double* A,
                            rocblas_int lda)
{
    return rocblas_syr(handle, uplo, n, alpha, x, incx, A, lda);
}

} // extern "C"
