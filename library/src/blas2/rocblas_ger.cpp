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
__global__ void ger_kernel(rocblas_int m,
                           rocblas_int n,
                           U alpha_device_host,
                           const T* __restrict__ x,
                           rocblas_int incx,
                           const T* __restrict__ y,
                           rocblas_int incy,
                           T* A,
                           rocblas_int lda)
{
    auto alpha = load_scalar(alpha_device_host);
    ssize_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ssize_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
        A[tx + lda * ty] += alpha * x[tx * incx] * y[ty * incy];
}

template <typename>
constexpr char rocblas_ger_name[] = "unknown";
template <>
constexpr char rocblas_ger_name<float>[] = "rocblas_sger";
template <>
constexpr char rocblas_ger_name<double>[] = "rocblas_dger";

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
rocblas_status rocblas_ger(rocblas_handle handle,
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
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!alpha)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_ger_name<T>, m, n, *alpha, x, incx, y, incy, A, lda);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f ger -r",
                      rocblas_precision_string<T>,
                      "-m",
                      m,
                      "-n",
                      n,
                      "--alpha",
                      *alpha,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--lda",
                      lda);
    }
    else
    {
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_ger_name<T>, m, n, alpha, x, incx, y, incy, A, lda);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(
            handle, rocblas_ger_name<T>, "M", m, "N", n, "incx", incx, "incy", incy, "lda", lda);

    if(!x || !y || !A)
        return rocblas_status_invalid_pointer;

    if(m < 0 || n < 0 || !incx || !incy || lda < m || lda < 1)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if(!m || !n)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int blocksX             = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int blocksY             = (n - 1) / GEMV_DIM_Y + 1;

    dim3 ger_grid(blocksX, blocksY);
    dim3 ger_threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(incx < 0)
        x += size_t(-incx) * (m - 1);
    if(incy < 0)
        y += size_t(-incy) * (n - 1);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(ger_kernel,
                           ger_grid,
                           ger_threads,
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
    else
        hipLaunchKernelGGL(ger_kernel,
                           ger_grid,
                           ger_threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           *alpha,
                           x,
                           incx,
                           y,
                           incy,
                           A,
                           lda);
    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sger(rocblas_handle handle,
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
    return rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

rocblas_status rocblas_dger(rocblas_handle handle,
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
    return rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

} // extern "C"
