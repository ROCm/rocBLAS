/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U>
__global__ void ger_kernel(rocblas_int m,
                           rocblas_int n,
                           U           alpha_device_host,
                           const T* __restrict__ x,
                           rocblas_int incx,
                           const T* __restrict__ y,
                           rocblas_int incy,
                           T*          A,
                           rocblas_int lda)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha = load_scalar(alpha_device_host);
        A[tx + lda * ty] += alpha * x[tx * incx] * y[ty * incy];
    }
}

template <typename T, typename U>
__global__ void ger_batched_kernel(rocblas_int m,
                                   rocblas_int n,
                                   U           alpha_device_host,
                                   const T* const __restrict__ xa[],
                                   rocblas_int incx,
                                   const T* const __restrict__ ya[],
                                   rocblas_int incy,
                                   T* const    Aa[],
                                   rocblas_int lda)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha = load_scalar(alpha_device_host);
        T*   A;
        const T* __restrict__ x;
        const T* __restrict__ y;
        A = Aa[hipBlockIdx_z];
        x = xa[hipBlockIdx_z];
        y = ya[hipBlockIdx_z];

        if(incx < 0)
            x -= ssize_t(incx) * (m - 1);
        if(incy < 0)
            y -= ssize_t(incy) * (n - 1);

        A[tx + lda * ty] += alpha * x[tx * incx] * y[ty * incy];
    }
}

template <typename T>
rocblas_status rocblas_ger_batched_template(rocblas_handle handle,
                                            rocblas_int    m,
                                            rocblas_int    n,
                                            const T*       alpha,
                                            const T* const x[],
                                            rocblas_int    incx,
                                            const T* const y[],
                                            rocblas_int    incy,
                                            T* const       A[],
                                            rocblas_int    lda,
                                            rocblas_int    batch_count)
{
    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 ger_batched_grid(blocksX, blocksY, batch_count);
    dim3 ger_batched_threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(ger_batched_kernel,
                           ger_batched_grid,
                           ger_batched_threads,
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
        hipLaunchKernelGGL(ger_batched_kernel,
                           ger_batched_grid,
                           ger_batched_threads,
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
