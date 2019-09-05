/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U>
__global__ void ger_strided_batched_kernel(rocblas_int m,
                                           rocblas_int n,
                                           U           alpha_device_host,
                                           const T* const __restrict__ xa,
                                           rocblas_int    shiftx,
                                           rocblas_int incx,
                                           rocblas_int stridex,
                                           const T* const __restrict__ ya,
                                           rocblas_int    shifty,
                                           rocblas_int incy,
                                           rocblas_int stridey,
                                           T* const    Aa,
                                           rocblas_int    shiftA,
                                           rocblas_int lda,
                                           rocblas_int strideA)
{

    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha = load_scalar(alpha_device_host);
        T*   A;
        const T* __restrict__ x;
        const T* __restrict__ y;
        A = Aa + hipBlockIdx_z * strideA + shiftA;
        x = xa + hipBlockIdx_z * stridex + shiftx;
        y = ya + hipBlockIdx_z * stridey + shifty;

        A[tx + lda * ty] += alpha * x[tx * incx] * y[ty * incy];
    }
}

template <typename T>
rocblas_status rocblas_ger_strided_batched_template(rocblas_handle handle,
                                                    rocblas_int    m,
                                                    rocblas_int    n,
                                                    const T*       alpha,
                                                    const T*       x,
                                                    rocblas_int    shiftx,
                                                    rocblas_int    incx,
                                                    rocblas_int    stridex,
                                                    const T*       y,
                                                    rocblas_int    shifty,
                                                    rocblas_int    incy,
                                                    rocblas_int    stridey,
                                                    T*             A,
                                                    rocblas_int    shiftA,
                                                    rocblas_int    lda,
                                                    rocblas_int    strideA,
                                                    rocblas_int    batch_count)
{
    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 ger_strided_batched_grid(blocksX, blocksY, batch_count);
    dim3 ger_strided_batched_threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(incx < 0)
        x -= ptrdiff_t(incx) * (m - 1);
    if(incy < 0)
        y -= ptrdiff_t(incy) * (n - 1);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(ger_strided_batched_kernel,
                           ger_strided_batched_grid,
                           ger_strided_batched_threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           alpha,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           y,
                           shifty,
                           incy,
                           stridey,
                           A,
                           shiftA,
                           lda,
                           strideA);
    else
        hipLaunchKernelGGL(ger_strided_batched_kernel,
                           ger_strided_batched_grid,
                           ger_strided_batched_threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           *alpha,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           y,
                           shifty,
                           incy,
                           stridey,
                           A,
                           shiftA,
                           lda,
                           strideA);
    return rocblas_status_success;
}
