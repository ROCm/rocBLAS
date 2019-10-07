/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U, typename V, typename W>
__global__ void ger_kernel(rocblas_int    m,
                           rocblas_int    n,
                           W              alpha_device_host,
                           rocblas_stride stride_alpha,
                           const U __restrict__ xa,
                           ptrdiff_t   shiftx,
                           rocblas_int incx,
                           rocblas_int stridex,
                           const U __restrict__ ya,
                           ptrdiff_t   shifty,
                           rocblas_int incy,
                           rocblas_int stridey,
                           V           Aa,
                           ptrdiff_t   shifta,
                           rocblas_int lda,
                           rocblas_int strideA)
{

    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha              = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
        T*   A                  = load_ptr_batch(Aa, hipBlockIdx_z, shifta, strideA);
        const T* __restrict__ x = load_ptr_batch(xa, hipBlockIdx_z, shiftx, stridex);
        const T* __restrict__ y = load_ptr_batch(ya, hipBlockIdx_z, shifty, stridey);

        A[tx + lda * ty] += alpha * x[tx * incx] * y[ty * incy];
    }
}

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_ger_template(rocblas_handle handle,
                                    rocblas_int    m,
                                    rocblas_int    n,
                                    const W*       alpha,
                                    rocblas_stride stride_alpha,
                                    const U*       x,
                                    rocblas_int    offsetx,
                                    rocblas_int    incx,
                                    rocblas_int    stridex,
                                    const U*       y,
                                    rocblas_int    offsety,
                                    rocblas_int    incy,
                                    rocblas_int    stridey,
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

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 grid(blocksX, blocksY, batch_count);
    dim3 threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(ger_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           alpha,
                           stride_alpha,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           y,
                           shifty,
                           incy,
                           stridey,
                           A,
                           offsetA,
                           lda,
                           strideA);
    else
        hipLaunchKernelGGL(ger_kernel<T>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           *alpha,
                           stride_alpha,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           y,
                           shifty,
                           incy,
                           stridey,
                           A,
                           offsetA,
                           lda,
                           strideA);
    return rocblas_status_success;
}
