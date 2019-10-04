/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U, typename V>
__global__ void copy_kernel(rocblas_int    n,
                            const U        xa,
                            ptrdiff_t      shiftx,
                            rocblas_int    incx,
                            rocblas_stride stridex,
                            V              ya,
                            ptrdiff_t      shifty,
                            rocblas_int    incy,
                            rocblas_stride stridey)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(tid < n)
    {
        const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
        T*       y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

        y[tid * incy] = x[tid * incx];
    }
}

template <rocblas_int NB, typename T, typename U, typename V>
rocblas_status rocblas_copy_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     V              y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    // Quick return if possible.
    if(n <= 0 || !batch_count)
        return rocblas_status_success;

    if(!x || !y)
        return rocblas_status_invalid_pointer;

    if(batch_count < 0)
        return rocblas_status_invalid_size;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = offsetx - ((incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0);
    ptrdiff_t shifty = offsety - ((incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0);

    int  blocks = (n - 1) / NB + 1;
    dim3 grid(blocks, batch_count);
    dim3 threads(NB);

    hipLaunchKernelGGL((copy_kernel<T>),
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       n,
                       x,
                       shiftx,
                       incx,
                       stridex,
                       y,
                       shifty,
                       incy,
                       stridey);

    return rocblas_status_success;
}
