/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "rocblas.h"

template <typename T, typename U, typename V>
__global__ void rocblas_scal_kernel(rocblas_int    n,
                                    V              alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    U              xa,
                                    ptrdiff_t      offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex)
{
    T*        x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto      alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= alpha;
}

template <rocblas_int NB, typename T, typename U, typename V>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_scal_template(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const V*       alpha,
                                                             rocblas_stride stride_alpha,
                                                             U              x,
                                                             rocblas_int    offsetx,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
    {
        return rocblas_status_success;
    }

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rocblas_scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           alpha,
                           stride_alpha,
                           x,
                           offsetx,
                           incx,
                           stridex);
    else // single alpha is on host
        hipLaunchKernelGGL(rocblas_scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           *alpha,
                           stride_alpha,
                           x,
                           offsetx,
                           incx,
                           stridex);

    return rocblas_status_success;
}

// alpha is entry in batched or strided_batched matrix
template <typename T, typename U, typename V>
__global__ void rocblas_scal_kernel(rocblas_int    n,
                                    const U*       alphaa,
                                    ptrdiff_t      offset_alpha,
                                    rocblas_int    inc_alpha,
                                    rocblas_stride stride_alpha,
                                    V*             xa,
                                    ptrdiff_t      offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex)
{
    T*        x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    const T*  alpha = load_ptr_batch(alphaa, hipBlockIdx_y, offset_alpha, stride_alpha);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= *alpha;
}

template <rocblas_int NB, typename T, typename U, typename V>
rocblas_status rocblas_scal_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const U*       alpha,
                                     rocblas_int    offset_alpha,
                                     rocblas_int    inc_alpha,
                                     rocblas_stride stride_alpha,
                                     V*             x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
    {
        return rocblas_status_success;
    }

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    hipLaunchKernelGGL(rocblas_scal_kernel<T>,
                       blocks,
                       threads,
                       0,
                       rocblas_stream,
                       n,
                       alpha,
                       offset_alpha,
                       inc_alpha,
                       stride_alpha,
                       x,
                       offsetx,
                       incx,
                       stridex);

    return rocblas_status_success;
}
