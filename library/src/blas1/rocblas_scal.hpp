/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.hpp"
#include "rocblas.h"

template <typename Tex, typename Ta, typename Tx>
__global__ void rocblas_scal_kernel(rocblas_int    n,
                                    Ta             alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    Tx             xa,
                                    ptrdiff_t      offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex)
{
    auto*     x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto      alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
    {
        Tex res       = (Tex)x[tid * incx] * alpha;
        x[tid * incx] = res;
    }
}

template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_scal_template(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const Ta*      alpha,
                                                             rocblas_stride stride_alpha,
                                                             Tx             x,
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

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rocblas_scal_kernel<Tex>,
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
        hipLaunchKernelGGL(rocblas_scal_kernel<Tex>,
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
template <typename Tex, typename Ta, typename Tx>
__global__ void rocblas_scal_kernel(rocblas_int    n,
                                    const Ta*      alphaa,
                                    ptrdiff_t      offset_alpha,
                                    rocblas_int    inc_alpha,
                                    rocblas_stride stride_alpha,
                                    Tx*            xa,
                                    ptrdiff_t      offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex)
{
    auto*       x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    const auto* alpha = load_ptr_batch(alphaa, hipBlockIdx_y, offset_alpha, stride_alpha);
    ptrdiff_t   tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= *alpha;
}

template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
rocblas_status rocblas_scal_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const Ta*      alpha,
                                     rocblas_int    offset_alpha,
                                     rocblas_int    inc_alpha,
                                     rocblas_stride stride_alpha,
                                     Tx*            x,
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

    hipLaunchKernelGGL(rocblas_scal_kernel<Tex>,
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
