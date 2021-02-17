/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
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

//!
//! @brief Optimized kernel for the SCAL floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
__global__ __launch_bounds__(NB) void sscal_2_kernel(rocblas_int    n,
                                                     Ta             alpha_device_host,
                                                     rocblas_stride stride_alpha,
                                                     Tx __restrict__ xa,
                                                     ptrdiff_t      offsetx,
                                                     rocblas_stride stridex)
{
    auto*     x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto      alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    ptrdiff_t tid   = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 2;

    if(tid < n - 1)
    {
        // Each thread access contiguous elements for example Thread '0' access indices '0' and '1' of the vector `x`
        for(rocblas_int j = 0; j < 2; ++j)
        {
            Tex res    = (Tex)x[tid + j] * alpha;
            x[tid + j] = res;
        }
    }

    // If `n` is odd then the computation of last element in the vector `x` is covered below.
    if(n % 2 != 0 && tid == n - 1)
    {
        Tex res = (Tex)x[tid] * alpha;
        x[tid]  = res;
    }
}

template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_scal_template(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const Ta*      alpha,
                                                             rocblas_stride stride_alpha,
                                                             Tx __restrict__ x,
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

    static constexpr bool using_rocblas_float
        = std::is_same<Tx, rocblas_float*>{} || std::is_same<Tx, rocblas_float* const*>{};

    if(!using_rocblas_float || incx != 1)
    {
        int  blocks = (n - 1) / NB + 1;
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL(rocblas_scal_kernel<Tex>,
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n,
                               alpha,
                               stride_alpha,
                               x,
                               offsetx,
                               incx,
                               stridex);
        else // single alpha is on host
            hipLaunchKernelGGL(rocblas_scal_kernel<Tex>,
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n,
                               *alpha,
                               stride_alpha,
                               x,
                               offsetx,
                               incx,
                               stridex);
    }
    else
    {
        // Kernel function for improving the performance of SSCAL when incx==1 and incy==1
        int  blocks = 1 + ((n - 1) / (NB * 2));
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL((sscal_2_kernel<NB, Tex>),
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n,
                               alpha,
                               stride_alpha,
                               x,
                               offsetx,
                               stridex);
        else // single alpha is on host
            hipLaunchKernelGGL((sscal_2_kernel<NB, Tex>),
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n,
                               *alpha,
                               stride_alpha,
                               x,
                               offsetx,
                               stridex);
    }
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

//!
//! @brief Optimized kernel for the SCAL floating points when alpha is entry in batched or strided_batched matrix
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
__global__ __launch_bounds__(NB) void sscal_2_kernel(rocblas_int    n,
                                                     const Ta*      alphaa,
                                                     ptrdiff_t      offset_alpha,
                                                     rocblas_stride stride_alpha,
                                                     Tx*            xa,
                                                     ptrdiff_t      offsetx,
                                                     rocblas_stride stridex)
{
    auto*       x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    const auto* alpha = load_ptr_batch(alphaa, hipBlockIdx_y, offset_alpha, stride_alpha);
    ptrdiff_t   tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // Each thread access contiguous elements for example Thread '0' access indices '0' and '1' of the vector `x`
        for(rocblas_int j = 0; j < 2; ++j)
        {
            x[tid + j] *= *alpha;
        }
    }

    // If `n` is odd then the computation of last element in the vector `x` is covered below.
    if(n % 2 != 0 && tid == n - 1)
    {
        x[tid] *= *alpha;
    }
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

    static constexpr bool using_rocblas_float
        = std::is_same<Tx*, rocblas_float*>{} || std::is_same<Tx*, rocblas_float* const*>{};

    if(!using_rocblas_float || incx != 1)
    {
        int  blocks = (n - 1) / NB + 1;
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);
        hipLaunchKernelGGL(rocblas_scal_kernel<Tex>,
                           grid,
                           threads,
                           0,
                           handle->get_stream(),
                           n,
                           alpha,
                           offset_alpha,
                           inc_alpha,
                           stride_alpha,
                           x,
                           offsetx,
                           incx,
                           stridex);
    }
    else
    {
        // Kernel function for improving the performance of SSCAL when incx==1 and incy==1
        int  blocks = 1 + ((n - 1) / (NB * 2));
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);
        hipLaunchKernelGGL((sscal_2_kernel<NB, Tex>),
                           grid,
                           threads,
                           0,
                           handle->get_stream(),
                           n,
                           alpha,
                           offset_alpha,
                           stride_alpha,
                           x,
                           offsetx,
                           stridex);
    }
    return rocblas_status_success;
}
