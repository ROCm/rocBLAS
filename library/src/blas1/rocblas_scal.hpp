/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"
#include "rocblas/rocblas.h"

template <typename Tex, typename Ta, typename Tx>
ROCBLAS_KERNEL void rocblas_scal_kernel(rocblas_int    n,
                                        Ta             alpha_device_host,
                                        rocblas_stride stride_alpha,
                                        Tx             xa,
                                        ptrdiff_t      offset_x,
                                        rocblas_int    incx,
                                        rocblas_stride stride_x)
{
    auto*     x     = load_ptr_batch(xa, hipBlockIdx_y, offset_x, stride_x);
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
ROCBLAS_KERNEL __launch_bounds__(NB) void sscal_2_kernel(rocblas_int    n,
                                                         Ta             alpha_device_host,
                                                         rocblas_stride stride_alpha,
                                                         Tx __restrict__ xa,
                                                         ptrdiff_t      offset_x,
                                                         rocblas_stride stride_x)
{
    auto*     x     = load_ptr_batch(xa, hipBlockIdx_y, offset_x, stride_x);
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

//!
//! @brief Optimized kernel for the SCAL half points.
//! @remark Increments are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Ta, typename Tx>
ROCBLAS_KERNEL __launch_bounds__(NB) void hscal_mlt_4_kernel(rocblas_int    n,
                                                             rocblas_int    n_mod_4,
                                                             rocblas_int    n_mlt_4,
                                                             Ta             alpha_device_host,
                                                             rocblas_stride stride_alpha,
                                                             Tx __restrict__ xa,
                                                             ptrdiff_t      offset_x,
                                                             rocblas_stride stride_x)
{

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);

    rocblas_half2 x0, x1;
    rocblas_half2 z0, z1;

    ptrdiff_t tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 4;

    if(tid < n - 3)
    {
        rocblas_half4* x
            = (rocblas_half4*)load_ptr_batch(xa, hipBlockIdx_y, offset_x + tid, stride_x);

        x0[0] = (*x)[0];
        x0[1] = (*x)[1];
        x1[0] = (*x)[2];
        x1[1] = (*x)[3];

        z0[0] = alpha * x0[0];
        z0[1] = alpha * x0[1];
        z1[0] = alpha * x1[0];
        z1[1] = alpha * x1[1];

        (*x)[0] = z0[0];
        (*x)[1] = z0[1];
        (*x)[2] = z1[0];
        (*x)[3] = z1[1];
    }

    // If `n_mod_4` is true then the computation of last few element in the vector `x` is covered below.
    if(n_mod_4)
    {
        //The last ThreadID which is a multiple of 4 should complete the computation of last few elements of vector `x`
        if(tid == n_mlt_4)
        {
            auto* x = load_ptr_batch(xa, hipBlockIdx_y, offset_x, stride_x);
            for(rocblas_int j = 0; j < n_mod_4; ++j)
            {
                x[tid + j] = x[tid + j] * alpha;
            }
        }
    }
}

template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   Tx             x,
                                   rocblas_int    offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
    {
        return rocblas_status_success;
    }

    static constexpr bool using_rocblas_float
        = std::is_same<Tx, rocblas_float*>{} || std::is_same<Tx, rocblas_float* const*>{};

    // Using rocblas_half ?
    static constexpr bool using_rocblas_half
        = std::is_same<Ta, rocblas_half>{} && std::is_same<Tex, rocblas_half>{};

    if(using_rocblas_float && incx == 1)
    {
        // Kernel function for improving the performance of SSCAL when incx==1
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
                               offset_x,
                               stride_x);
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
                               offset_x,
                               stride_x);
    }
    else if(using_rocblas_half && incx == 1)
    {
        // Kernel function for improving the performance of HSCAL when incx==1
        rocblas_int n_mod_4 = n & 3; // n mod 4
        rocblas_int n_mlt_4 = n & ~(rocblas_int)3; // multiple of 4
        int         blocks  = 1 + ((n - 1) / (NB * 4));
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL((hscal_mlt_4_kernel<NB>),
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n,
                               n_mod_4,
                               n_mlt_4,
                               (const rocblas_half*)alpha,
                               stride_alpha,
                               x,
                               offset_x,
                               stride_x);
        else // single alpha is on host
            hipLaunchKernelGGL((hscal_mlt_4_kernel<NB>),
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n,
                               n_mod_4,
                               n_mlt_4,
                               load_scalar((const rocblas_half*)alpha),
                               stride_alpha,
                               x,
                               offset_x,
                               stride_x);
    }
    else
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
                               offset_x,
                               incx,
                               stride_x);
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
                               offset_x,
                               incx,
                               stride_x);
    }
    return rocblas_status_success;
}
