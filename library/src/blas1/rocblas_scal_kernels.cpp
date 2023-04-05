/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "handle.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_scal.hpp"

template <rocblas_int NB, typename T, typename Tex, typename Ta, typename Tx>
ROCBLAS_KERNEL(NB)
rocblas_scal_kernel(rocblas_int    n,
                    Ta             alpha_device_host,
                    rocblas_stride stride_alpha,
                    Tx             xa,
                    rocblas_stride offset_x,
                    rocblas_int    incx,
                    rocblas_stride stride_x)
{
    auto* x     = load_ptr_batch(xa, blockIdx.y, offset_x, stride_x);
    auto  alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

    if(alpha == 1)
        return;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // bound
    if(tid < n)
    {
        Tex res                = (Tex)x[tid * int64_t(incx)] * alpha;
        x[tid * int64_t(incx)] = (T)res;
    }
}

//!
//! @brief Optimized kernel for the SCAL floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename T, typename Tex, typename Ta, typename Tx>
ROCBLAS_KERNEL(NB)
rocblas_sscal_2_kernel(rocblas_int    n,
                       Ta             alpha_device_host,
                       rocblas_stride stride_alpha,
                       Tx __restrict__ xa,
                       rocblas_stride offset_x,
                       rocblas_stride stride_x)
{
    auto* x     = load_ptr_batch(xa, blockIdx.y, offset_x, stride_x);
    auto  alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

    if(alpha == 1)
        return;

    uint32_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if(tid < n - 1)
    {
        // Each thread access contiguous elements for example Thread '0' access indices '0' and '1' of the vector `x`
        for(int32_t j = 0; j < 2; ++j)
        {
            Tex res    = (Tex)x[tid + j] * alpha;
            x[tid + j] = (T)res;
        }
    }

    // If `n` is odd then the computation of last element in the vector `x` is covered below.
    if(n % 2 != 0 && tid == n - 1)
    {
        Tex res = (Tex)x[tid] * alpha;
        x[tid]  = (T)res;
    }
}

//!
//! @brief Optimized kernel for the SCAL when the compute and alpha type is half precision.
//! @remark Increments are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Ta, typename Tx>
ROCBLAS_KERNEL(NB)
rocblas_hscal_mlt_4_kernel(rocblas_int    n,
                           rocblas_int    n_mod_4,
                           rocblas_int    n_mlt_4,
                           Ta             alpha_device_host,
                           rocblas_stride stride_alpha,
                           Tx __restrict__ xa,
                           rocblas_stride offset_x,
                           rocblas_stride stride_x)
{
    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

    if(alpha == 1)
        return;

    rocblas_half2 x0, x1;
    rocblas_half2 z0, z1;

    uint32_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if(tid < n - 3)
    {
        rocblas_half4* x = (rocblas_half4*)load_ptr_batch(xa, blockIdx.y, offset_x + tid, stride_x);

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
            auto* x = load_ptr_batch(xa, blockIdx.y, offset_x, stride_x);
            for(int32_t j = 0; j < n_mod_4; ++j)
            {
                x[tid + j] = x[tid + j] * alpha;
            }
        }
    }
}

template <rocblas_int NB, typename T, typename Tex, typename Ta, typename Tx>
rocblas_status rocblas_internal_scal_template(rocblas_handle handle,
                                              rocblas_int    n,
                                              const Ta*      alpha,
                                              rocblas_stride stride_alpha,
                                              Tx             x,
                                              rocblas_stride offset_x,
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
        int32_t blocks = 1 + ((n - 1) / (NB * 2));
        dim3    grid(blocks, batch_count);
        dim3    threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL((rocblas_sscal_2_kernel<NB, T, Tex>),
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
            hipLaunchKernelGGL((rocblas_sscal_2_kernel<NB, T, Tex>),
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
        int32_t n_mod_4 = n & 3; // n mod 4
        int32_t n_mlt_4 = n & ~(rocblas_int)3; // multiple of 4
        int32_t blocks  = 1 + ((n - 1) / (NB * 4));
        dim3    grid(blocks, batch_count);
        dim3    threads(NB);

        if constexpr(using_rocblas_half)
        {
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                hipLaunchKernelGGL((rocblas_hscal_mlt_4_kernel<NB>),
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
                hipLaunchKernelGGL((rocblas_hscal_mlt_4_kernel<NB>),
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
    }
    else
    {
        int  blocks = (n - 1) / NB + 1;
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL((rocblas_scal_kernel<NB, T, Tex>),
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
            hipLaunchKernelGGL((rocblas_scal_kernel<NB, T, Tex>),
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

template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   T*             x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   rocblas_int    batch_count)
{
    return rocblas_internal_scal_template<ROCBLAS_SCAL_NB, T, T>(
        handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_batched_template(rocblas_handle handle,
                                           rocblas_int    n,
                                           const Ta*      alpha,
                                           rocblas_stride stride_alpha,
                                           T* const*      x,
                                           rocblas_stride offset_x,
                                           rocblas_int    incx,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count)
{
    return rocblas_internal_scal_template<ROCBLAS_SCAL_NB, T, T>(
        handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files scal*.cpp

// clang-format off
#ifdef INSTANTIATE_SCAL_TEMPLATE
#error INSTANTIATE_SCAL_TEMPLATE already defined
#endif

#define INSTANTIATE_SCAL_TEMPLATE(T_, Ta_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status            \
        rocblas_internal_scal_template<T_, Ta_>(rocblas_handle handle,  \
                                           rocblas_int    n,            \
                                           const Ta_*     alpha,        \
                                           rocblas_stride stride_alpha, \
                                           T_*            x,            \
                                           rocblas_stride offset_x,     \
                                           rocblas_int    incx,         \
                                           rocblas_stride stride_x,     \
                                           rocblas_int    batch_count);

// Not exporting execution type
INSTANTIATE_SCAL_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_SCAL_TEMPLATE(rocblas_half, float)
INSTANTIATE_SCAL_TEMPLATE(float, float)
INSTANTIATE_SCAL_TEMPLATE(double, double)
INSTANTIATE_SCAL_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_SCAL_TEMPLATE(rocblas_double_complex, rocblas_double_complex)
INSTANTIATE_SCAL_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_SCAL_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_SCAL_TEMPLATE

#ifdef INSTANTIATE_SCAL_BATCHED_TEMPLATE
#error INSTANTIATE_SCAL_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_SCAL_BATCHED_TEMPLATE(T_, Ta_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_scal_batched_template<T_, Ta_>(rocblas_handle handle,  \
                                                   rocblas_int    n,            \
                                                   const Ta_*     alpha,        \
                                                   rocblas_stride stride_alpha, \
                                                   T_* const*     x,            \
                                                   rocblas_stride offset_x,     \
                                                   rocblas_int    incx,         \
                                                   rocblas_stride stride_x,     \
                                                   rocblas_int    batch_count);

INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_half, float)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(float, float)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(double, double)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_SCAL_BATCHED_TEMPLATE

#ifdef INSTANTIATE_SCAL_EX_TEMPLATE
#error INSTANTIATE_SCAL_EX_TEMPLATE already defined
#endif

#define INSTANTIATE_SCAL_EX_TEMPLATE(NB_, T_, Tex_, Ta_, Tx_)                                \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
        rocblas_internal_scal_template<NB_, T_, Tex_, Ta_, Tx_>(rocblas_handle handle,       \
                                                                rocblas_int    n,            \
                                                                const Ta_*     alpha,        \
                                                                rocblas_stride stride_alpha, \
                                                                Tx_            x,            \
                                                                rocblas_stride offset_x,     \
                                                                rocblas_int    incx,         \
                                                                rocblas_stride stride_x,     \
                                                                rocblas_int    batch_count);

// Instantiations for scal_ex
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_half, float, rocblas_half, rocblas_half*)
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_half, float, float, rocblas_half*)
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_bfloat16*)
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, float, rocblas_bfloat16*)

INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_half, float, rocblas_half, rocblas_half* const*)
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_half, float, float, rocblas_half* const*)
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_bfloat16* const*)
INSTANTIATE_SCAL_EX_TEMPLATE(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, float, rocblas_bfloat16* const*)

#undef INSTANTIATE_SCAL_EX_TEMPLATE

// clang-format on
