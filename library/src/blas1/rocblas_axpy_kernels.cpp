/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_axpy.hpp"

//!
//! @brief General kernel (batched, strided batched) of axpy.
//!
template <rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
axpy_kernel(rocblas_int    n,
            Ta             alpha_device_host,
            rocblas_stride stride_alpha,
            Tx __restrict__ x,
            rocblas_stride offset_x,
            rocblas_int    incx,
            rocblas_stride stride_x,
            Ty __restrict__ y,
            rocblas_stride offset_y,
            rocblas_int    incy,
            rocblas_stride stride_y)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    if(!alpha)
    {
        return;
    }

    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offset_x + tid * incx, stride_x);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offset_y + tid * incy, stride_y);

        *ty = (*ty) + Tex(alpha) * (*tx);
    }
}

//!
//! @brief Optimized kernel for the AXPY floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
saxpy_2_kernel(rocblas_int    n,
               Ta             alpha_device_host,
               rocblas_stride stride_alpha,
               Tx __restrict__ x,
               rocblas_stride offset_x,
               rocblas_stride stride_x,
               Ty __restrict__ y,
               rocblas_stride offset_y,
               rocblas_stride stride_y)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    if(!alpha)
    {
        return;
    }
    auto* tx = load_ptr_batch(x, hipBlockIdx_y, offset_x, stride_x);
    auto* ty = load_ptr_batch(y, hipBlockIdx_y, offset_y, stride_y);

    ptrdiff_t tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 2;

    if(tid < n - 1)
    {
        // Each thread access contiguous elements for example Thread '0' access indices '0' and '1' of the vectors `x` and `y`
        for(rocblas_int j = 0; j < 2; ++j)
        {
            ty[tid + j] = ty[tid + j] + Tex(alpha) * tx[tid + j];
        }
    }

    // If `n` is odd then the computation of last element in the vectors is covered below.
    if(n % 2 != 0 && tid == n - 1)
    {
        ty[tid] = ty[tid] + Tex(alpha) * tx[tid];
    }
}

//!
//! @brief Large batch size kernel (batched, strided batched) of axpy.
//!
template <int DIM_X, int DIM_Y, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
axpy_kernel_batched(rocblas_int    n,
                    Ta             alpha_device_host,
                    rocblas_stride stride_alpha,
                    Tx             x,
                    rocblas_stride offset_x,
                    rocblas_int    incx,
                    rocblas_stride stride_x,
                    Ty             y,
                    rocblas_stride offset_y,
                    rocblas_int    incy,
                    rocblas_stride stride_y,
                    rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    if(!alpha)
    {
        return;
    }
    Tex ex_alph = Tex(alpha);

    ptrdiff_t tid = hipBlockIdx_x * DIM_X + hipThreadIdx_x;
    int       bid = 4 * (hipBlockIdx_y * DIM_Y + hipThreadIdx_y);
    if(tid < n)
    {
        offset_x += tid * incx;
        offset_y += tid * incy;

        for(int i = 0; i < 4; i++)
        {
            if(bid + i < batch_count)
            {
                auto tx = load_ptr_batch(x, bid + i, offset_x, stride_x);
                auto ty = load_ptr_batch(y, bid + i, offset_y, stride_y);

                *ty += ex_alph * (*tx);
            }
        }
    }
}

//!
//! @brief Optimized kernel for the remaining part of 8 half floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
haxpy_mod_8_kernel(rocblas_int    n_mod_8,
                   Ta             alpha_device_host,
                   rocblas_stride stride_alpha,
                   Tx             x,
                   ptrdiff_t      offset_x,
                   rocblas_stride stride_x,
                   Ty             y,
                   ptrdiff_t      offset_y,
                   rocblas_stride stride_y)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    if(!alpha)
        return;

    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n_mod_8)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offset_x + tid, stride_x);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offset_y + tid, stride_y);
        *ty += alpha * (*tx);
    }
}

//!
//! @brief Optimized kernel for the groups of 8 half floating points.
//!
template <rocblas_int NB, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
haxpy_mlt_8_kernel(rocblas_int    n_mlt_8,
                   Ta             alpha_device_host,
                   rocblas_stride stride_alpha,
                   Tx             x,
                   rocblas_stride offset_x,
                   rocblas_stride stride_x,
                   Ty             y,
                   rocblas_stride offset_y,
                   rocblas_stride stride_y)
{
    // Load alpha into both sides of a rocblas_half2 for fma instructions.
    auto alpha_value = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    union
    {
        rocblas_half2 value;
        uint32_t      data;
    } alpha_h2 = {{alpha_value, alpha_value}};

    if(!(alpha_h2.data & 0x7fff))
    {
        return;
    }

    ptrdiff_t t8id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    rocblas_half2 y0, y1, y2, y3;
    rocblas_half2 x0, x1, x2, x3;
    rocblas_half2 z0, z1, z2, z3;

    auto tid = t8id * 8;
    if(tid < n_mlt_8)
    {
        //
        // Cast to rocblas_half8.
        // The reason rocblas_half8 does not appear in the signature
        // is due to the generalization of the non-batched/batched/strided batched case.
        // But the purpose of this routine is to specifically doing calculation with rocblas_half8 but also being general.
        // Then we can consider it is acceptable.
        //
        const rocblas_half8* ax
            = (const rocblas_half8*)load_ptr_batch(x, hipBlockIdx_y, offset_x + tid, stride_x);
        rocblas_half8* ay
            = (rocblas_half8*)load_ptr_batch(y, hipBlockIdx_y, offset_y + tid, stride_y);

        y0[0] = (*ay)[0];
        y0[1] = (*ay)[1];
        y1[0] = (*ay)[2];
        y1[1] = (*ay)[3];
        y2[0] = (*ay)[4];
        y2[1] = (*ay)[5];
        y3[0] = (*ay)[6];
        y3[1] = (*ay)[7];

        x0[0] = (*ax)[0];
        x0[1] = (*ax)[1];
        x1[0] = (*ax)[2];
        x1[1] = (*ax)[3];
        x2[0] = (*ax)[4];
        x2[1] = (*ax)[5];
        x3[0] = (*ax)[6];
        x3[1] = (*ax)[7];

        z0 = rocblas_fmadd_half2(alpha_h2.value, x0, y0);
        z1 = rocblas_fmadd_half2(alpha_h2.value, x1, y1);
        z2 = rocblas_fmadd_half2(alpha_h2.value, x2, y2);
        z3 = rocblas_fmadd_half2(alpha_h2.value, x3, y3);

        (*ay)[0] = z0[0];
        (*ay)[1] = z0[1];
        (*ay)[2] = z1[0];
        (*ay)[3] = z1[1];
        (*ay)[4] = z2[0];
        (*ay)[5] = z2[1];
        (*ay)[6] = z3[0];
        (*ay)[7] = z3[1];
    }
}

//!
//! @brief General template to compute y = a * x + y.
//!
template <int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   Tx             x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   Ty             y,
                                   rocblas_stride offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   rocblas_int    batch_count)
{
    if(n <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    // Using rocblas_half ?
    static constexpr bool using_rocblas_half
        //cppcheck-suppress duplicateExpression
        = std::is_same<Ta, rocblas_half>::value && std::is_same<Tex, rocblas_half>::value;

    // Using float ?
    static constexpr bool using_rocblas_float
        = std::is_same<Ty, rocblas_float*>{} || std::is_same<Ty, rocblas_float* const*>{};

    static constexpr rocblas_stride stride_0 = 0;

    //  unit_inc is True only if incx == 1  && incy == 1.
    bool unit_inc = (incx == 1 && incy == 1);

    if(using_rocblas_half && unit_inc)
    {
        //
        // Optimized version of rocblas_half, where incx == 1 and incy == 1.
        // TODO: always use an optimized version.
        //
        //
        // Note: Do not use pointer arithmetic with x and y when passing parameters.
        // The kernel will do the cast if needed.
        //
        rocblas_int n_mod_8 = n & 7; // n mod 8
        rocblas_int n_mlt_8 = n & ~(rocblas_int)7; // multiple of 8
        int         blocks  = (n / 8 - 1) / NB + 1;
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            // clang-format off
            hipLaunchKernelGGL((haxpy_mlt_8_kernel<NB>), grid, threads, 0, handle->get_stream(), n_mlt_8,
                               (const rocblas_half*)alpha, stride_alpha, x, offset_x, stride_x, y, offset_y, stride_y);
            // clang-format on
            if(n_mod_8)
            {
                //
                // cleanup non-multiple of 8
                //
                // clang-format off
                hipLaunchKernelGGL((haxpy_mod_8_kernel<NB>), dim3(1, batch_count), n_mod_8, 0, handle->get_stream(), n_mod_8,
                                    alpha, stride_alpha, x, n_mlt_8 + offset_x, stride_x, y, n_mlt_8 + offset_y, stride_y);
                // clang-format on
            }
        }
        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            hipLaunchKernelGGL((haxpy_mlt_8_kernel<NB>), grid, threads, 0, handle->get_stream(),
                                n_mlt_8,load_scalar((const rocblas_half*)alpha), stride_0, x, offset_x, stride_x, y, offset_y, stride_y);
            // clang-format on

            if(n_mod_8)
            {
                // clang-format off
                hipLaunchKernelGGL((haxpy_mod_8_kernel<NB>), dim3(1, batch_count), n_mod_8, 0, handle->get_stream(), n_mod_8,
                                   *alpha, stride_0, x, n_mlt_8 + offset_x, stride_x, y, n_mlt_8 + offset_y, stride_y);
                // clang-format on
            }
        }
    }

    else if(using_rocblas_float && unit_inc && batch_count <= 8192)
    {
        // Optimized kernel for float Datatype when incx==1 && incy==1 && batch_count <= 8192
        dim3 blocks(1 + ((n - 1) / (NB * 2)), batch_count);
        dim3 threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // clang-format off
            hipLaunchKernelGGL((saxpy_2_kernel<NB, Tex>), blocks, threads, 0, handle->get_stream(), n, alpha,
                               stride_alpha, x, offset_x, stride_x, y, offset_y, stride_y);
            // clang-format on
        }

        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            hipLaunchKernelGGL((saxpy_2_kernel<NB, Tex>), blocks, threads, 0, handle->get_stream(), n, *alpha,
                               stride_0, x, offset_x, stride_x, y, offset_y, stride_y);
            // clang-format on
        }
    }

    else if(batch_count > 8192 && std::is_same<Ta, float>::value)
    {
        // Optimized kernel for float Datatype when batch_count > 8192
        ptrdiff_t shift_x = offset_x + ((incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0);
        ptrdiff_t shift_y = offset_y + ((incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0);

        constexpr int DIM_X = 128;
        constexpr int DIM_Y = 8;

        dim3 blocks((n - 1) / (DIM_X) + 1, (batch_count - 1) / (DIM_Y * 4) + 1);
        dim3 threads(DIM_X, DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            // clang-format off
            hipLaunchKernelGGL((axpy_kernel_batched<DIM_X, DIM_Y, Tex>), blocks, threads, 0, handle->get_stream(), n, alpha,
                               stride_alpha, x, shift_x, incx, stride_x, y, shift_y, incy, stride_y, batch_count);
            // clang-format on
        }
        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            hipLaunchKernelGGL((axpy_kernel_batched<DIM_X, DIM_Y, Tex>), blocks, threads, 0, handle->get_stream(), n, *alpha,
                               stride_0, x, shift_x, incx, stride_x, y, shift_y, incy, stride_y, batch_count);
            // clang-format on
        }
    }

    else
    {
        // Default kernel for AXPY
        ptrdiff_t shift_x = offset_x + ((incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0);
        ptrdiff_t shift_y = offset_y + ((incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0);

        dim3 blocks((n - 1) / (NB) + 1, batch_count);
        dim3 threads(NB);
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            // clang-format off
            hipLaunchKernelGGL((axpy_kernel<NB, Tex>), blocks, threads, 0, handle->get_stream(), n, alpha,
                               stride_alpha, x, shift_x, incx, stride_x, y,shift_y, incy, stride_y);
            // clang-format on
        }
        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            hipLaunchKernelGGL((axpy_kernel<NB, Tex>), blocks, threads, 0, handle->get_stream(), n, *alpha,
                               stride_0, x, shift_x, incx, stride_x, y, shift_y, incy, stride_y);
            // clang-format on
        }
    }
    return rocblas_status_success;
}

template <typename T, typename U>
rocblas_status rocblas_axpy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    //constant vector `x` is checked only once if is_input is true.
    if(is_input)
    {
        rocblas_status check_numerics_status
            = rocblas_internal_check_numerics_vector_template(function_name,
                                                              handle,
                                                              n,
                                                              x,
                                                              offset_x,
                                                              inc_x,
                                                              stride_x,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          y,
                                                          offset_y,
                                                          inc_y,
                                                          stride_y,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// If there are any changes in template parameters in the files *axpy*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off
#ifdef INSTANTIATE_AXPY_TEMPLATE
#error INSTANTIATE_AXPY_TEMPLATE already defined
#endif

#define INSTANTIATE_AXPY_TEMPLATE(NB_, Tex_, Ta_, Tx_, Ty_)                 \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                   \
rocblas_status rocblas_internal_axpy_template<NB_, Tex_, Ta_, Tx_, Ty_>     \
                                              (rocblas_handle handle,       \
                                               rocblas_int    n,            \
                                               const Ta_*           alpha,  \
                                               rocblas_stride stride_alpha, \
                                               Tx_             x,           \
                                               rocblas_stride      offset_x,     \
                                               rocblas_int    incx,         \
                                               rocblas_stride stride_x,     \
                                               Ty_             y,           \
                                               rocblas_stride      offset_y,     \
                                               rocblas_int    incy,         \
                                               rocblas_stride stride_y,     \
                                               rocblas_int    batch_count);

// rocblas_Xaxpy and rocblas_axpy_ex
INSTANTIATE_AXPY_TEMPLATE(256,  float,  float, float const*, float*)
INSTANTIATE_AXPY_TEMPLATE(256, double, double, double const*, double*)
INSTANTIATE_AXPY_TEMPLATE(256, rocblas_half, rocblas_half, rocblas_half const*, rocblas_half*)
INSTANTIATE_AXPY_TEMPLATE(256, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_AXPY_TEMPLATE(256, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)

// rocblas_Xaxpy_batched and rocblas_axpy_batched_ex
INSTANTIATE_AXPY_TEMPLATE(256,  float,  float, float const* const*, float* const*)
INSTANTIATE_AXPY_TEMPLATE(256, double, double, double const* const*, double* const*)
INSTANTIATE_AXPY_TEMPLATE(256, rocblas_half, rocblas_half, rocblas_half const* const*, rocblas_half* const*)
INSTANTIATE_AXPY_TEMPLATE(256, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_AXPY_TEMPLATE(256, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const* const*, rocblas_double_complex* const*)

// rocblas_axpy_ex
INSTANTIATE_AXPY_TEMPLATE(256, float, rocblas_half, rocblas_half const*, rocblas_half*)
INSTANTIATE_AXPY_TEMPLATE(256, float, float       , rocblas_half const*, rocblas_half*)

// rocblas_axpy_batched_ex
INSTANTIATE_AXPY_TEMPLATE(256, float, rocblas_half, rocblas_half const* const*, rocblas_half* const*)
INSTANTIATE_AXPY_TEMPLATE(256, float, float       , rocblas_half const* const*, rocblas_half* const*)

#undef INSTANTIATE_AXPY_TEMPLATE

#ifdef INSTANTIATE_AXPY_CHECK_NUMERICS
#error INSTANTIATE_AXPY_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_AXPY_CHECK_NUMERICS(T_, U_)                                            \
template rocblas_status rocblas_axpy_check_numerics<T_, U_>(const char*    function_name,  \
                                                            rocblas_handle handle,         \
                                                            rocblas_int    n,              \
                                                            T_             x,              \
                                                            rocblas_stride offset_x,       \
                                                            rocblas_int    inc_x,          \
                                                            rocblas_stride stride_x,       \
                                                            U_             y,              \
                                                            rocblas_stride offset_y,       \
                                                            rocblas_int    inc_y,          \
                                                            rocblas_stride stride_y,       \
                                                            rocblas_int    batch_count,    \
                                                            const int      check_numerics, \
                                                            bool           is_input);

INSTANTIATE_AXPY_CHECK_NUMERICS(const float*, float*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const double*, double*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_half*, rocblas_half*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_float_complex*, rocblas_float_complex*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_double_complex*, rocblas_double_complex*)

INSTANTIATE_AXPY_CHECK_NUMERICS(const float* const*, float* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const double* const*, double* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_half* const*, rocblas_half* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_float_complex* const*, rocblas_float_complex* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_double_complex* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_AXPY_CHECK_NUMERICS
// clang-format on
