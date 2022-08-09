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
#include "rocblas_hpr.hpp"

template <typename T, typename U>
__device__ void
    hpr_kernel_calc(bool upper, rocblas_int n, U alpha, const T* x, rocblas_int incx, T* AP)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int index = upper ? ((ty * (ty + 1)) / 2) + tx : ((ty * (2 * n - ty + 1)) / 2) + (tx - ty);

    if(upper ? ty < n && tx < ty : tx < n && ty < tx)
        AP[index] += alpha * x[tx * incx] * conj(x[ty * incx]);
    else if(tx == ty && tx < n)
    {
        U x_real  = std::real(x[tx * incx]);
        U x_imag  = std::imag(x[tx * incx]);
        AP[index] = std::real(AP[index]) + alpha * ((x_real * x_real) + (x_imag * x_imag));
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_hpr_kernel(bool           upper,
                   rocblas_int    n,
                   TScal          alphaa,
                   TConstPtr      xa,
                   rocblas_stride shift_x,
                   rocblas_int    incx,
                   rocblas_stride stride_x,
                   TPtr           APa,
                   rocblas_stride shift_A,
                   rocblas_stride stride_A)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the number of threads as template parameters indicate.

    auto alpha = load_scalar(alphaa);
    if(!alpha)
        return;

    auto*       AP = load_ptr_batch(APa, blockIdx.z, shift_A, stride_A);
    const auto* x  = load_ptr_batch(xa, blockIdx.z, shift_x, stride_x);

    hpr_kernel_calc(upper, n, alpha, x, incx, AP);
}

/**
 * TScal     is always: const U* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 * and U is the scalar type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hpr_template(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    rocblas_int    n,
                                    TScal          alpha,
                                    TConstPtr      x,
                                    rocblas_stride offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    TPtr           AP,
                                    rocblas_stride offset_A,
                                    rocblas_stride stride_A,
                                    rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;

    static constexpr int HPR_DIM_X = 128;
    static constexpr int HPR_DIM_Y = 8;
    rocblas_int          blocksX   = (n - 1) / HPR_DIM_X + 1;
    rocblas_int          blocksY   = (n - 1) / HPR_DIM_Y + 1;

    dim3 hpr_grid(blocksX, blocksY, batch_count);
    dim3 hpr_threads(HPR_DIM_X, HPR_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL((rocblas_hpr_kernel<HPR_DIM_X, HPR_DIM_Y>),
                           hpr_grid,
                           hpr_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           AP,
                           offset_A,
                           stride_A);
    }
    else
        hipLaunchKernelGGL((rocblas_hpr_kernel<HPR_DIM_X, HPR_DIM_Y>),
                           hpr_grid,
                           hpr_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           *alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           AP,
                           offset_A,
                           stride_A);

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_hp_matrix_template for checking Matrix `AP` which is a Hermitian Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_hpr_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          rocblas_int    n,
                                          T              AP,
                                          rocblas_stride offset_a,
                                          rocblas_stride stride_a,
                                          U              x,
                                          rocblas_stride offset_x,
                                          rocblas_int    inc_x,
                                          rocblas_stride stride_x,
                                          rocblas_int    batch_count,
                                          const int      check_numerics,
                                          bool           is_input)
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

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *hpr*.cpp

// clang-format off

#ifdef INSTANTIATE_HPR_TEMPLATE
#error INSTANTIATE_HPR_TEMPLATE already defined
#endif

#define INSTANTIATE_HPR_TEMPLATE(TScal_, TConstPtr_, TPtr_)              \
template rocblas_status rocblas_hpr_template<TScal_, TConstPtr_, TPtr_>  \
                                   (rocblas_handle handle,               \
                                    rocblas_fill   uplo,                 \
                                    rocblas_int    n,                    \
                                    TScal_          alpha,               \
                                    TConstPtr_      x,                   \
                                    rocblas_stride offset_x,             \
                                    rocblas_int    incx,                 \
                                    rocblas_stride stride_x,             \
                                    TPtr_           AP,                  \
                                    rocblas_stride  offset_A,             \
                                    rocblas_stride stride_A,             \
                                    rocblas_int    batch_count);

INSTANTIATE_HPR_TEMPLATE(float const*, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HPR_TEMPLATE(double const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HPR_TEMPLATE(float const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HPR_TEMPLATE(double const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HPR_TEMPLATE

#ifdef INSTANTIATE_HPR_NUMERICS
#error INSTANTIATE_HPR_NUMERICS already defined
#endif

#define INSTANTIATE_HPR_NUMERICS(T_, U_)                                   \
template rocblas_status rocblas_hpr_check_numerics<T_, U_>                 \
                                         (const char*    function_name,    \
                                          rocblas_handle handle,           \
                                          rocblas_int    n,                \
                                          T_             AP,               \
                                          rocblas_stride    offset_a,         \
                                          rocblas_stride stride_a,         \
                                          U_             x,                \
                                          rocblas_stride    offset_x,         \
                                          rocblas_int    inc_x,            \
                                          rocblas_stride stride_x,         \
                                          rocblas_int    batch_count,      \
                                          const int      check_numerics,   \
                                          bool           is_input);

INSTANTIATE_HPR_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INSTANTIATE_HPR_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INSTANTIATE_HPR_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INSTANTIATE_HPR_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INSTANTIATE_HPR_NUMERICS

// clang-format on
