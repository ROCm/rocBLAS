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

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_spr2.hpp"

template <typename T>
__device__ void rocblas_spr2_kernel_calc(bool        is_upper,
                                         rocblas_int n,
                                         T           alpha,
                                         const T*    x,
                                         rocblas_int incx,
                                         const T*    y,
                                         rocblas_int incy,
                                         T*          AP)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int index = is_upper ? ((ty * (ty + 1)) / 2) + tx : ((ty * (2 * n - ty + 1)) / 2) + (tx - ty);

    if(is_upper ? ty < n && tx <= ty : tx < n && ty <= tx)
        AP[index] += alpha * x[tx * incx] * y[ty * incy] + alpha * y[tx * incy] * x[ty * incx];
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TStruct, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_spr2_kernel(bool           host_ptr_mode,
                    bool           is_upper,
                    rocblas_int    n,
                    TStruct        alpha_device_host,
                    TConstPtr      xa,
                    rocblas_stride shift_x,
                    rocblas_int    incx,
                    rocblas_stride stride_x,
                    TConstPtr      ya,
                    rocblas_stride shift_y,
                    rocblas_int    incy,
                    rocblas_stride stride_y,
                    TPtr           APa,
                    rocblas_stride shift_A,
                    rocblas_stride stride_A)
{
    auto alpha = host_ptr_mode ? alpha_device_host.value : load_scalar(alpha_device_host.ptr);
    if(!alpha)
        return;

    auto*       AP = load_ptr_batch(APa, blockIdx.z, shift_A, stride_A);
    const auto* x  = load_ptr_batch(xa, blockIdx.z, shift_x, stride_x);
    const auto* y  = load_ptr_batch(ya, blockIdx.z, shift_y, stride_y);

    rocblas_spr2_kernel_calc(is_upper, n, alpha, x, incx, y, incy, AP);
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the bast type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_spr2_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     TScal const*   alpha,
                                     TConstPtr      x,
                                     rocblas_stride offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     TConstPtr      y,
                                     rocblas_stride offset_y,
                                     rocblas_int    incy,
                                     rocblas_stride stride_y,
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
    ptrdiff_t shift_y = incy < 0 ? offset_y - ptrdiff_t(incy) * (n - 1) : offset_y;

    static constexpr int SPR2_DIM_X = 128;
    static constexpr int SPR2_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / SPR2_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / SPR2_DIM_Y + 1;

    dim3 spr2_grid(blocksX, blocksY, batch_count);
    dim3 spr2_threads(SPR2_DIM_X, SPR2_DIM_Y);

    bool                            host_mode = handle->pointer_mode == rocblas_pointer_mode_host;
    rocblas_internal_val_ptr<TScal> alpha_device_host(host_mode, alpha);

    ROCBLAS_LAUNCH_KERNEL((rocblas_spr2_kernel<SPR2_DIM_X, SPR2_DIM_Y>),
                          spr2_grid,
                          spr2_threads,
                          0,
                          handle->get_stream(),
                          host_mode,
                          uplo == rocblas_fill_upper,
                          n,
                          alpha_device_host,
                          x,
                          shift_x,
                          incx,
                          stride_x,
                          y,
                          shift_y,
                          incy,
                          stride_y,
                          AP,
                          offset_A,
                          stride_A);

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_sp_matrix_template for checking Matrix `A` which is a Symmetric Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_spr2_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           U              x,
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

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
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

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *spr2*.cpp

// clang-format off

#ifdef INSTANTIATE_SPR2_TEMPLATE
#error INSTANTIATE_SPR2_TEMPLATE already defined
#endif

#define INSTANTIATE_SPR2_TEMPLATE(TScal_, TConstPtr_, TPtr_)             \
template rocblas_status rocblas_spr2_template<TScal_, TConstPtr_, TPtr_> \
                                    (rocblas_handle handle,              \
                                     rocblas_fill   uplo,                \
                                     rocblas_int    n,                   \
                                     TScal_ const * alpha,               \
                                     TConstPtr_      x,                  \
                                     rocblas_stride    offset_x,            \
                                     rocblas_int    incx,                \
                                     rocblas_stride stride_x,            \
                                     TConstPtr_      y,                  \
                                     rocblas_stride    offset_y,            \
                                     rocblas_int    incy,                \
                                     rocblas_stride stride_y,            \
                                     TPtr_           AP,                 \
                                     rocblas_stride    offset_A,            \
                                     rocblas_stride stride_A,            \
                                     rocblas_int    batch_count);

INSTANTIATE_SPR2_TEMPLATE(float , float const*, float*)
INSTANTIATE_SPR2_TEMPLATE(double , double const*, double*)
INSTANTIATE_SPR2_TEMPLATE(float , float const* const*, float* const*)
INSTANTIATE_SPR2_TEMPLATE(double , double const* const*, double* const*)

#undef INSTANTIATE_SPR2_TEMPLATE

#ifdef INSTANTIATE_SPR2_NUMERICS
#error INSTANTIATE_SPR2_NUMERICS already defined
#endif

#define INSTANTIATE_SPR2_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_spr2_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_int    n,              \
                                           T_             A,              \
                                           rocblas_stride    offset_a,       \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_stride    offset_x,       \
                                           rocblas_int    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           U_             y,              \
                                           rocblas_stride    offset_y,       \
                                           rocblas_int    inc_y,          \
                                           rocblas_stride stride_y,       \
                                           rocblas_int    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_SPR2_NUMERICS(float*, float const*)
INSTANTIATE_SPR2_NUMERICS(double*, double const*)
INSTANTIATE_SPR2_NUMERICS(float* const*, float const* const*)
INSTANTIATE_SPR2_NUMERICS(double* const*, double const* const*)

#undef INSTANTIATE_SPR2_NUMERICS

// clang-format on
