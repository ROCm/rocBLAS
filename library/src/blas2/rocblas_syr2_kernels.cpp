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
#include "rocblas.h"
#include "rocblas_syr2.hpp"

template <typename T>
__device__ void rocblas_syr2_kernel_calc(bool        is_upper,
                                         rocblas_int n,
                                         T           alpha,
                                         const T*    x,
                                         rocblas_int incx,
                                         const T*    y,
                                         rocblas_int incy,
                                         T*          A,
                                         rocblas_int lda)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(is_upper ? ty < n && tx <= ty : tx < n && ty <= tx)
        A[tx + ty * lda]
            += alpha * x[tx * incx] * y[ty * incy] + alpha * y[tx * incy] * x[ty * incx];
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_syr2_kernel(bool           is_upper,
                    rocblas_int    n,
                    TScal          alphaa,
                    TConstPtr      xa,
                    rocblas_stride shift_x,
                    rocblas_int    incx,
                    rocblas_stride stride_x,
                    TConstPtr      ya,
                    rocblas_stride shift_y,
                    rocblas_int    incy,
                    rocblas_stride stride_y,
                    TPtr           Aa,
                    rocblas_int    lda,
                    rocblas_stride shift_A,
                    rocblas_stride stride_A)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the number of threads as template parameters indicate.

    auto alpha = load_scalar(alphaa);
    if(!alpha)
        return;

    auto*       A = load_ptr_batch(Aa, blockIdx.z, shift_A, stride_A);
    const auto* x = load_ptr_batch(xa, blockIdx.z, shift_x, stride_x);
    const auto* y = load_ptr_batch(ya, blockIdx.z, shift_y, stride_y);

    rocblas_syr2_kernel_calc(is_upper, n, alpha, x, incx, y, incy, A, lda);
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the bast type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_syr2_template(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              TScal          alpha,
                                              TConstPtr      x,
                                              rocblas_stride offset_x,
                                              rocblas_int    incx,
                                              rocblas_stride stride_x,
                                              TConstPtr      y,
                                              rocblas_stride offset_y,
                                              rocblas_int    incy,
                                              rocblas_stride stride_y,
                                              TPtr           A,
                                              rocblas_int    lda,
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

    static constexpr int SYR2_DIM_X = 128;
    static constexpr int SYR2_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / SYR2_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / SYR2_DIM_Y + 1;

    dim3 syr2_grid(blocksX, blocksY, batch_count);
    dim3 syr2_threads(SYR2_DIM_X, SYR2_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_syr2_kernel<SYR2_DIM_X, SYR2_DIM_Y>),
                           syr2_grid,
                           syr2_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           y,
                           shift_y,
                           incy,
                           stride_y,
                           A,
                           lda,
                           offset_A,
                           stride_A);
    else
        hipLaunchKernelGGL((rocblas_syr2_kernel<SYR2_DIM_X, SYR2_DIM_Y>),
                           syr2_grid,
                           syr2_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           *alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           y,
                           shift_y,
                           incy,
                           stride_y,
                           A,
                           lda,
                           offset_A,
                           stride_A);

    return rocblas_status_success;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T*       x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   const T*       y,
                                   rocblas_stride offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   T*             A,
                                   rocblas_int    lda,
                                   rocblas_stride offset_A,
                                   rocblas_stride stride_A,
                                   rocblas_int    batch_count)
{
    return rocblas_internal_syr2_template<const T*>(handle,
                                                    uplo,
                                                    n,
                                                    alpha,
                                                    x,
                                                    offset_x,
                                                    incx,
                                                    stride_x,
                                                    y,
                                                    offset_y,
                                                    incy,
                                                    stride_y,
                                                    A,
                                                    lda,
                                                    offset_A,
                                                    stride_A,
                                                    batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2_batched_template(rocblas_handle  handle,
                                           rocblas_fill    uplo,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           const T* const* x,
                                           rocblas_stride  offset_x,
                                           rocblas_int     incx,
                                           rocblas_stride  stride_x,
                                           const T* const* y,
                                           rocblas_stride  offset_y,
                                           rocblas_int     incy,
                                           rocblas_stride  stride_y,
                                           T* const*       A,
                                           rocblas_int     lda,
                                           rocblas_stride  offset_A,
                                           rocblas_stride  stride_A,
                                           rocblas_int     batch_count)
{
    return rocblas_internal_syr2_template(handle,
                                          uplo,
                                          n,
                                          alpha,
                                          x,
                                          offset_x,
                                          incx,
                                          stride_x,
                                          y,
                                          offset_y,
                                          incy,
                                          stride_y,
                                          A,
                                          lda,
                                          offset_A,
                                          stride_A,
                                          batch_count);
}

template <typename T, typename U>
rocblas_status rocblas_syr2_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_int    lda,
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
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          uplo,
                                                          rocblas_client_symmetric_matrix,
                                                          n,
                                                          n,
                                                          A,
                                                          offset_a,
                                                          lda,
                                                          stride_a,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    if(is_input)
    {
        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
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
    }

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *syr2*.cpp

// clang-format off

#ifdef INSTANTIATE_SYR2_TEMPLATE
#error INSTANTIATE_SYR2_TEMPLATE already defined
#endif

#define INSTANTIATE_SYR2_TEMPLATE(T_)    \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status        \
    rocblas_internal_syr2_template<T_>                          \
                                  (rocblas_handle handle,       \
                                   rocblas_fill   uplo,         \
                                   rocblas_int    n,            \
                                   const T_*      alpha,        \
                                   const T_*      x,            \
                                   rocblas_stride offset_x,     \
                                   rocblas_int    incx,         \
                                   rocblas_stride stride_x,     \
                                   const T_*      y,            \
                                   rocblas_stride offset_y,     \
                                   rocblas_int    incy,         \
                                   rocblas_stride stride_y,     \
                                   T_*            A,            \
                                   rocblas_int    lda,          \
                                   rocblas_stride offset_A,     \
                                   rocblas_stride stride_A,     \
                                   rocblas_int    batch_count);

INSTANTIATE_SYR2_TEMPLATE(float)
INSTANTIATE_SYR2_TEMPLATE(double)
INSTANTIATE_SYR2_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYR2_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYR2_TEMPLATE

#ifdef INSTANTIATE_SYR2_BATCHED_TEMPLATE
#error INSTANTIATE_SYR2_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_SYR2_BATCHED_TEMPLATE(T_)    \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status         \
    rocblas_internal_syr2_batched_template<T_>                   \
                                  (rocblas_handle   handle,       \
                                   rocblas_fill     uplo,         \
                                   rocblas_int      n,            \
                                   const T_*        alpha,        \
                                   const T_* const* x,            \
                                   rocblas_stride   offset_x,     \
                                   rocblas_int      incx,         \
                                   rocblas_stride   stride_x,     \
                                   const T_* const* y,            \
                                   rocblas_stride   offset_y,     \
                                   rocblas_int      incy,         \
                                   rocblas_stride   stride_y,     \
                                   T_* const*       A,            \
                                   rocblas_int      lda,          \
                                   rocblas_stride   offset_A,     \
                                   rocblas_stride   stride_A,     \
                                   rocblas_int      batch_count);

INSTANTIATE_SYR2_BATCHED_TEMPLATE(float)
INSTANTIATE_SYR2_BATCHED_TEMPLATE(double)
INSTANTIATE_SYR2_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYR2_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYR2_BATCHED_TEMPLATE

#ifdef INSTANTIATE_SYR2_NUMERICS
#error INSTANTIATE_SYR2_NUMERICS already defined
#endif

#define INSTANTIATE_SYR2_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_syr2_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_fill   uplo,           \
                                           rocblas_int    n,              \
                                           T_             A,              \
                                           rocblas_stride    offset_a,    \
                                           rocblas_int    lda,            \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_stride    offset_x,    \
                                           rocblas_int    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           U_             y,              \
                                           rocblas_stride    offset_y,    \
                                           rocblas_int    inc_y,          \
                                           rocblas_stride stride_y,       \
                                           rocblas_int    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_SYR2_NUMERICS(float*, float const*)
INSTANTIATE_SYR2_NUMERICS(double*, double const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INSTANTIATE_SYR2_NUMERICS(float* const*, float const* const*)
INSTANTIATE_SYR2_NUMERICS(double* const*, double const* const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INSTANTIATE_SYR2_NUMERICS

// clang-format on
