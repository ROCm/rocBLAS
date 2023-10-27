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

#include "rocblas_axpy_kernels.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_axpy.hpp"
#include "rocblas_block_sizes.h"

template <typename T, typename U>
rocblas_status rocblas_axpy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
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

#ifdef INSTANTIATE_AXPY_CHECK_NUMERICS
#error INSTANTIATE_AXPY_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_AXPY_CHECK_NUMERICS(T_, U_)                                                \
    template rocblas_status rocblas_axpy_check_numerics<T_, U_>(const char*    function_name,  \
                                                                rocblas_handle handle,         \
                                                                int64_t        n,              \
                                                                T_             x,              \
                                                                rocblas_stride offset_x,       \
                                                                int64_t        inc_x,          \
                                                                rocblas_stride stride_x,       \
                                                                U_             y,              \
                                                                rocblas_stride offset_y,       \
                                                                int64_t        inc_y,          \
                                                                rocblas_stride stride_y,       \
                                                                int64_t        batch_count,    \
                                                                const int      check_numerics, \
                                                                bool           is_input);

INSTANTIATE_AXPY_CHECK_NUMERICS(const float*, float*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const double*, double*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_half*, rocblas_half*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_bfloat16*, rocblas_bfloat16*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_float_complex*, rocblas_float_complex*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_double_complex*, rocblas_double_complex*)

INSTANTIATE_AXPY_CHECK_NUMERICS(const float* const*, float* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const double* const*, double* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_half* const*, rocblas_half* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_bfloat16* const*, rocblas_bfloat16* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_float_complex* const*, rocblas_float_complex* const*)
INSTANTIATE_AXPY_CHECK_NUMERICS(const rocblas_double_complex* const*,
                                rocblas_double_complex* const*)

#undef INSTANTIATE_AXPY_CHECK_NUMERICS

#ifdef INSTANTIATE_AXPY_LAUNCHER
#error INSTANTIATE_AXPY_LAUNCHER already defined
#endif

#define INSTANTIATE_AXPY_LAUNCHER(NB_, Tex_, Ta_, Tx_, Ty_)                                        \
    template rocblas_status rocblas_internal_axpy_launcher<rocblas_int, NB_, Tex_, Ta_, Tx_, Ty_>( \
        rocblas_handle handle,                                                                     \
        rocblas_int    n,                                                                          \
        Ta_            alpha,                                                                      \
        rocblas_stride stride_alpha,                                                               \
        Tx_            x,                                                                          \
        rocblas_stride offset_x,                                                                   \
        rocblas_int    incx,                                                                       \
        rocblas_stride stride_x,                                                                   \
        Ty_            y,                                                                          \
        rocblas_stride offset_y,                                                                   \
        rocblas_int    incy,                                                                       \
        rocblas_stride stride_y,                                                                   \
        rocblas_int    batch_count);

// non-batched
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, rocblas_half, const rocblas_half*, const rocblas_half*, rocblas_half*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB, float, const float*, const float*, float*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB, double, const double*, const double*, double*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          rocblas_float_complex,
                          const rocblas_float_complex*,
                          const rocblas_float_complex*,
                          rocblas_float_complex*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          rocblas_double_complex,
                          const rocblas_double_complex*,
                          const rocblas_double_complex*,
                          rocblas_double_complex*)

// batched
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          rocblas_half,
                          const rocblas_half*,
                          const rocblas_half* const*,
                          rocblas_half* const*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB, float, const float*, const float* const*, float* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, double, const double*, const double* const*, double* const*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          rocblas_float_complex,
                          const rocblas_float_complex*,
                          const rocblas_float_complex* const*,
                          rocblas_float_complex* const*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          rocblas_double_complex,
                          const rocblas_double_complex*,
                          const rocblas_double_complex* const*,
                          rocblas_double_complex* const*)

// axpy_ex
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const rocblas_bfloat16*, const rocblas_bfloat16*, rocblas_bfloat16*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const float*, const rocblas_bfloat16*, rocblas_bfloat16*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const rocblas_half*, const rocblas_half*, rocblas_half*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB, float, const float*, const rocblas_half*, rocblas_half*)

// axpy_batched_ex
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          float,
                          const rocblas_bfloat16*,
                          const rocblas_bfloat16* const*,
                          rocblas_bfloat16* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const float*, const rocblas_bfloat16* const*, rocblas_bfloat16* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const rocblas_half*, const rocblas_half* const*, rocblas_half* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const float*, const rocblas_half* const*, rocblas_half* const*)

#undef INSTANTIATE_AXPY_LAUNCHER
// clang-format on
