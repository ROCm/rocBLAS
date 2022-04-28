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
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_dgmm.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_dgmm_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<float>[] = "rocblas_sdgmm_strided_batched";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<double>[] = "rocblas_ddgmm_strided_batched";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_cdgmm_strided_batched";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zdgmm_strided_batched";

    template <typename T>
    rocblas_status rocblas_dgmm_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_side   side,
                                                     rocblas_int    m,
                                                     rocblas_int    n,
                                                     const T*       A,
                                                     rocblas_int    lda,
                                                     rocblas_stride stride_a,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             C,
                                                     rocblas_int    ldc,
                                                     rocblas_stride stride_c,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;

        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto side_letter = rocblas_side_letter(side);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_dgmm_strided_batched_name<T>,
                          side,
                          m,
                          n,
                          A,
                          lda,
                          stride_a,
                          x,
                          incx,
                          stride_x,
                          C,
                          ldc,
                          stride_c,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f dgmm_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--side",
                          side_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--lda",
                          lda,
                          "--stride_a",
                          stride_a,
                          "--incx",
                          incx,
                          "--stride_x",
                          stride_x,
                          "--ldc",
                          ldc,
                          "--stride_c",
                          stride_c,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_dgmm_strided_batched_name<T>,
                            "side",
                            side_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "--stride_a",
                            stride_a,
                            "incx",
                            incx,
                            "--stride_x",
                            stride_x,
                            "ldc",
                            ldc,
                            "--stride_c",
                            stride_c,
                            "--batch_count",
                            batch_count);
            }
        }

        if(m < 0 || n < 0 || ldc < m || lda < m || batch_count < 0)
            return rocblas_status_invalid_size;

        if(!m || !n || !batch_count)
            return rocblas_status_success;

        if(!A || !C || !x)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_stride offset_a = 0, offset_x = 0, offset_c = 0;

        return rocblas_dgmm_template(handle,
                                     side,
                                     m,
                                     n,
                                     A,
                                     offset_a,
                                     lda,
                                     stride_a,
                                     x,
                                     offset_x,
                                     incx,
                                     stride_x,
                                     C,
                                     offset_c,
                                     ldc,
                                     stride_c,
                                     batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle handle,            \
                                 rocblas_side   side,              \
                                 rocblas_int    m,                 \
                                 rocblas_int    n,                 \
                                 const T_*      A,                 \
                                 rocblas_int    lda,               \
                                 rocblas_stride stride_a,          \
                                 const T_*      x,                 \
                                 rocblas_int    incx,              \
                                 rocblas_stride stride_x,          \
                                 T_*            C,                 \
                                 rocblas_int    ldc,               \
                                 rocblas_stride stride_c,          \
                                 rocblas_int    batch_count)       \
    try                                                            \
    {                                                              \
        return rocblas_dgmm_strided_batched_impl<T_>(handle,       \
                                                     side,         \
                                                     m,            \
                                                     n,            \
                                                     A,            \
                                                     lda,          \
                                                     stride_a,     \
                                                     x,            \
                                                     incx,         \
                                                     stride_x,     \
                                                     C,            \
                                                     ldc,          \
                                                     stride_c,     \
                                                     batch_count); \
    }                                                              \
    catch(...)                                                     \
    {                                                              \
        return exception_to_rocblas_status();                      \
    }

IMPL(rocblas_sdgmm_strided_batched, float);
IMPL(rocblas_ddgmm_strided_batched, double);
IMPL(rocblas_cdgmm_strided_batched, rocblas_float_complex);
IMPL(rocblas_zdgmm_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
