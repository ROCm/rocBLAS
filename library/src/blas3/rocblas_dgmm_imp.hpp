/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_dgmm.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_dgmm_name[] = "unknown";
    template <>
    constexpr char rocblas_dgmm_name<float>[] = ROCBLAS_API_STR(rocblas_sdgmm);
    template <>
    constexpr char rocblas_dgmm_name<double>[] = ROCBLAS_API_STR(rocblas_ddgmm);
    template <>
    constexpr char rocblas_dgmm_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_cdgmm);
    template <>
    constexpr char rocblas_dgmm_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_zdgmm);

    template <typename API_INT, typename T>
    rocblas_status rocblas_dgmm_impl(rocblas_handle handle,
                                     rocblas_side   side,
                                     API_INT        m,
                                     API_INT        n,
                                     const T*       A,
                                     API_INT        lda,
                                     const T*       x,
                                     API_INT        incx,
                                     T*             C,
                                     API_INT        ldc)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;

        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto side_letter = rocblas_side_letter(side);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_dgmm_name<T>, side, m, n, A, lda, x, incx, C, ldc);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          ROCBLAS_API_BENCH " -f dgmm -r",
                          rocblas_precision_string<T>,
                          "--side",
                          side_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--lda",
                          lda,
                          "--incx",
                          incx,
                          "--ldc",
                          ldc);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_dgmm_name<T>,
                            "side",
                            side_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "ldc",
                            ldc);
        }

        static constexpr rocblas_stride offset_A = 0, offset_x = 0, offset_C = 0;
        static constexpr rocblas_int    batch_count = 1;
        static constexpr rocblas_stride stride_A = 0, stride_x = 0, stride_C = 0;

        rocblas_status arg_status = rocblas_dgmm_arg_check<API_INT>(
            handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status dgmm_check_numerics_status
                = rocblas_dgmm_check_numerics(rocblas_dgmm_name<T>,
                                              handle,
                                              side,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              stride_A,
                                              x,
                                              incx,
                                              stride_x,
                                              C,
                                              ldc,
                                              stride_C,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(dgmm_check_numerics_status != rocblas_status_success)
                return dgmm_check_numerics_status;
        }

        rocblas_status status = rocblas_status_success;
        status                = ROCBLAS_API(rocblas_internal_dgmm_launcher)(handle,
                                                             side,
                                                             m,
                                                             n,
                                                             A,
                                                             offset_A,
                                                             lda,
                                                             stride_A,
                                                             x,
                                                             offset_x,
                                                             incx,
                                                             stride_x,
                                                             C,
                                                             offset_C,
                                                             ldc,
                                                             stride_C,
                                                             batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status dgmm_check_numerics_status
                = rocblas_dgmm_check_numerics(rocblas_dgmm_name<T>,
                                              handle,
                                              side,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              stride_A,
                                              x,
                                              incx,
                                              stride_x,
                                              C,
                                              ldc,
                                              stride_C,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(dgmm_check_numerics_status != rocblas_status_success)
                return dgmm_check_numerics_status;
        }
        return status;
    }
} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, TI_, T_)                                                    \
    rocblas_status routine_name_(rocblas_handle handle,                                 \
                                 rocblas_side   side,                                   \
                                 TI_            m,                                      \
                                 TI_            n,                                      \
                                 const T_*      A,                                      \
                                 TI_            lda,                                    \
                                 const T_*      x,                                      \
                                 TI_            incx,                                   \
                                 T_*            C,                                      \
                                 TI_            ldc)                                    \
    try                                                                                 \
    {                                                                                   \
        return rocblas_dgmm_impl<TI_, T_>(handle, side, m, n, A, lda, x, incx, C, ldc); \
    }                                                                                   \
    catch(...)                                                                          \
    {                                                                                   \
        return exception_to_rocblas_status();                                           \
    }

#define INST_DGMM_C_API(TI_)                                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_sdgmm), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_ddgmm), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cdgmm), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zdgmm), TI_, rocblas_double_complex); \
    } // extern "C"
