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
#include "rocblas_geam.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_geam_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_geam_batched_name<float>[] = ROCBLAS_API_STR(rocblas_sgeam_batched);
    template <>
    constexpr char rocblas_geam_batched_name<double>[] = ROCBLAS_API_STR(rocblas_dgeam_batched);
    template <>
    constexpr char rocblas_geam_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cgeam_batched);
    template <>
    constexpr char rocblas_geam_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zgeam_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_geam_batched_impl(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_operation transB,
                                             API_INT           m,
                                             API_INT           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             API_INT           lda,
                                             const T*          beta,
                                             const T* const    B[],
                                             API_INT           ldb,
                                             T* const          C[],
                                             API_INT           ldc,
                                             API_INT           batch_count)
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
            auto transA_letter = rocblas_transpose_letter(transA);
            auto transB_letter = rocblas_transpose_letter(transB);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_geam_batched_name<T>,
                          transA,
                          transB,
                          m,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          A,
                          lda,
                          LOG_TRACE_SCALAR_VALUE(handle, beta),
                          B,
                          ldb,
                          C,
                          ldc,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          ROCBLAS_API_BENCH " -f geam_batched -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          transA_letter,
                          "--transposeB",
                          transB_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          LOG_BENCH_SCALAR_VALUE(handle, alpha),
                          "--lda",
                          lda,
                          LOG_BENCH_SCALAR_VALUE(handle, beta),
                          "--ldb",
                          ldb,
                          "--ldc",
                          ldc,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_geam_batched_name<T>,
                            "transA",
                            transA_letter,
                            "transB",
                            transB_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "ldb",
                            ldb,
                            "ldc",
                            ldc,
                            "--batch_count",
                            batch_count);
        }

        static constexpr rocblas_stride offset_a = 0, offset_b = 0, offset_c = 0;
        static constexpr rocblas_stride stride_a = 0, stride_b = 0, stride_c = 0;

        rocblas_status arg_status = rocblas_geam_arg_check<API_INT>(
            handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status geam_check_numerics_status
                = rocblas_geam_check_numerics(rocblas_geam_batched_name<T>,
                                              handle,
                                              transA,
                                              transB,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              stride_a,
                                              B,
                                              ldb,
                                              stride_b,
                                              C,
                                              ldc,
                                              stride_c,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(geam_check_numerics_status != rocblas_status_success)
                return geam_check_numerics_status;
        }
        rocblas_status status = rocblas_status_success;

        status = ROCBLAS_API(rocblas_geam_launcher)(handle,
                                                    transA,
                                                    transB,
                                                    m,
                                                    n,
                                                    alpha,
                                                    A,
                                                    offset_a,
                                                    lda,
                                                    stride_a,
                                                    beta,
                                                    B,
                                                    offset_b,
                                                    ldb,
                                                    stride_b,
                                                    C,
                                                    offset_c,
                                                    ldc,
                                                    stride_c,
                                                    batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status geam_check_numerics_status
                = rocblas_geam_check_numerics(rocblas_geam_batched_name<T>,
                                              handle,
                                              transA,
                                              transB,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              stride_a,
                                              B,
                                              ldb,
                                              stride_b,
                                              C,
                                              ldc,
                                              stride_c,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(geam_check_numerics_status != rocblas_status_success)
                return geam_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                         \
    rocblas_status routine_name_(rocblas_handle    handle,                                   \
                                 rocblas_operation transA,                                   \
                                 rocblas_operation transB,                                   \
                                 TI_               m,                                        \
                                 TI_               n,                                        \
                                 const T_*         alpha,                                    \
                                 const T_* const   A[],                                      \
                                 TI_               lda,                                      \
                                 const T_*         beta,                                     \
                                 const T_* const   B[],                                      \
                                 TI_               ldb,                                      \
                                 T_* const         C[],                                      \
                                 TI_               ldc,                                      \
                                 TI_               batch_count)                              \
    try                                                                                      \
    {                                                                                        \
        return rocblas_geam_batched_impl<TI_, T_>(                                           \
            handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batch_count); \
    }                                                                                        \
    catch(...)                                                                               \
    {                                                                                        \
        return exception_to_rocblas_status();                                                \
    }

#define INST_GEAM_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                           \
    IMPL(ROCBLAS_API(rocblas_sgeam_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dgeam_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cgeam_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zgeam_batched), TI_, rocblas_double_complex); \
    } // extern "C"
