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
#include "rocblas_symm_hemm.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_symm_name[] = "unknown";
    template <>
    constexpr char rocblas_symm_name<float>[] = ROCBLAS_API_STR(rocblas_ssymm_strided_batched);
    template <>
    constexpr char rocblas_symm_name<double>[] = ROCBLAS_API_STR(rocblas_dsymm_strided_batched);
    template <>
    constexpr char rocblas_symm_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_csymm_strided_batched);
    template <>
    constexpr char rocblas_symm_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zsymm_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_symm_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_side   side,
                                                     rocblas_fill   uplo,
                                                     API_INT        m,
                                                     API_INT        n,
                                                     const T*       alpha,
                                                     const T*       A,
                                                     API_INT        lda,
                                                     rocblas_stride stride_a,
                                                     const T*       B,
                                                     API_INT        ldb,
                                                     rocblas_stride stride_b,
                                                     const T*       beta,
                                                     T*             C,
                                                     API_INT        ldc,
                                                     rocblas_stride stride_c,
                                                     API_INT        batch_count)
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
            auto uplo_letter = rocblas_fill_letter(uplo);
            auto side_letter = rocblas_side_letter(side);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_symm_name<T>,
                          side,
                          uplo,
                          m,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          A,
                          lda,
                          stride_a,
                          B,
                          ldb,
                          stride_b,
                          LOG_TRACE_SCALAR_VALUE(handle, beta),
                          C,
                          ldc,
                          stride_c,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          ROCBLAS_API_BENCH " -f symm_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--side",
                          side_letter,
                          "--uplo",
                          uplo_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          LOG_BENCH_SCALAR_VALUE(handle, alpha),
                          "--lda",
                          lda,
                          "--stride_a",
                          stride_a,
                          "--ldb",
                          ldb,
                          "--stride_b",
                          stride_b,
                          LOG_BENCH_SCALAR_VALUE(handle, beta),
                          "--ldc",
                          ldc,
                          "--stride_c",
                          stride_c,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_symm_name<T>,
                            "side",
                            side_letter,
                            "uplo",
                            uplo_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "stride_a",
                            stride_a,
                            "ldb",
                            ldb,
                            "stride_b",
                            stride_b,
                            "ldc",
                            ldc,
                            "stride_c",
                            stride_c,
                            "batch_count",
                            batch_count);
        }

        static constexpr rocblas_stride offset_C = 0, offset_A = 0, offset_B = 0;

        rocblas_status arg_status = rocblas_symm_arg_check<API_INT>(handle,
                                                                    side,
                                                                    uplo,
                                                                    m,
                                                                    n,
                                                                    alpha,
                                                                    A,
                                                                    offset_A,
                                                                    lda,
                                                                    stride_a,
                                                                    B,
                                                                    offset_B,
                                                                    ldb,
                                                                    stride_b,
                                                                    beta,
                                                                    C,
                                                                    offset_C,
                                                                    ldc,
                                                                    stride_c,
                                                                    batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        static constexpr bool HERMITIAN = false;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status symm_check_numerics_status
                = rocblas_hemm_symm_check_numerics<HERMITIAN>(rocblas_symm_name<T>,
                                                              handle,
                                                              side,
                                                              uplo,
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

            if(symm_check_numerics_status != rocblas_status_success)
                return symm_check_numerics_status;
        }

        rocblas_status status = rocblas_status_success;

        status = ROCBLAS_API(rocblas_internal_symm_template)<T>(handle,
                                                                side,
                                                                uplo,
                                                                m,
                                                                n,
                                                                alpha,
                                                                A,
                                                                offset_A,
                                                                lda,
                                                                stride_a,
                                                                B,
                                                                offset_B,
                                                                ldb,
                                                                stride_b,
                                                                beta,
                                                                C,
                                                                offset_C,
                                                                ldc,
                                                                stride_c,
                                                                batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status symm_check_numerics_status
                = rocblas_hemm_symm_check_numerics<HERMITIAN>(rocblas_symm_name<T>,
                                                              handle,
                                                              side,
                                                              uplo,
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

            if(symm_check_numerics_status != rocblas_status_success)
                return symm_check_numerics_status;
        }
        return status;
    }

}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, TI_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle handle,                 \
                                 rocblas_side   side,                   \
                                 rocblas_fill   uplo,                   \
                                 TI_            m,                      \
                                 TI_            n,                      \
                                 const T_*      alpha,                  \
                                 const T_*      A,                      \
                                 TI_            lda,                    \
                                 rocblas_stride stride_a,               \
                                 const T_*      B,                      \
                                 TI_            ldb,                    \
                                 rocblas_stride stride_b,               \
                                 const T_*      beta,                   \
                                 T_*            C,                      \
                                 TI_            ldc,                    \
                                 rocblas_stride stride_c,               \
                                 TI_            batch_count)            \
    try                                                                 \
    {                                                                   \
        return rocblas_symm_strided_batched_impl<TI_, T_>(handle,       \
                                                          side,         \
                                                          uplo,         \
                                                          m,            \
                                                          n,            \
                                                          alpha,        \
                                                          A,            \
                                                          lda,          \
                                                          stride_a,     \
                                                          B,            \
                                                          ldb,          \
                                                          stride_b,     \
                                                          beta,         \
                                                          C,            \
                                                          ldc,          \
                                                          stride_c,     \
                                                          batch_count); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

#define INST_SYMM_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_ssymm_strided_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dsymm_strided_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_csymm_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zsymm_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
