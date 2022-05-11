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
#include "rocblas_geam.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_geam_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_geam_strided_batched_name<float>[] = "rocblas_sgeam_strided_batched";
    template <>
    constexpr char rocblas_geam_strided_batched_name<double>[] = "rocblas_dgeam_strided_batched";
    template <>
    constexpr char rocblas_geam_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_cgeam_strided_batched";
    template <>
    constexpr char rocblas_geam_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zgeam_strided_batched";

    template <typename T>
    rocblas_status rocblas_geam_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     const T*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_stride    stride_a,
                                                     const T*          beta,
                                                     const T*          B,
                                                     rocblas_int       ldb,
                                                     rocblas_stride    stride_b,
                                                     T*                C,
                                                     rocblas_int       ldc,
                                                     rocblas_stride    stride_c,
                                                     rocblas_int       batch_count)
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
                          rocblas_geam_strided_batched_name<T>,
                          transA,
                          transB,
                          m,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          A,
                          lda,
                          stride_a,
                          LOG_TRACE_SCALAR_VALUE(handle, beta),
                          B,
                          ldb,
                          stride_b,
                          C,
                          ldc,
                          stride_c,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f geam_strided_batched -r",
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
                          "--stride_a",
                          stride_a,
                          LOG_BENCH_SCALAR_VALUE(handle, beta),
                          "--ldb",
                          ldb,
                          "--stride_b",
                          stride_b,
                          "--ldc",
                          ldc,
                          "--stride_c",
                          stride_c,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_geam_strided_batched_name<T>,
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
                            "--stride_a",
                            stride_a,
                            "ldb",
                            ldb,
                            "--stride_b",
                            stride_b,
                            "ldc",
                            ldc,
                            "--stride_c",
                            stride_c,
                            "--batch_count",
                            batch_count);
        }

        if(m < 0 || n < 0 || ldc < m || lda < (transA == rocblas_operation_none ? m : n)
           || ldb < (transB == rocblas_operation_none ? m : n) || batch_count < 0)
            return rocblas_status_invalid_size;

        if(!m || !n || !batch_count)
            return rocblas_status_success;

        if(!A || !B || !C)
            return rocblas_status_invalid_pointer;

        if((C == A && (lda != ldc || transA != rocblas_operation_none))
           || (C == B && (ldb != ldc || transB != rocblas_operation_none)))
            return rocblas_status_invalid_size;

        if(!alpha || !beta)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_stride offset_a = 0, offset_b = 0, offset_c = 0;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status geam_check_numerics_status
                = rocblas_geam_check_numerics(rocblas_geam_strided_batched_name<T>,
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

        status = rocblas_geam_template(handle,
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
                = rocblas_geam_check_numerics(rocblas_geam_strided_batched_name<T>,
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

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle    handle,         \
                                 rocblas_operation transA,         \
                                 rocblas_operation transB,         \
                                 rocblas_int       m,              \
                                 rocblas_int       n,              \
                                 const T_*         alpha,          \
                                 const T_*         A,              \
                                 rocblas_int       lda,            \
                                 rocblas_stride    stride_a,       \
                                 const T_*         beta,           \
                                 const T_*         B,              \
                                 rocblas_int       ldb,            \
                                 rocblas_stride    stride_b,       \
                                 T_*               C,              \
                                 rocblas_int       ldc,            \
                                 rocblas_stride    stride_c,       \
                                 rocblas_int       batch_count)    \
    try                                                            \
    {                                                              \
        return rocblas_geam_strided_batched_impl<T_>(handle,       \
                                                     transA,       \
                                                     transB,       \
                                                     m,            \
                                                     n,            \
                                                     alpha,        \
                                                     A,            \
                                                     lda,          \
                                                     stride_a,     \
                                                     beta,         \
                                                     B,            \
                                                     ldb,          \
                                                     stride_b,     \
                                                     C,            \
                                                     ldc,          \
                                                     stride_c,     \
                                                     batch_count); \
    }                                                              \
    catch(...)                                                     \
    {                                                              \
        return exception_to_rocblas_status();                      \
    }

IMPL(rocblas_sgeam_strided_batched, float);
IMPL(rocblas_dgeam_strided_batched, double);
IMPL(rocblas_cgeam_strided_batched, rocblas_float_complex);
IMPL(rocblas_zgeam_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
