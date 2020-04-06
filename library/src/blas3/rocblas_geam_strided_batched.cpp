/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_geam.hpp"
#include "utility.h"

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

        auto layer_mode = handle->layer_mode;

        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto transA_letter = rocblas_transpose_letter(transA);
            auto transB_letter = rocblas_transpose_letter(transB);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_geam_strided_batched_name<T>,
                              transA,
                              transB,
                              m,
                              n,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              stride_a,
                              log_trace_scalar_value(beta),
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
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              stride_a,
                              LOG_BENCH_SCALAR_VALUE(beta),
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
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_geam_strided_batched_name<T>,
                              transA,
                              transB,
                              m,
                              n,
                              alpha,
                              A,
                              lda,
                              stride_a,
                              beta,
                              B,
                              ldb,
                              stride_b,
                              C,
                              ldc,
                              stride_c,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
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

        static constexpr rocblas_int offset_a = 0, offset_b = 0, offset_c = 0;

        return rocblas_geam_template(handle,
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
