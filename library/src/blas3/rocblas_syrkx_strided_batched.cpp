/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_syrkx.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syrkx_name[] = "unknown";
    template <>
    constexpr char rocblas_syrkx_name<float>[] = "rocblas_ssyrkx_strided_batched";
    template <>
    constexpr char rocblas_syrkx_name<double>[] = "rocblas_dsyrkx_strided_batched";
    template <>
    constexpr char rocblas_syrkx_name<rocblas_float_complex>[] = "rocblas_csyrkx_strided_batched";
    template <>
    constexpr char rocblas_syrkx_name<rocblas_double_complex>[] = "rocblas_zsyrkx_strided_batched";

    template <typename T>
    rocblas_status rocblas_syrkx_strided_batched_impl(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      const T*          beta,
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
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_syrkx_name<T>,
                              uplo,
                              transA,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              stride_a,
                              B,
                              ldb,
                              stride_b,
                              log_trace_scalar_value(beta),
                              C,
                              ldc,
                              stride_c,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syrkx_strided_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "-n",
                              n,
                              "-k",
                              k,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              stride_a,
                              "--ldb",
                              ldb,
                              "--stride_b",
                              stride_b,
                              LOG_BENCH_SCALAR_VALUE(beta),
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
                              rocblas_syrkx_name<T>,
                              uplo,
                              transA,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              stride_a,
                              B,
                              ldb,
                              stride_b,
                              log_trace_scalar_value(beta),
                              C,
                              ldc,
                              stride_c,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syrkx_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "N",
                            n,
                            "K",
                            k,
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

        static constexpr rocblas_int offset_C = 0, offset_A = 0, offset_B = 0;

        // syr2k arg check is equivalent
        rocblas_status arg_status = rocblas_syr2k_arg_check(handle,
                                                            uplo,
                                                            transA,
                                                            n,
                                                            k,
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

        static constexpr bool is2K = false; // syrkx
        return rocblas_syr2k_template<is2K>(handle,
                                            uplo,
                                            transA,
                                            n,
                                            k,
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
    }

}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                 \
    rocblas_status routine_name_(rocblas_handle    handle,      \
                                 rocblas_fill      uplo,        \
                                 rocblas_operation transA,      \
                                 rocblas_int       n,           \
                                 rocblas_int       k,           \
                                 const T_*         alpha,       \
                                 const T_*         A,           \
                                 rocblas_int       lda,         \
                                 rocblas_stride    stride_a,    \
                                 const T_*         B,           \
                                 rocblas_int       ldb,         \
                                 rocblas_stride    stride_b,    \
                                 const T_*         beta,        \
                                 T_*               C,           \
                                 rocblas_int       ldc,         \
                                 rocblas_stride    stride_c,    \
                                 rocblas_int       batch_count) \
    try                                                         \
    {                                                           \
        return rocblas_syrkx_strided_batched_impl(handle,       \
                                                  uplo,         \
                                                  transA,       \
                                                  n,            \
                                                  k,            \
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
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

IMPL(rocblas_ssyrkx_strided_batched, float);
IMPL(rocblas_dsyrkx_strided_batched, double);
IMPL(rocblas_csyrkx_strided_batched, rocblas_float_complex);
IMPL(rocblas_zsyrkx_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
