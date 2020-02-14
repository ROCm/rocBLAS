/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_syrk.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syrk_name[] = "unknown";
    template <>
    constexpr char rocblas_syrk_name<float>[] = "rocblas_ssyrk_strided_batched";
    template <>
    constexpr char rocblas_syrk_name<double>[] = "rocblas_dsyrk_strided_batched";
    template <>
    constexpr char rocblas_syrk_name<rocblas_float_complex>[] = "rocblas_csyrk_strided_batched";
    template <>
    constexpr char rocblas_syrk_name<rocblas_double_complex>[] = "rocblas_zsyrk_strided_batched";

    template <typename T, typename U>
    rocblas_status rocblas_syrk_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transA,
                                                     rocblas_int       n,
                                                     rocblas_int       k,
                                                     const U*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_stride    stride_a,
                                                     const U*          beta,
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
                              rocblas_syrk_name<T>,
                              uplo,
                              transA,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              stride_a,
                              log_trace_scalar_value(beta),
                              C,
                              ldc,
                              stride_c,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syrk_strided_batched -r",
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
                              rocblas_syrk_name<T>,
                              uplo,
                              transA,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              stride_a,
                              log_trace_scalar_value(beta),
                              C,
                              ldc,
                              stride_c,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syrk_name<T>,
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
                            "ldc",
                            ldc,
                            "stride_c",
                            stride_c,
                            "batch_count",
                            batch_count);
        }

        static constexpr rocblas_int offset_C = 0, offset_A = 0;

        rocblas_status arg_status = rocblas_syrk_arg_check(handle,
                                                           uplo,
                                                           transA,
                                                           n,
                                                           k,
                                                           alpha,
                                                           A,
                                                           offset_A,
                                                           lda,
                                                           stride_a,
                                                           beta,
                                                           C,
                                                           offset_C,
                                                           ldc,
                                                           stride_c,
                                                           batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        return rocblas_syrk_template(handle,
                                     uplo,
                                     transA,
                                     n,
                                     k,
                                     alpha,
                                     A,
                                     offset_A,
                                     lda,
                                     stride_a,
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
                                 const T_*         beta,        \
                                 T_*               C,           \
                                 rocblas_int       ldc,         \
                                 rocblas_stride    stride_c,    \
                                 rocblas_int       batch_count) \
    try                                                         \
    {                                                           \
        return rocblas_syrk_strided_batched_impl(handle,        \
                                                 uplo,          \
                                                 transA,        \
                                                 n,             \
                                                 k,             \
                                                 alpha,         \
                                                 A,             \
                                                 lda,           \
                                                 stride_a,      \
                                                 beta,          \
                                                 C,             \
                                                 ldc,           \
                                                 stride_c,      \
                                                 batch_count);  \
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

IMPL(rocblas_ssyrk_strided_batched, float);
IMPL(rocblas_dsyrk_strided_batched, double);
IMPL(rocblas_csyrk_strided_batched, rocblas_float_complex);
IMPL(rocblas_zsyrk_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
