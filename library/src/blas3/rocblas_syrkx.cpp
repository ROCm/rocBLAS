/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_syrkx.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syrkx_name[] = "unknown";
    template <>
    constexpr char rocblas_syrkx_name<float>[] = "rocblas_ssyrkx";
    template <>
    constexpr char rocblas_syrkx_name<double>[] = "rocblas_dsyrkx";
    template <>
    constexpr char rocblas_syrkx_name<rocblas_float_complex>[] = "rocblas_csyrkx";
    template <>
    constexpr char rocblas_syrkx_name<rocblas_double_complex>[] = "rocblas_zsyrkx";

    template <typename T>
    rocblas_status rocblas_syrkx_impl(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          B,
                                      rocblas_int       ldb,
                                      const T*          beta,
                                      T*                C,
                                      rocblas_int       ldc)
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
                              B,
                              ldb,
                              log_trace_scalar_value(beta),
                              C,
                              ldc);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syrkx -r",
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
                              "--ldb",
                              ldb,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--ldc",
                              ldc);
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
                              B,
                              ldb,
                              log_trace_scalar_value(beta),
                              C,
                              ldc);
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
                            "ldb",
                            ldb,
                            "ldc",
                            ldc);
        }

        static constexpr rocblas_int    offset_C = 0, offset_A = 0, offset_B = 0, batch_count = 1;
        static constexpr rocblas_stride stride_C = 0, stride_A = 0, stride_B = 0;

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
                                                            stride_A,
                                                            B,
                                                            offset_B,
                                                            ldb,
                                                            stride_B,
                                                            beta,
                                                            C,
                                                            offset_C,
                                                            ldc,
                                                            stride_C,
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
                                            stride_A,
                                            B,
                                            offset_B,
                                            ldb,
                                            stride_B,
                                            beta,
                                            C,
                                            offset_C,
                                            ldc,
                                            stride_C,
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

#define IMPL(routine_name_, T_)                                               \
    rocblas_status routine_name_(rocblas_handle    handle,                    \
                                 rocblas_fill      uplo,                      \
                                 rocblas_operation transA,                    \
                                 rocblas_int       n,                         \
                                 rocblas_int       k,                         \
                                 const T_*         alpha,                     \
                                 const T_*         A,                         \
                                 rocblas_int       lda,                       \
                                 const T_*         B,                         \
                                 rocblas_int       ldb,                       \
                                 const T_*         beta,                      \
                                 T_*               C,                         \
                                 rocblas_int       ldc)                       \
    try                                                                       \
    {                                                                         \
        return rocblas_syrkx_impl(                                            \
            handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        return exception_to_rocblas_status();                                 \
    }

IMPL(rocblas_ssyrkx, float);
IMPL(rocblas_dsyrkx, double);
IMPL(rocblas_csyrkx, rocblas_float_complex);
IMPL(rocblas_zsyrkx, rocblas_double_complex);

#undef IMPL

} // extern "C"
