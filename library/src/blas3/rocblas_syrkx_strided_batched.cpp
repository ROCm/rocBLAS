/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.hpp"
#include "rocblas_syrkx.hpp"
#include "utility.hpp"

#define SSYRKX_MIN_NB 16
#define DSYRKX_MIN_NB 16
#define CSYRKX_MIN_NB 8
#define ZSYRKX_MIN_NB 8

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

    template <int MIN_NB, typename T>
    rocblas_status rocblas_syrkx_strided_batched_impl(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation trans,
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
            auto uplo_letter  = rocblas_fill_letter(uplo);
            auto trans_letter = rocblas_transpose_letter(trans);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_syrkx_name<T>,
                          uplo,
                          trans,
                          n,
                          k,
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
                          "./rocblas-bench -f syrkx_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--uplo",
                          uplo_letter,
                          "--transposeA",
                          trans_letter,
                          "-n",
                          n,
                          "-k",
                          k,
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
                            rocblas_syrkx_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            trans_letter,
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

        static constexpr rocblas_int offset_c = 0, offset_a = 0, offset_b = 0;

        // syr2k arg check is equivalent
        rocblas_status arg_status = rocblas_syr2k_arg_check(handle,
                                                            uplo,
                                                            trans,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            offset_a,
                                                            lda,
                                                            stride_a,
                                                            B,
                                                            offset_b,
                                                            ldb,
                                                            stride_b,
                                                            beta,
                                                            C,
                                                            offset_c,
                                                            ldc,
                                                            stride_c,
                                                            batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        static constexpr bool BATCHED = false;

        rocblas_int n2          = rocblas_operation_none == trans ? k : n;
        bool        i64_indices = (n2 * size_t(lda) > std::numeric_limits<rocblas_int>::max())
                           || (n2 * size_t(ldb) > std::numeric_limits<rocblas_int>::max())
                           || (n * size_t(ldc) > std::numeric_limits<rocblas_int>::max());

        if(i64_indices)
        {
            return rocblas_internal_syrkx_template<MIN_NB, BATCHED, T>(handle,
                                                                       uplo,
                                                                       trans,
                                                                       n,
                                                                       k,
                                                                       alpha,
                                                                       A,
                                                                       size_t(offset_a),
                                                                       size_t(lda),
                                                                       stride_a,
                                                                       B,
                                                                       size_t(offset_b),
                                                                       size_t(ldb),
                                                                       stride_b,
                                                                       beta,
                                                                       C,
                                                                       size_t(offset_a),
                                                                       size_t(ldc),
                                                                       stride_c,
                                                                       batch_count);
        }
        else
        {
            return rocblas_internal_syrkx_template<MIN_NB, BATCHED, T>(handle,
                                                                       uplo,
                                                                       trans,
                                                                       n,
                                                                       k,
                                                                       alpha,
                                                                       A,
                                                                       offset_a,
                                                                       lda,
                                                                       stride_a,
                                                                       B,
                                                                       offset_b,
                                                                       ldb,
                                                                       stride_b,
                                                                       beta,
                                                                       C,
                                                                       offset_c,
                                                                       ldc,
                                                                       stride_c,
                                                                       batch_count);
        }

        return rocblas_internal_syrkx_template<MIN_NB, BATCHED, T>(handle,
                                                                   uplo,
                                                                   trans,
                                                                   n,
                                                                   k,
                                                                   alpha,
                                                                   A,
                                                                   offset_a,
                                                                   lda,
                                                                   stride_a,
                                                                   B,
                                                                   offset_b,
                                                                   ldb,
                                                                   stride_b,
                                                                   beta,
                                                                   C,
                                                                   offset_c,
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

#define IMPL(routine_name_, T_, MIN_NB)                                 \
    rocblas_status routine_name_(rocblas_handle    handle,              \
                                 rocblas_fill      uplo,                \
                                 rocblas_operation trans,               \
                                 rocblas_int       n,                   \
                                 rocblas_int       k,                   \
                                 const T_*         alpha,               \
                                 const T_*         A,                   \
                                 rocblas_int       lda,                 \
                                 rocblas_stride    stride_a,            \
                                 const T_*         B,                   \
                                 rocblas_int       ldb,                 \
                                 rocblas_stride    stride_b,            \
                                 const T_*         beta,                \
                                 T_*               C,                   \
                                 rocblas_int       ldc,                 \
                                 rocblas_stride    stride_c,            \
                                 rocblas_int       batch_count)         \
    try                                                                 \
    {                                                                   \
        return rocblas_syrkx_strided_batched_impl<MIN_NB>(handle,       \
                                                          uplo,         \
                                                          trans,        \
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
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

IMPL(rocblas_ssyrkx_strided_batched, float, SSYRKX_MIN_NB);
IMPL(rocblas_dsyrkx_strided_batched, double, DSYRKX_MIN_NB);
IMPL(rocblas_csyrkx_strided_batched, rocblas_float_complex, CSYRKX_MIN_NB);
IMPL(rocblas_zsyrkx_strided_batched, rocblas_double_complex, ZSYRKX_MIN_NB);

#undef IMPL
#undef SSYRKX_MIN_NB
#undef DSYRKX_MIN_NB
#undef CSYRKX_MIN_NB
#undef ZSYRKX_MIN_NB

} // extern "C"
