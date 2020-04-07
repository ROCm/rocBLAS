/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_geam.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_geam_name[] = "unknown";
    template <>
    constexpr char rocblas_geam_name<float>[] = "rocblas_sgeam";
    template <>
    constexpr char rocblas_geam_name<double>[] = "rocblas_dgeam";
    template <>
    constexpr char rocblas_geam_name<rocblas_float_complex>[] = "rocblas_cgeam";
    template <>
    constexpr char rocblas_geam_name<rocblas_double_complex>[] = "rocblas_zgeam";

    template <typename T>
    rocblas_status rocblas_geam_impl(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     const T*          A,
                                     rocblas_int       lda,
                                     const T*          beta,
                                     const T*          B,
                                     rocblas_int       ldb,
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
            auto transA_letter = rocblas_transpose_letter(transA);
            auto transB_letter = rocblas_transpose_letter(transB);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_geam_name<T>,
                              transA,
                              transB,
                              m,
                              n,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              log_trace_scalar_value(beta),
                              B,
                              ldb,
                              C,
                              ldc);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f geam -r",
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
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--ldb",
                              ldb,
                              "--ldc",
                              ldc);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_geam_name<T>,
                              transA,
                              transB,
                              m,
                              n,
                              alpha,
                              A,
                              lda,
                              beta,
                              B,
                              ldb,
                              C,
                              ldc);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_geam_name<T>,
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
                            ldc);
        }

        if(m < 0 || n < 0 || ldc < m || lda < (transA == rocblas_operation_none ? m : n)
           || ldb < (transB == rocblas_operation_none ? m : n))
            return rocblas_status_invalid_size;

        if(!m || !n)
            return rocblas_status_success;

        if(!A || !B || !C)
            return rocblas_status_invalid_pointer;

        if((C == A && (lda != ldc || transA != rocblas_operation_none))
           || (C == B && (ldb != ldc || transB != rocblas_operation_none)))
            return rocblas_status_invalid_size;

        if(!alpha || !beta)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int    offset_A = 0, offset_B = 0, offset_C = 0, batch_count = 1;
        static constexpr rocblas_stride stride_A = 0, stride_B = 0, stride_C = 0;

        return rocblas_geam_template(handle,
                                     transA,
                                     transB,
                                     m,
                                     n,
                                     alpha,
                                     A,
                                     offset_A,
                                     lda,
                                     stride_A,
                                     beta,
                                     B,
                                     offset_B,
                                     ldb,
                                     stride_B,
                                     C,
                                     offset_C,
                                     ldc,
                                     stride_C,
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

#define IMPL(routine_name_, T_)                                                 \
    rocblas_status routine_name_(rocblas_handle    handle,                      \
                                 rocblas_operation transA,                      \
                                 rocblas_operation transB,                      \
                                 rocblas_int       m,                           \
                                 rocblas_int       n,                           \
                                 const T_*         alpha,                       \
                                 const T_*         A,                           \
                                 rocblas_int       lda,                         \
                                 const T_*         beta,                        \
                                 const T_*         B,                           \
                                 rocblas_int       ldb,                         \
                                 T_*               C,                           \
                                 rocblas_int       ldc)                         \
    try                                                                         \
    {                                                                           \
        return rocblas_geam_impl<T_>(                                           \
            handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc); \
    }                                                                           \
    catch(...)                                                                  \
    {                                                                           \
        return exception_to_rocblas_status();                                   \
    }

IMPL(rocblas_sgeam, float);
IMPL(rocblas_dgeam, double);
IMPL(rocblas_cgeam, rocblas_float_complex);
IMPL(rocblas_zgeam, rocblas_double_complex);

#undef IMPL

} // extern "C"
