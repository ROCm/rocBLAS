/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_trmm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_trmm_name[] = "unknown";
    template <>
    constexpr char rocblas_trmm_name<float>[] = "rocblas_strmm";
    template <>
    constexpr char rocblas_trmm_name<double>[] = "rocblas_dtrmm";
    template <>
    constexpr char rocblas_trmm_name<rocblas_float_complex>[] = "rocblas_ctrmm";
    template <>
    constexpr char rocblas_trmm_name<rocblas_double_complex>[] = "rocblas_ztrmm";

    template <typename T>
    rocblas_status rocblas_trmm_impl(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation transa,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     const T*          a,
                                     rocblas_int       lda,
                                     T*                c,
                                     rocblas_int       ldc)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        // gemm based trmm block sizes
        constexpr rocblas_int RB = 128;
        constexpr rocblas_int CB = 128;

        // work arrays dt1 and dt2 are used in trmm
        rocblas_int size_dt1 = RB * CB;
        rocblas_int size_dt2 = CB * CB;

        size_t dev_bytes = (size_dt1 + size_dt2) * sizeof(T);
        if(handle->is_device_memory_size_query())
        {
            if(m == 0 || n == 0)
                return rocblas_status_size_unchanged;

            return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto layer_mode = handle->layer_mode;
        if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile)
           && (!handle->is_device_memory_size_query()))
        {
            auto side_letter   = rocblas_side_letter(side);
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transa_letter = rocblas_transpose_letter(transa);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trmm_name<T>,
                              side,
                              uplo,
                              transa,
                              diag,
                              m,
                              n,
                              log_trace_scalar_value(alpha),
                              a,
                              lda,
                              c,
                              ldc);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f trmm -r",
                              rocblas_precision_string<T>,
                              "--side",
                              side_letter,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transa_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--ldb",
                              ldc);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trmm_name<T>,
                              side,
                              uplo,
                              transa,
                              diag,
                              m,
                              n,
                              log_trace_scalar_value(alpha),
                              a,
                              lda,
                              c,
                              ldc);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_trmm_name<T>,
                            "side",
                            side_letter,
                            "uplo",
                            uplo_letter,
                            "transa",
                            transa_letter,
                            "diag",
                            diag_letter,
                            "m",
                            m,
                            "n",
                            n,
                            "lda",
                            lda,
                            "ldb",
                            ldc);
            }
        }

        rocblas_int nrowa = rocblas_side_left == side ? m : n;

        if(m < 0 || n < 0 || lda < nrowa || ldc < m)
            return rocblas_status_invalid_size;

        if(m == 0 || n == 0)
            return rocblas_status_success;

        if(!a || !c || !alpha)
            return rocblas_status_invalid_pointer;

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        rocblas_stride stride_a    = 0;
        rocblas_stride stride_c    = 0;
        rocblas_stride stride_mem  = 0;
        rocblas_int    batch_count = 1;

        return rocblas_trmm_template<false, RB, CB, T>(handle,
                                                       side,
                                                       uplo,
                                                       transa,
                                                       diag,
                                                       m,
                                                       n,
                                                       alpha,
                                                       a,
                                                       lda,
                                                       stride_a,
                                                       c,
                                                       ldc,
                                                       stride_c,
                                                       batch_count,
                                                       (T*)mem,
                                                       stride_mem);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strmm(rocblas_handle    handle,
                             rocblas_side      side,
                             rocblas_fill      uplo,
                             rocblas_operation transa,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             rocblas_int       n,
                             const float*      alpha,
                             const float*      a,
                             rocblas_int       lda,
                             float*            c,
                             rocblas_int       ldc)
try
{
    return rocblas_trmm_impl(handle, side, uplo, transa, diag, m, n, alpha, a, lda, c, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrmm(rocblas_handle    handle,
                             rocblas_side      side,
                             rocblas_fill      uplo,
                             rocblas_operation transa,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             rocblas_int       n,
                             const double*     alpha,
                             const double*     a,
                             rocblas_int       lda,
                             double*           c,
                             rocblas_int       ldc)
try
{
    return rocblas_trmm_impl(handle, side, uplo, transa, diag, m, n, alpha, a, lda, c, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrmm(rocblas_handle               handle,
                             rocblas_side                 side,
                             rocblas_fill                 uplo,
                             rocblas_operation            transa,
                             rocblas_diagonal             diag,
                             rocblas_int                  m,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* a,
                             rocblas_int                  lda,
                             rocblas_float_complex*       c,
                             rocblas_int                  ldc)
try
{
    return rocblas_trmm_impl(handle, side, uplo, transa, diag, m, n, alpha, a, lda, c, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrmm(rocblas_handle                handle,
                             rocblas_side                  side,
                             rocblas_fill                  uplo,
                             rocblas_operation             transa,
                             rocblas_diagonal              diag,
                             rocblas_int                   m,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* a,
                             rocblas_int                   lda,
                             rocblas_double_complex*       c,
                             rocblas_int                   ldc)
try
{
    return rocblas_trmm_impl(handle, side, uplo, transa, diag, m, n, alpha, a, lda, c, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"

/* ============================================================================================ */
