/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_trmm.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_trmm_name[] = "unknown";
    template <>
    constexpr char rocblas_trmm_name<float>[] = "rocblas_strided_batched_strmm";
    template <>
    constexpr char rocblas_trmm_name<double>[] = "rocblas_strided_batched_dtrmm";
    template <>
    constexpr char rocblas_trmm_name<rocblas_float_complex>[] = "rocblas_strided_batched_ctrmm";
    template <>
    constexpr char rocblas_trmm_name<rocblas_double_complex>[] = "rocblas_strided_batched_ztrmm";

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
                                     rocblas_stride    stride_a,
                                     T*                c,
                                     rocblas_int       ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

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
                              stride_a,
                              c,
                              ldc,
                              stride_c,
                              batch_count);

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
                              "--alpha",
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--stride_A",
                              stride_a,
                              "--ldb",
                              ldc,
                              "--stride_B",
                              stride_c,
                              "--batch_count",
                              batch_count);
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
                              alpha,
                              a,
                              lda,
                              stride_a,
                              c,
                              ldc,
                              stride_c,
                              batch_count);
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
                            "stride_A",
                            stride_a,
                            "ldb",
                            ldc,
                            "stride_B",
                            stride_c,
                            "batch_count",
                            batch_count);
            }
        }

        rocblas_int nrowa = rocblas_side_left == side ? m : n;

        if(m < 0 || n < 0 || lda < nrowa || ldc < m || batch_count < 0)
            return rocblas_status_invalid_size;

        if(m == 0 || n == 0 || batch_count == 0)
        {
            if(handle->is_device_memory_size_query())
                return rocblas_status_size_unchanged;
            else
                return rocblas_status_success;
        }

        if(!a || !c || !alpha)
            return rocblas_status_invalid_pointer;

        // gemm based trmm block sizes
        constexpr rocblas_int RB = 128;
        constexpr rocblas_int CB = 128;

        // work arrays dt1 and dt2 are used in trmm
        rocblas_int size_dt1 = RB * CB;
        rocblas_int size_dt2 = CB * CB;

        size_t dev_bytes = (size_dt1 + size_dt2) * batch_count * sizeof(T);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        rocblas_stride stride_mem = size_dt1 + size_dt2;

        return rocblas_trmm_template<RB, CB, T>(handle,
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

rocblas_status rocblas_strmm_strided_batched(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transa,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const float*      alpha,
                                             const float*      a,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_a,
                                             float*            c,
                                             rocblas_int       ldc,
                                             rocblas_stride    stride_c,
                                             rocblas_int       batch_count)
try
{
    return rocblas_trmm_impl(handle,
                             side,
                             uplo,
                             transa,
                             diag,
                             m,
                             n,
                             alpha,
                             a,
                             stride_a,
                             lda,
                             c,
                             ldc,
                             stride_c,
                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrmm_strided_batched(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transa,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const double*     alpha,
                                             const double*     a,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_a,
                                             double*           c,
                                             rocblas_int       ldc,
                                             rocblas_stride    stride_c,
                                             rocblas_int       batch_count)
try
{
    return rocblas_trmm_impl(handle,
                             side,
                             uplo,
                             transa,
                             diag,
                             m,
                             n,
                             alpha,
                             a,
                             stride_a,
                             lda,
                             c,
                             ldc,
                             stride_c,
                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrmm_strided_batched(rocblas_handle               handle,
                                             rocblas_side                 side,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            transa,
                                             rocblas_diagonal             diag,
                                             rocblas_int                  m,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* a,
                                             rocblas_int                  lda,
                                             rocblas_stride               stride_a,
                                             rocblas_float_complex*       c,
                                             rocblas_int                  ldc,
                                             rocblas_stride               stride_c,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_trmm_impl(handle,
                             side,
                             uplo,
                             transa,
                             diag,
                             m,
                             n,
                             alpha,
                             a,
                             stride_a,
                             lda,
                             c,
                             ldc,
                             stride_c,
                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrmm_strided_batched(rocblas_handle                handle,
                                             rocblas_side                  side,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             transa,
                                             rocblas_diagonal              diag,
                                             rocblas_int                   m,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* a,
                                             rocblas_int                   lda,
                                             rocblas_stride                stride_a,
                                             rocblas_double_complex*       c,
                                             rocblas_int                   ldc,
                                             rocblas_stride                stride_c,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_trmm_impl(handle,
                             side,
                             uplo,
                             transa,
                             diag,
                             m,
                             n,
                             alpha,
                             a,
                             stride_a,
                             lda,
                             c,
                             ldc,
                             stride_c,
                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"

/* ============================================================================================ */
