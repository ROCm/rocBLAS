/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas/rocblas.h"
#include "rocblas_gemv.hpp"
#include "rocblas_tbsv.hpp"
#include "utility.hpp"

namespace
{
    constexpr rocblas_int STBSV_BLOCK = 512;
    constexpr rocblas_int DTBSV_BLOCK = 512;

    template <typename>
    constexpr char rocblas_tbsv_name[] = "unknown";
    template <>
    constexpr char rocblas_tbsv_name<float>[] = "rocblas_stbsv_batched";
    template <>
    constexpr char rocblas_tbsv_name<double>[] = "rocblas_dtbsv_batched";
    template <>
    constexpr char rocblas_tbsv_name<rocblas_float_complex>[] = "rocblas_ctbsv_batched";
    template <>
    constexpr char rocblas_tbsv_name<rocblas_double_complex>[] = "rocblas_ztbsv_batched";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_tbsv_batched_impl(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             T* const          x[],
                                             rocblas_int       incx,
                                             rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_tbsv_name<T>,
                      uplo,
                      transA,
                      diag,
                      n,
                      k,
                      A,
                      lda,
                      x,
                      incx,
                      batch_count);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                    log_bench(handle,
                              "./rocblas-bench -f tbsv_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-n",
                              n,
                              "-k",
                              k,
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              "--batch_count",
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_tbsv_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "N",
                            n,
                            "K",
                            k,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "batch_count",
                            batch_count);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;
        if(n < 0 || k < 0 || lda < k + 1 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;
        if(!n || !batch_count)
            return rocblas_status_success;
        if(!A || !x)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tbsv_check_numerics_status
                = rocblas_tbsv_check_numerics(rocblas_tbsv_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tbsv_check_numerics_status != rocblas_status_success)
                return tbsv_check_numerics_status;
        }

        rocblas_status status = rocblas_tbsv_template<BLOCK>(
            handle, uplo, transA, diag, n, k, A, 0, lda, 0, x, 0, incx, 0, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tbsv_check_numerics_status
                = rocblas_tbsv_check_numerics(rocblas_tbsv_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tbsv_check_numerics_status != rocblas_status_success)
                return tbsv_check_numerics_status;
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

rocblas_status rocblas_stbsv_batched(rocblas_handle     handle,
                                     rocblas_fill       uplo,
                                     rocblas_operation  transA,
                                     rocblas_diagonal   diag,
                                     rocblas_int        n,
                                     rocblas_int        k,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     float* const       x[],
                                     rocblas_int        incx,
                                     rocblas_int        batch_count)
try
{
    return rocblas_tbsv_batched_impl<STBSV_BLOCK>(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtbsv_batched(rocblas_handle      handle,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         n,
                                     rocblas_int         k,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     double* const       x[],
                                     rocblas_int         incx,
                                     rocblas_int         batch_count)
try
{
    return rocblas_tbsv_batched_impl<DTBSV_BLOCK>(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctbsv_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_operation                  transA,
                                     rocblas_diagonal                   diag,
                                     rocblas_int                        n,
                                     rocblas_int                        k,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        lda,
                                     rocblas_float_complex* const       x[],
                                     rocblas_int                        incx,
                                     rocblas_int                        batch_count)
try
{
    return rocblas_tbsv_batched_impl<STBSV_BLOCK>(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztbsv_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_operation                   transA,
                                     rocblas_diagonal                    diag,
                                     rocblas_int                         n,
                                     rocblas_int                         k,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         lda,
                                     rocblas_double_complex* const       x[],
                                     rocblas_int                         incx,
                                     rocblas_int                         batch_count)
try
{
    return rocblas_tbsv_batched_impl<DTBSV_BLOCK>(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
