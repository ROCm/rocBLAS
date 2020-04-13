/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_tpsv.hpp"
#include "utility.h"

namespace
{
    constexpr rocblas_int STPSV_BLOCK = 512;
    constexpr rocblas_int DTPSV_BLOCK = 512;

    template <typename>
    constexpr char rocblas_tpsv_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_tpsv_batched_name<float>[] = "rocblas_stpsv_batched";
    template <>
    constexpr char rocblas_tpsv_batched_name<double>[] = "rocblas_dtpsv_batched";
    template <>
    constexpr char rocblas_tpsv_batched_name<rocblas_float_complex>[] = "rocblas_ctpsv_batched";
    template <>
    constexpr char rocblas_tpsv_batched_name<rocblas_double_complex>[] = "rocblas_ztpsv_batched";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_tpsv_batched_impl(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       n,
                                             const T* const    AP[],
                                             T* const          x[],
                                             rocblas_int       incx,
                                             rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_tpsv_batched_name<T>,
                      uplo,
                      transA,
                      diag,
                      n,
                      AP,
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
                              "./rocblas-bench -f tpsv_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-n",
                              n,
                              "--incx",
                              incx,
                              "--batch_count",
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_tpsv_batched_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "N",
                            n,
                            "incx",
                            incx,
                            "batch_count",
                            batch_count);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;
        if(n < 0 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;
        if(!n || !batch_count)
            return rocblas_status_success;
        if(!AP || !x)
            return rocblas_status_invalid_pointer;

        rocblas_status status = rocblas_tpsv_template<BLOCK>(
            handle, uplo, transA, diag, n, AP, 0, 0, x, 0, incx, 0, batch_count);

        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_stpsv_batched(rocblas_handle     handle,
                                     rocblas_fill       uplo,
                                     rocblas_operation  transA,
                                     rocblas_diagonal   diag,
                                     rocblas_int        n,
                                     const float* const AP[],
                                     float* const       x[],
                                     rocblas_int        incx,
                                     rocblas_int        batch_count)
try
{
    return rocblas_tpsv_batched_impl<STPSV_BLOCK>(
        handle, uplo, transA, diag, n, AP, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtpsv_batched(rocblas_handle      handle,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         n,
                                     const double* const AP[],
                                     double* const       x[],
                                     rocblas_int         incx,
                                     rocblas_int         batch_count)
try
{
    return rocblas_tpsv_batched_impl<DTPSV_BLOCK>(
        handle, uplo, transA, diag, n, AP, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctpsv_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_operation                  transA,
                                     rocblas_diagonal                   diag,
                                     rocblas_int                        n,
                                     const rocblas_float_complex* const AP[],
                                     rocblas_float_complex* const       x[],
                                     rocblas_int                        incx,

                                     rocblas_int batch_count)
try
{
    return rocblas_tpsv_batched_impl<STPSV_BLOCK>(
        handle, uplo, transA, diag, n, AP, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztpsv_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_operation                   transA,
                                     rocblas_diagonal                    diag,
                                     rocblas_int                         n,
                                     const rocblas_double_complex* const AP[],
                                     rocblas_double_complex* const       x[],
                                     rocblas_int                         incx,
                                     rocblas_int                         batch_count)
try
{
    return rocblas_tpsv_batched_impl<DTPSV_BLOCK>(
        handle, uplo, transA, diag, n, AP, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
