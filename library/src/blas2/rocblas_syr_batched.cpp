/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_syr_batched.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syr_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_syr_batched_name<float>[] = "rocblas_ssyr_batched";
    template <>
    constexpr char rocblas_syr_batched_name<double>[] = "rocblas_dsyr_batched";

    template <typename T>
    rocblas_status rocblas_syr_batched_impl(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const T*       alpha,
                                            const T* const x[],
                                            rocblas_int    shiftx,
                                            rocblas_int    incx,
                                            T* const       A[],
                                            rocblas_int    shiftA,
                                            rocblas_int    lda,
                                            rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        if(!alpha)
            return rocblas_status_invalid_pointer;
        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(
                        handle, rocblas_syr_batched_name<T>, uplo, n, *alpha, 0, x, incx, A, lda);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syr_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              "--alpha",
                              *alpha,
                              "--incx",
                              incx,
                              "--lda",
                              lda,
                              "--batch",
                              batch_count);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_syr_batched_name<T>,
                              uplo,
                              n,
                              alpha,
                              x,
                              incx,
                              A,
                              lda,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syr_batched_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "incx",
                            incx,
                            "lda",
                            lda,
                            "batch",
                            batch_count);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!x || !A)
            return rocblas_status_invalid_pointer;
        if(n < 0 || !incx || lda < n || lda < 1 || batch_count < 0)
            return rocblas_status_invalid_size;

        return rocblas_syr_batched_template(
            handle, uplo, n, alpha, 0, x, 0, incx, A, 0, lda, batch_count);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_ssyr_batched(rocblas_handle     handle,
                                    rocblas_fill       uplo,
                                    rocblas_int        n,
                                    const float*       alpha,
                                    const float* const x[],
                                    rocblas_int        incx,
                                    float* const       A[],
                                    rocblas_int        lda,
                                    rocblas_int        batch_count)
{
    return rocblas_syr_batched_impl(handle, uplo, n, alpha, x, 0, incx, A, 0, lda, batch_count);
}

rocblas_status rocblas_dsyr_batched(rocblas_handle      handle,
                                    rocblas_fill        uplo,
                                    rocblas_int         n,
                                    const double*       alpha,
                                    const double* const x[],
                                    rocblas_int         incx,
                                    double* const       A[],
                                    rocblas_int         lda,
                                    rocblas_int         batch_count)
{
    return rocblas_syr_batched_impl(handle, uplo, n, alpha, x, 0, incx, A, 0, lda, batch_count);
}

} // extern "C"
