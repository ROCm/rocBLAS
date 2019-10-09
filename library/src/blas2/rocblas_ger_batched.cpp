/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_ger.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_ger_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_ger_batched_name<float>[] = "rocblas_sger_batched";
    template <>
    constexpr char rocblas_ger_batched_name<double>[] = "rocblas_dger_batched";

    template <typename T>
    rocblas_status rocblas_ger_batched_impl(rocblas_handle handle,
                                            rocblas_int    m,
                                            rocblas_int    n,
                                            const T*       alpha,
                                            const T* const x[],
                                            rocblas_int    incx,
                                            const T* const y[],
                                            rocblas_int    incy,
                                            T* const       A[],
                                            rocblas_int    lda,
                                            rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        if(!alpha)
            return rocblas_status_invalid_pointer;

        auto layer_mode = handle->layer_mode;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_ger_batched_name<T>,
                          m,
                          n,
                          *alpha,
                          x,
                          incx,
                          y,
                          incy,
                          A,
                          lda,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f ger_batched -r",
                          rocblas_precision_string<T>,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--alpha",
                          *alpha,
                          "--incx",
                          incx,
                          "--incy",
                          incy,
                          "--lda",
                          lda,
                          "--batch",
                          batch_count);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_ger_batched_name<T>,
                          m,
                          n,
                          alpha,
                          x,
                          incx,
                          y,
                          incy,
                          A,
                          lda,
                          batch_count);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_ger_batched_name<T>,
                        "M",
                        m,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "lda",
                        lda,
                        "batch_count",
                        batch_count);

        if(!x || !y || !A)
            return rocblas_status_invalid_pointer;

        if(m < 0 || n < 0 || !incx || !incy || lda < m || lda < 1 || batch_count < 0)
            return rocblas_status_invalid_size;

        rocblas_ger_template<T>(
            handle, m, n, alpha, 0, x, 0, incx, 0, y, 0, incy, 0, A, 0, lda, 0, batch_count);

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sger_batched(rocblas_handle     handle,
                                    rocblas_int        m,
                                    rocblas_int        n,
                                    const float*       alpha,
                                    const float* const x[],
                                    rocblas_int        incx,
                                    const float* const y[],
                                    rocblas_int        incy,
                                    float* const       A[],
                                    rocblas_int        lda,
                                    rocblas_int        batch_count)
{
    return rocblas_ger_batched_impl(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

rocblas_status rocblas_dger_batched(rocblas_handle      handle,
                                    rocblas_int         m,
                                    rocblas_int         n,
                                    const double*       alpha,
                                    const double* const x[],
                                    rocblas_int         incx,
                                    const double* const y[],
                                    rocblas_int         incy,
                                    double* const       A[],
                                    rocblas_int         lda,
                                    rocblas_int         batch_count)
{
    return rocblas_ger_batched_impl(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

} // extern "C"
