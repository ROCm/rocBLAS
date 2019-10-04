/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_ger.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_ger_name[] = "unknown";
    template <>
    constexpr char rocblas_ger_name<float>[] = "rocblas_sger";
    template <>
    constexpr char rocblas_ger_name<double>[] = "rocblas_dger";

    template <typename T>
    rocblas_status rocblas_ger_impl(rocblas_handle handle,
                                    rocblas_int    m,
                                    rocblas_int    n,
                                    const T*       alpha,
                                    const T*       x,
                                    rocblas_int    incx,
                                    const T*       y,
                                    rocblas_int    incy,
                                    T*             A,
                                    rocblas_int    lda)
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
                log_trace(handle, rocblas_ger_name<T>, m, n, *alpha, x, incx, y, incy, A, lda);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f ger -r",
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
                          lda);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_ger_name<T>, m, n, alpha, x, incx, y, incy, A, lda);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_ger_name<T>,
                        "M",
                        m,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "lda",
                        lda);

        if(!x || !y || !A)
            return rocblas_status_invalid_pointer;

        if(m < 0 || n < 0 || !incx || !incy || lda < m || lda < 1)
            return rocblas_status_invalid_size;

        rocblas_ger_template<T>(
            handle, m, n, alpha, 0, x, 0, incx, 0, y, 0, incy, 0, A, 0, lda, 0, 1);

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sger(rocblas_handle handle,
                            rocblas_int    m,
                            rocblas_int    n,
                            const float*   alpha,
                            const float*   x,
                            rocblas_int    incx,
                            const float*   y,
                            rocblas_int    incy,
                            float*         A,
                            rocblas_int    lda)
{
    return rocblas_ger_impl(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

rocblas_status rocblas_dger(rocblas_handle handle,
                            rocblas_int    m,
                            rocblas_int    n,
                            const double*  alpha,
                            const double*  x,
                            rocblas_int    incx,
                            const double*  y,
                            rocblas_int    incy,
                            double*        A,
                            rocblas_int    lda)
{
    return rocblas_ger_impl(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

} // extern "C"
