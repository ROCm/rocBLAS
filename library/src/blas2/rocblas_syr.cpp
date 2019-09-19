/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_syr.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_syr_name[] = "unknown";
    template <>
    constexpr char rocblas_syr_name<float>[] = "rocblas_ssyr";
    template <>
    constexpr char rocblas_syr_name<double>[] = "rocblas_dsyr";

    template <typename T>
    rocblas_status rocblas_syr_impl(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    rocblas_int    n,
                                    const T*       alpha,
                                    const T*       x,
                                    rocblas_int    incx,
                                    T*             A,
                                    rocblas_int    lda)
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
                    log_trace(handle, rocblas_syr_name<T>, uplo, n, *alpha, x, incx, A, lda);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f syr -r",
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
                              lda);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle, rocblas_syr_name<T>, uplo, n, alpha, x, incx, A, lda);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syr_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "incx",
                            incx,
                            "lda",
                            lda);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!x || !A)
            return rocblas_status_invalid_pointer;
        if(n < 0 || !incx || lda < n || lda < 1)
            return rocblas_status_invalid_size;

        return rocblas_syr_template(handle, uplo, n, alpha, x, incx, A, lda);
    }

}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_ssyr(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const float*   alpha,
                            const float*   x,
                            rocblas_int    incx,
                            float*         A,
                            rocblas_int    lda)
{
    return rocblas_syr_impl(handle, uplo, n, alpha, x, incx, A, lda);
}

rocblas_status rocblas_dsyr(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const double*  alpha,
                            const double*  x,
                            rocblas_int    incx,
                            double*        A,
                            rocblas_int    lda)
{
    return rocblas_syr_impl(handle, uplo, n, alpha, x, incx, A, lda);
}

} // extern "C"
