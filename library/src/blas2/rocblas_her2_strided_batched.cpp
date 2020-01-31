/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_her2.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_her2_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_her2_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_cher2_strided_batched";
    template <>
    constexpr char rocblas_her2_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zher2_strided_batched";

    template <typename T>
    rocblas_status rocblas_her2_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     const T*       y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stridey,
                                                     T*             A,
                                                     rocblas_int    lda,
                                                     rocblas_stride strideA,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_her2_strided_batched_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              stridex,
                              y,
                              incy,
                              stridey,
                              A,
                              lda,
                              strideA,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f her2_strided_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--incx",
                              incx,
                              "--stride_x",
                              stridex,
                              "--incy",
                              incy,
                              "--stride_y",
                              stridey,
                              "--lda",
                              lda,
                              "--stride_a",
                              strideA,
                              "--batch_count",
                              batch_count);
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_her2_strided_batched_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              x,
                              incx,
                              stridex,
                              y,
                              incy,
                              stridey,
                              A,
                              lda,
                              strideA,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_her2_strided_batched_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "incx",
                            incx,
                            "stride_x",
                            stridex,
                            "incy",
                            incy,
                            "stride_y",
                            stridey,
                            "lda",
                            lda,
                            "stride_a",
                            strideA,
                            "batch_count",
                            batch_count);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;
        if(n < 0 || !incx || !incy || batch_count < 0 || lda < n || lda < 1)
            return rocblas_status_invalid_size;
        if(!n || !batch_count)
            return rocblas_status_success;
        if(!x || !y || !A || !alpha)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int offset_x = 0, offset_y = 0, offset_A = 0;
        return rocblas_her2_template(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     x,
                                     offset_x,
                                     incx,
                                     stridex,
                                     y,
                                     offset_y,
                                     incy,
                                     stridey,
                                     A,
                                     lda,
                                     offset_A,
                                     strideA,
                                     batch_count);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_cher2_strided_batched(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             const rocblas_float_complex* y,
                                             rocblas_int                  incy,
                                             rocblas_stride               stridey,
                                             rocblas_float_complex*       A,
                                             rocblas_int                  lda,
                                             rocblas_stride               strideA,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_her2_strided_batched_impl(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zher2_strided_batched(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             const rocblas_double_complex* y,
                                             rocblas_int                   incy,
                                             rocblas_stride                stridey,
                                             rocblas_double_complex*       A,
                                             rocblas_int                   lda,
                                             rocblas_stride                strideA,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_her2_strided_batched_impl(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
