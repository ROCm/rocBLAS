/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "logging.hpp"
#include "rocblas_her2.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_her2_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_her2_batched_name<rocblas_float_complex>[] = "rocblas_cher2_batched";
    template <>
    constexpr char rocblas_her2_batched_name<rocblas_double_complex>[] = "rocblas_zher2_batched";

    template <typename T>
    rocblas_status rocblas_her2_batched_impl(rocblas_handle handle,
                                             rocblas_fill   uplo,
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

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_her2_batched_name<T>,
                          uplo,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          0,
                          x,
                          incx,
                          y,
                          incy,
                          A,
                          lda);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f her2_batched -r",
                          rocblas_precision_string<T>,
                          "--uplo",
                          uplo_letter,
                          "-n",
                          n,
                          LOG_BENCH_SCALAR_VALUE(handle, alpha),
                          "--incx",
                          incx,
                          "--incy",
                          incy,
                          "--lda",
                          lda,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_her2_batched_name<T>,
                            "uplo",
                            uplo_letter,
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
        }

        static constexpr rocblas_stride offset_x = 0, offset_y = 0, offset_A = 0;
        static constexpr rocblas_stride stride_x = 0, stride_y = 0, stride_A = 0;

        rocblas_status arg_status = rocblas_her2_arg_check(handle,
                                                           uplo,
                                                           n,
                                                           alpha,
                                                           x,
                                                           offset_x,
                                                           incx,
                                                           stride_x,
                                                           y,
                                                           offset_y,
                                                           incy,
                                                           stride_y,
                                                           A,
                                                           lda,
                                                           offset_A,
                                                           stride_A,
                                                           batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status her2_check_numerics_status
                = rocblas_her2_check_numerics(rocblas_her2_batched_name<T>,
                                              handle,
                                              uplo,
                                              n,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(her2_check_numerics_status != rocblas_status_success)
                return her2_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_her2_batched_template(handle,
                                                                       uplo,
                                                                       n,
                                                                       alpha,
                                                                       x,
                                                                       offset_x,
                                                                       incx,
                                                                       stride_x,
                                                                       y,
                                                                       offset_y,
                                                                       incy,
                                                                       stride_y,
                                                                       A,
                                                                       lda,
                                                                       offset_A,
                                                                       stride_A,
                                                                       batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status her2_check_numerics_status
                = rocblas_her2_check_numerics(rocblas_her2_batched_name<T>,
                                              handle,
                                              uplo,
                                              n,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(her2_check_numerics_status != rocblas_status_success)
                return her2_check_numerics_status;
        }
        return status;
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_cher2_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_int                        n,
                                     const rocblas_float_complex*       alpha,
                                     const rocblas_float_complex* const x[],
                                     rocblas_int                        incx,
                                     const rocblas_float_complex* const y[],
                                     rocblas_int                        incy,
                                     rocblas_float_complex* const       A[],
                                     rocblas_int                        lda,
                                     rocblas_int                        batch_count)
try
{
    return rocblas_her2_batched_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zher2_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_int                         n,
                                     const rocblas_double_complex*       alpha,
                                     const rocblas_double_complex* const x[],
                                     rocblas_int                         incx,
                                     const rocblas_double_complex* const y[],
                                     rocblas_int                         incy,
                                     rocblas_double_complex* const       A[],
                                     rocblas_int                         lda,
                                     rocblas_int                         batch_count)
try
{
    return rocblas_her2_batched_impl(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
