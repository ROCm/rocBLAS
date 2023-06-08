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
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_gemv.hpp"
#include "utility.hpp"

namespace
{
    template <typename, typename>
    constexpr char rocblas_gemv_name[] = "unknown";
    template <>
    constexpr char rocblas_gemv_name<float, float>[] = "rocblas_sgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<double, double>[] = "rocblas_dgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_float_complex, rocblas_float_complex>[]
        = "rocblas_cgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_double_complex, rocblas_double_complex>[]
        = "rocblas_zgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_half, rocblas_half>[]
        = "rocblas_hshgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_half, float>[] = "rocblas_hssgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_bfloat16, rocblas_bfloat16>[]
        = "rocblas_tstgemv_strided_batched";
    template <>
    constexpr char rocblas_gemv_name<rocblas_bfloat16, float>[] = "rocblas_tssgemv_strided_batched";

    template <typename Ti, typename Tex = Ti, typename To = Ti>
    rocblas_status rocblas_gemv_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_operation transA,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     const Tex*        alpha,
                                                     const Ti*         A,
                                                     rocblas_int       lda,
                                                     rocblas_stride    strideA,
                                                     const Ti*         x,
                                                     rocblas_int       incx,
                                                     rocblas_stride    stridex,
                                                     const Tex*        beta,
                                                     To*               y,
                                                     rocblas_int       incy,
                                                     rocblas_stride    stridey,
                                                     rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes
            = rocblas_internal_gemv_kernel_workspace_size<Tex>(transA, m, n, batch_count);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;

        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto transA_letter = rocblas_transpose_letter(transA);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemv_name<Ti, To>,
                          transA,
                          m,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          A,
                          lda,
                          strideA,
                          x,
                          incx,
                          stridex,
                          LOG_TRACE_SCALAR_VALUE(handle, beta),
                          y,
                          incy,
                          stridey,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if constexpr(std::is_same<Ti, rocblas_half>{}
                             || std::is_same<Ti, rocblas_bfloat16>{})
                {
                    log_bench(handle,
                              "./rocblas-bench -f gemv_strided_batched --a_type",
                              rocblas_precision_string<Ti>,
                              "--c_type",
                              rocblas_precision_string<To>,
                              "--compute_type",
                              rocblas_precision_string<Tex>,
                              "--transposeA",
                              transA_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(handle, alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              strideA,
                              "--incx",
                              incx,
                              "--stride_x",
                              stridex,
                              LOG_BENCH_SCALAR_VALUE(handle, beta),
                              "--incy",
                              incy,
                              "--stride_y",
                              stridey,
                              "--batch_count",
                              batch_count);
                }
                else
                {
                    log_bench(handle,
                              "./rocblas-bench -f gemv_strided_batched -r",
                              rocblas_precision_string<Ti>,
                              "--transposeA",
                              transA_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(handle, alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              strideA,
                              "--incx",
                              incx,
                              "--stride_x",
                              stridex,
                              LOG_BENCH_SCALAR_VALUE(handle, beta),
                              "--incy",
                              incy,
                              "--stride_y",
                              stridey,
                              "--batch_count",
                              batch_count);
                }
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_gemv_name<Ti, To>,
                            "transA",
                            transA_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "stride_a",
                            strideA,
                            "incx",
                            incx,
                            "stride_x",
                            stridex,
                            "incy",
                            incy,
                            "stride_y",
                            stridey,
                            "batch_count",
                            batch_count);
        }

        rocblas_status arg_status = rocblas_internal_gemv_arg_check(handle,
                                                                    transA,
                                                                    m,
                                                                    n,
                                                                    alpha,
                                                                    0,
                                                                    A,
                                                                    0,
                                                                    lda,
                                                                    strideA,
                                                                    x,
                                                                    0,
                                                                    incx,
                                                                    stridex,
                                                                    beta,
                                                                    0,
                                                                    y,
                                                                    0,
                                                                    incy,
                                                                    stridey,
                                                                    batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        rocblas_status perf_status = rocblas_status_success;
        auto           w_mem       = handle->device_malloc(dev_bytes);
        if(!w_mem)
            perf_status = rocblas_status_perf_degraded;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status gemv_check_numerics_status
                = rocblas_gemv_check_numerics(rocblas_gemv_name<Ti, To>,
                                              handle,
                                              transA,
                                              m,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              strideA,
                                              x,
                                              0,
                                              incx,
                                              stridex,
                                              y,
                                              0,
                                              incy,
                                              stridey,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(gemv_check_numerics_status != rocblas_status_success)
                return gemv_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_gemv_template(handle,
                                                               transA,
                                                               m,
                                                               n,
                                                               alpha,
                                                               0,
                                                               A,
                                                               0,
                                                               lda,
                                                               strideA,
                                                               x,
                                                               0,
                                                               incx,
                                                               stridex,
                                                               beta,
                                                               0,
                                                               y,
                                                               0,
                                                               incy,
                                                               stridey,
                                                               batch_count,
                                                               (Tex*)w_mem);

        status = (status != rocblas_status_success) ? status : perf_status;
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status gemv_check_numerics_status
                = rocblas_gemv_check_numerics(rocblas_gemv_name<Ti, To>,
                                              handle,
                                              transA,
                                              m,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              strideA,
                                              x,
                                              0,
                                              incx,
                                              stridex,
                                              y,
                                              0,
                                              incy,
                                              stridey,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(gemv_check_numerics_status != rocblas_status_success)
                return gemv_check_numerics_status;
        }
        return status;
    }
} //namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_sgemv_strided_batched(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const float*      alpha,
                                             const float*      A,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             const float*      x,
                                             rocblas_int       incx,
                                             rocblas_stride    stridex,
                                             const float*      beta,
                                             float*            y,
                                             rocblas_int       incy,
                                             rocblas_stride    stridey,
                                             rocblas_int       batch_count)
try
{
    return rocblas_gemv_strided_batched_impl(handle,
                                             transA,
                                             m,
                                             n,
                                             alpha,
                                             A,
                                             lda,
                                             strideA,
                                             x,
                                             incx,
                                             stridex,
                                             beta,
                                             y,
                                             incy,
                                             stridey,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dgemv_strided_batched(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const double*     alpha,
                                             const double*     A,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             const double*     x,
                                             rocblas_int       incx,
                                             rocblas_stride    stridex,
                                             const double*     beta,
                                             double*           y,
                                             rocblas_int       incy,
                                             rocblas_stride    stridey,
                                             rocblas_int       batch_count)
try
{
    return rocblas_gemv_strided_batched_impl(handle,
                                             transA,
                                             m,
                                             n,
                                             alpha,
                                             A,
                                             lda,
                                             strideA,
                                             x,
                                             incx,
                                             stridex,
                                             beta,
                                             y,
                                             incy,
                                             stridey,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cgemv_strided_batched(rocblas_handle               handle,
                                             rocblas_operation            transA,
                                             rocblas_int                  m,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_stride               strideA,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             const rocblas_float_complex* beta,
                                             rocblas_float_complex*       y,
                                             rocblas_int                  incy,
                                             rocblas_stride               stridey,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_gemv_strided_batched_impl(handle,
                                             transA,
                                             m,
                                             n,
                                             alpha,
                                             A,
                                             lda,
                                             strideA,
                                             x,
                                             incx,
                                             stridex,
                                             beta,
                                             y,
                                             incy,
                                             stridey,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zgemv_strided_batched(rocblas_handle                handle,
                                             rocblas_operation             transA,
                                             rocblas_int                   m,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_stride                strideA,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             const rocblas_double_complex* beta,
                                             rocblas_double_complex*       y,
                                             rocblas_int                   incy,
                                             rocblas_stride                stridey,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_gemv_strided_batched_impl(handle,
                                             transA,
                                             m,
                                             n,
                                             alpha,
                                             A,
                                             lda,
                                             strideA,
                                             x,
                                             incx,
                                             stridex,
                                             beta,
                                             y,
                                             incy,
                                             stridey,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_hshgemv_strided_batched(rocblas_handle      handle,
                                               rocblas_operation   transA,
                                               rocblas_int         m,
                                               rocblas_int         n,
                                               const float*        alpha,
                                               const rocblas_half* A,
                                               rocblas_int         lda,
                                               rocblas_stride      strideA,
                                               const rocblas_half* x,
                                               rocblas_int         incx,
                                               rocblas_stride      stridex,
                                               const float*        beta,
                                               rocblas_half*       y,
                                               rocblas_int         incy,
                                               rocblas_stride      stridey,
                                               rocblas_int         batch_count)
try
{
    return rocblas_gemv_strided_batched_impl<rocblas_half, float, rocblas_half>(handle,
                                                                                transA,
                                                                                m,
                                                                                n,
                                                                                alpha,
                                                                                A,
                                                                                lda,
                                                                                strideA,
                                                                                x,
                                                                                incx,
                                                                                stridex,
                                                                                beta,
                                                                                y,
                                                                                incy,
                                                                                stridey,
                                                                                batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_hssgemv_strided_batched(rocblas_handle      handle,
                                               rocblas_operation   transA,
                                               rocblas_int         m,
                                               rocblas_int         n,
                                               const float*        alpha,
                                               const rocblas_half* A,
                                               rocblas_int         lda,
                                               rocblas_stride      strideA,
                                               const rocblas_half* x,
                                               rocblas_int         incx,
                                               rocblas_stride      stridex,
                                               const float*        beta,
                                               float*              y,
                                               rocblas_int         incy,
                                               rocblas_stride      stridey,
                                               rocblas_int         batch_count)
try
{
    return rocblas_gemv_strided_batched_impl<rocblas_half, float, float>(handle,
                                                                         transA,
                                                                         m,
                                                                         n,
                                                                         alpha,
                                                                         A,
                                                                         lda,
                                                                         strideA,
                                                                         x,
                                                                         incx,
                                                                         stridex,
                                                                         beta,
                                                                         y,
                                                                         incy,
                                                                         stridey,
                                                                         batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_tstgemv_strided_batched(rocblas_handle          handle,
                                               rocblas_operation       transA,
                                               rocblas_int             m,
                                               rocblas_int             n,
                                               const float*            alpha,
                                               const rocblas_bfloat16* A,
                                               rocblas_int             lda,
                                               rocblas_stride          strideA,
                                               const rocblas_bfloat16* x,
                                               rocblas_int             incx,
                                               rocblas_stride          stridex,
                                               const float*            beta,
                                               rocblas_bfloat16*       y,
                                               rocblas_int             incy,
                                               rocblas_stride          stridey,
                                               rocblas_int             batch_count)
try
{
    return rocblas_gemv_strided_batched_impl<rocblas_bfloat16, float, rocblas_bfloat16>(
        handle,
        transA,
        m,
        n,
        alpha,
        A,
        lda,
        strideA,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_tssgemv_strided_batched(rocblas_handle          handle,
                                               rocblas_operation       transA,
                                               rocblas_int             m,
                                               rocblas_int             n,
                                               const float*            alpha,
                                               const rocblas_bfloat16* A,
                                               rocblas_int             lda,
                                               rocblas_stride          strideA,
                                               const rocblas_bfloat16* x,
                                               rocblas_int             incx,
                                               rocblas_stride          stridex,
                                               const float*            beta,
                                               float*                  y,
                                               rocblas_int             incy,
                                               rocblas_stride          stridey,
                                               rocblas_int             batch_count)
try
{
    return rocblas_gemv_strided_batched_impl<rocblas_bfloat16, float, float>(handle,
                                                                             transA,
                                                                             m,
                                                                             n,
                                                                             alpha,
                                                                             A,
                                                                             lda,
                                                                             strideA,
                                                                             x,
                                                                             incx,
                                                                             stridex,
                                                                             beta,
                                                                             y,
                                                                             incy,
                                                                             stridey,
                                                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
