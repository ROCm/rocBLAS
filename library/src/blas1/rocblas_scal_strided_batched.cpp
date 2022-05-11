/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_scal.hpp"
#include "utility.hpp"

namespace
{

    template <typename T, typename = T>
    constexpr char rocblas_scal_name[] = "unknown";
    template <>
    constexpr char rocblas_scal_name<float>[] = "rocblas_sscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<double>[] = "rocblas_dscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex>[] = "rocblas_cscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex>[] = "rocblas_zscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex, float>[]
        = "rocblas_csscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex, double>[]
        = "rocblas_zdscal_strided_batched";

    template <rocblas_int NB, typename T, typename U>
    rocblas_status rocblas_scal_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const U*       alpha,
                                                     T*             x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;

        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_scal_name<T, U>,
                      n,
                      LOG_TRACE_SCALAR_VALUE(handle, alpha),
                      x,
                      incx,
                      stridex,
                      batch_count);

        // there are an extra 2 scal functions, thus
        // the -r mode will not work correctly. Substitute
        // with --a_type and --b_type (?)
        // ANSWER: -r is syntatic sugar; the types can be specified separately
        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            log_bench(handle,
                      "./rocblas-bench -f scal_strided_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "-n",
                      n,
                      LOG_BENCH_SCALAR_VALUE(handle, alpha),
                      "--incx",
                      incx,
                      "--stride_x",
                      stridex,
                      "--batch_count",
                      batch_count);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_scal_name<T, U>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stride_x",
                        stridex,
                        "batch_count",
                        batch_count);

        if(n <= 0 || incx <= 0 || batch_count <= 0)
            return rocblas_status_success;
        if(!x || !alpha)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status check_numerics_status
                = rocblas_internal_check_numerics_vector_template(rocblas_scal_name<T>,
                                                                  handle,
                                                                  n,
                                                                  x,
                                                                  0,
                                                                  incx,
                                                                  stridex,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }
        rocblas_status status = rocblas_internal_scal_template<NB, T>(
            handle, n, alpha, 0, x, 0, incx, stridex, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status check_numerics_status
                = rocblas_internal_check_numerics_vector_template(rocblas_scal_name<T>,
                                                                  handle,
                                                                  n,
                                                                  x,
                                                                  0,
                                                                  incx,
                                                                  stridex,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
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

rocblas_status rocblas_sscal_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   alpha,
                                             float*         x,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             rocblas_int    batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dscal_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  alpha,
                                             double*        x,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             rocblas_int    batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cscal_strided_batched(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             rocblas_float_complex*       x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             rocblas_int                  batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zscal_strided_batched(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             rocblas_double_complex*       x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             rocblas_int                   batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

// Scal with a real alpha & complex vector
rocblas_status rocblas_csscal_strided_batched(rocblas_handle         handle,
                                              rocblas_int            n,
                                              const float*           alpha,
                                              rocblas_float_complex* x,
                                              rocblas_int            incx,
                                              rocblas_stride         stridex,
                                              rocblas_int            batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zdscal_strided_batched(rocblas_handle          handle,
                                              rocblas_int             n,
                                              const double*           alpha,
                                              rocblas_double_complex* x,
                                              rocblas_int             incx,
                                              rocblas_stride          stridex,
                                              rocblas_int             batch_count)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
