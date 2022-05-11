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
#include "rocblas_copy.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_copy_name[] = "unknown";
    template <>
    constexpr char rocblas_copy_name<float>[] = "rocblas_scopy";
    template <>
    constexpr char rocblas_copy_name<double>[] = "rocblas_dcopy";
    template <>
    constexpr char rocblas_copy_name<rocblas_half>[] = "rocblas_hcopy";
    template <>
    constexpr char rocblas_copy_name<rocblas_float_complex>[] = "rocblas_ccopy";
    template <>
    constexpr char rocblas_copy_name<rocblas_double_complex>[] = "rocblas_zcopy";

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_copy_impl(
        rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_copy_name<T>, n, x, incx, y, incy);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f copy -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_copy_name<T>, "N", n, "incx", incx, "incy", incy);

        if(n <= 0)
            return rocblas_status_success;
        if(!x || !y)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status copy_check_numerics_status
                = rocblas_copy_check_numerics(rocblas_copy_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(copy_check_numerics_status != rocblas_status_success)
                return copy_check_numerics_status;
        }

        rocblas_status status
            = rocblas_copy_template<false, NB>(handle, n, x, 0, incx, 0, y, 0, incy, 0, 1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status copy_check_numerics_status
                = rocblas_copy_check_numerics(rocblas_copy_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(copy_check_numerics_status != rocblas_status_success)
                return copy_check_numerics_status;
        }
        return status;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_scopy(rocblas_handle handle,
                             rocblas_int    n,
                             const float*   x,
                             rocblas_int    incx,
                             float*         y,
                             rocblas_int    incy)
try
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dcopy(rocblas_handle handle,
                             rocblas_int    n,
                             const double*  x,
                             rocblas_int    incx,
                             double*        y,
                             rocblas_int    incy)
try
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_hcopy(rocblas_handle      handle,
                             rocblas_int         n,
                             const rocblas_half* x,
                             rocblas_int         incx,
                             rocblas_half*       y,
                             rocblas_int         incy)
try
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ccopy(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             rocblas_float_complex*       y,
                             rocblas_int                  incy)
try
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zcopy(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             rocblas_double_complex*       y,
                             rocblas_int                   incy)
try
{
    constexpr int NB = 256;
    return rocblas_copy_impl<NB>(handle, n, x, incx, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
