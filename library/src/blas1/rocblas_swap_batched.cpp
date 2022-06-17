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
#include "logging.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_swap.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_swap_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_swap_batched_name<float>[] = "rocblas_sswap_batched";
    template <>
    constexpr char rocblas_swap_batched_name<double>[] = "rocblas_dswap_batched";
    template <>
    constexpr char rocblas_swap_batched_name<rocblas_float_complex>[] = "rocblas_cswap_batched";
    template <>
    constexpr char rocblas_swap_batched_name<rocblas_double_complex>[] = "rocblas_zswap_batched";

    template <class T>
    rocblas_status rocblas_swap_batched_impl(rocblas_handle handle,
                                             rocblas_int    n,
                                             T* const       x[],
                                             rocblas_int    incx,
                                             T* const       y[],
                                             rocblas_int    incy,
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_swap_batched_name<T>, n, x, incx, y, incy, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f swap_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch_count",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_swap_batched_name<T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch_count",
                        batch_count);

        // Quick return if possible.
        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;

        if(!x || !y)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status swap_check_numerics_status
                = rocblas_swap_check_numerics(rocblas_swap_batched_name<T>,
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
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(swap_check_numerics_status != rocblas_status_success)
                return swap_check_numerics_status;
        }

        constexpr rocblas_int NB = ROCBLAS_SWAP_NB;
        rocblas_status        status
            = rocblas_swap_template<NB>(handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count);
        if(status != rocblas_status_success)
            return status;
        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status swap_check_numerics_status
                = rocblas_swap_check_numerics(rocblas_swap_batched_name<T>,
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
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(swap_check_numerics_status != rocblas_status_success)
                return swap_check_numerics_status;
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

rocblas_status rocblas_sswap_batched(rocblas_handle handle,
                                     rocblas_int    n,
                                     float* const   x[],
                                     rocblas_int    incx,
                                     float* const   y[],
                                     rocblas_int    incy,
                                     rocblas_int    batch_count)
try
{
    return rocblas_swap_batched_impl(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dswap_batched(rocblas_handle handle,
                                     rocblas_int    n,
                                     double* const  x[],
                                     rocblas_int    incx,
                                     double* const  y[],
                                     rocblas_int    incy,
                                     rocblas_int    batch_count)
try
{
    return rocblas_swap_batched_impl(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cswap_batched(rocblas_handle               handle,
                                     rocblas_int                  n,
                                     rocblas_float_complex* const x[],
                                     rocblas_int                  incx,
                                     rocblas_float_complex* const y[],
                                     rocblas_int                  incy,
                                     rocblas_int                  batch_count)
try
{
    return rocblas_swap_batched_impl(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zswap_batched(rocblas_handle                handle,
                                     rocblas_int                   n,
                                     rocblas_double_complex* const x[],
                                     rocblas_int                   incx,
                                     rocblas_double_complex* const y[],
                                     rocblas_int                   incy,
                                     rocblas_int                   batch_count)
try
{
    return rocblas_swap_batched_impl(handle, n, x, incx, y, incy, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
