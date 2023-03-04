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
#include "rocblas_block_sizes.h"
#include "rocblas_dot.hpp"
#include "utility.hpp"

namespace
{

    // HIP support up to 1024 threads/work itemes per thread block/work group
    // setting to 512 for gfx803.
    constexpr int NB = ROCBLAS_DOT_NB;

    template <bool, typename>
    constexpr char rocblas_dot_batched_name[] = "unknown";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, float>[] = "rocblas_sdot_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, double>[] = "rocblas_ddot_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, rocblas_half>[] = "rocblas_hdot_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, rocblas_bfloat16>[] = "rocblas_bfdot_batched";
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_float_complex>[]
        = "rocblas_cdotc_batched";
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_float_complex>[]
        = "rocblas_cdotu_batched";
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_double_complex>[]
        = "rocblas_zdotc_batched";
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_double_complex>[]
        = "rocblas_zdotu_batched";

    // allocate workspace inside this API
    template <bool CONJ, typename T, typename T2 = T>
    inline rocblas_status rocblas_dot_batched_impl(rocblas_handle handle,
                                                   rocblas_int    n,
                                                   const T* const x[],
                                                   rocblas_int    incx,
                                                   const T* const y[],
                                                   rocblas_int    incy,
                                                   rocblas_int    batch_count,
                                                   T*             results)
    {
        static constexpr int WIN = rocblas_dot_WIN<T>();

        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB * WIN, T2>(n, batch_count);
        if(handle->is_device_memory_size_query())
        {
            if(n <= 0 || batch_count <= 0)
                return rocblas_status_size_unchanged;
            else
                return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_dot_batched_name<CONJ, T>, n, x, incx, y, incy, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f dot_batched -r",
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
                        rocblas_dot_batched_name<CONJ, T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch_count",
                        batch_count);

        // Quick return if possible.
        if(batch_count <= 0)
        {
            return rocblas_status_success;
        }

        if(n <= 0)
        {
            if(!results)
                return rocblas_status_invalid_pointer;
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    results, 0, sizeof(*results) * batch_count, handle->get_stream()));
            else
                memset(results, 0, sizeof(*results) * batch_count);
            return rocblas_status_success;
        }

        if(!x || !y || !results)
            return rocblas_status_invalid_pointer;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status dot_check_numerics_status
                = rocblas_dot_check_numerics(rocblas_dot_batched_name<CONJ, T>,
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
            if(dot_check_numerics_status != rocblas_status_success)
                return dot_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_dot_template<NB, CONJ, T>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count, results, (T2*)w_mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status dot_check_numerics_status
                = rocblas_dot_check_numerics(rocblas_dot_batched_name<CONJ, T>,
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
            if(dot_check_numerics_status != rocblas_status_success)
                return dot_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(name_, conj_, T_, Tex_)                            \
    rocblas_status name_(rocblas_handle  handle,                \
                         rocblas_int     n,                     \
                         const T_* const x[],                   \
                         rocblas_int     incx,                  \
                         const T_* const y[],                   \
                         rocblas_int     incy,                  \
                         rocblas_int     batch_count,           \
                         T_*             results)               \
    try                                                         \
    {                                                           \
        return rocblas_dot_batched_impl<conj_, T_, Tex_>(       \
            handle, n, x, incx, y, incy, batch_count, results); \
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

IMPL(rocblas_sdot_batched, false, float, float);
IMPL(rocblas_ddot_batched, false, double, double);
IMPL(rocblas_hdot_batched, false, rocblas_half, rocblas_half);
IMPL(rocblas_bfdot_batched, false, rocblas_bfloat16, float);
IMPL(rocblas_cdotu_batched, false, rocblas_float_complex, rocblas_float_complex);
IMPL(rocblas_zdotu_batched, false, rocblas_double_complex, rocblas_double_complex);
IMPL(rocblas_cdotc_batched, true, rocblas_float_complex, rocblas_float_complex);
IMPL(rocblas_zdotc_batched, true, rocblas_double_complex, rocblas_double_complex);

#undef IMPL

} // extern "C"
