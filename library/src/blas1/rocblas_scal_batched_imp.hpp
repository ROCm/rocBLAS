/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "utility.hpp"

namespace
{
    template <typename T, typename = T>
    constexpr char rocblas_scal_name[] = "unknown";
    template <>
    constexpr char rocblas_scal_name<float>[] = ROCBLAS_API_STR(sscal_batched);
    template <>
    constexpr char rocblas_scal_name<double>[] = ROCBLAS_API_STR(dscal_batched);
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex>[] = ROCBLAS_API_STR(cscal_batched);
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex>[] = ROCBLAS_API_STR(zscal_batched);
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex, float>[]
        = ROCBLAS_API_STR(csscal_batched);
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex, double>[]
        = ROCBLAS_API_STR(zdscal_batched);

    template <typename API_INT, typename T, typename U>
    rocblas_status rocblas_scal_batched_impl(rocblas_handle handle,
                                             API_INT        n,
                                             const U*       alpha,
                                             T* const       x[],
                                             API_INT        incx,
                                             API_INT        batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        if constexpr(std::is_same_v<API_INT, int>)
        {
            if(batch_count > c_YZ_grid_launch_limit && handle->isYZGridDim16bit())
            {
                return rocblas_status_invalid_size;
            }
        }

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;

        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_scal_name<T, U>,
                      n,
                      LOG_TRACE_SCALAR_VALUE(handle, alpha),
                      x,
                      incx,
                      batch_count);

        // there are an extra 2 scal functions, thus
        // the -r mode will not work correctly. Substitute
        // with --a_type and --b_type (?)
        // ANSWER: -r is syntatic sugar; the types can be specified separately
        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            log_bench(handle,
                      ROCBLAS_API_BENCH " -f scal_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "-n",
                      n,
                      LOG_BENCH_SCALAR_VALUE(handle, alpha),
                      "--incx",
                      incx,
                      "--batch_count",
                      batch_count);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(
                handle, rocblas_scal_name<T, U>, "N", n, "incx", incx, "batch_count", batch_count);

        if(n <= 0 || incx <= 0 || batch_count <= 0)
            return rocblas_status_success;
        if(!x || !alpha)
            return rocblas_status_invalid_pointer;

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(*alpha == 1)
                return rocblas_status_success;
        }

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
                                                                  0,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }
        rocblas_status status = ROCBLAS_API(rocblas_internal_scal_batched_template)(
            handle, n, alpha, 0, x, 0, incx, 0, batch_count);
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
                                                                  0,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        return status;
    }
}

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(name_, TI_, TA_, T_)                                                                 \
    rocblas_status name_(                                                                         \
        rocblas_handle handle, TI_ n, const TA_* alpha, T_* const x[], TI_ incx, TI_ batch_count) \
    try                                                                                           \
    {                                                                                             \
        return rocblas_scal_batched_impl<TI_>(handle, n, alpha, x, incx, batch_count);            \
    }                                                                                             \
    catch(...)                                                                                    \
    {                                                                                             \
        return exception_to_rocblas_status();                                                     \
    }

#define INST_SCAL_BATCHED_C_API(TI_)                                                               \
    extern "C" {                                                                                   \
    IMPL(ROCBLAS_API(rocblas_sscal_batched), TI_, float, float);                                   \
    IMPL(ROCBLAS_API(rocblas_dscal_batched), TI_, double, double);                                 \
    IMPL(ROCBLAS_API(rocblas_cscal_batched), TI_, rocblas_float_complex, rocblas_float_complex);   \
    IMPL(ROCBLAS_API(rocblas_zscal_batched), TI_, rocblas_double_complex, rocblas_double_complex); \
    IMPL(ROCBLAS_API(rocblas_csscal_batched), TI_, float, rocblas_float_complex);                  \
    IMPL(ROCBLAS_API(rocblas_zdscal_batched), TI_, double, rocblas_double_complex);                \
    } // extern "C"

// Scal with a real alpha & complex vector (cs and zd forms)
