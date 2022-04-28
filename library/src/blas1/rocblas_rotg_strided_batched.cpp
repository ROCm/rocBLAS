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
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_rotg.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = "rocblas_srotg_strided_batched";
    template <>
    constexpr char rocblas_rotg_name<double>[] = "rocblas_drotg_strided_batched";
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = "rocblas_crotg_strided_batched";
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = "rocblas_zrotg_strided_batched";

    template <class T, class U>
    rocblas_status rocblas_rotg_strided_batched_impl(rocblas_handle handle,
                                                     T*             a,
                                                     rocblas_stride stride_a,
                                                     T*             b,
                                                     rocblas_stride stride_b,
                                                     U*             c,
                                                     rocblas_stride stride_c,
                                                     T*             s,
                                                     rocblas_stride stride_s,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_rotg_name<T>,
                      a,
                      stride_a,
                      b,
                      stride_b,
                      c,
                      stride_c,
                      s,
                      stride_s,
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotg_strided_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "--stride_a",
                      stride_a,
                      "--stride_b",
                      stride_b,
                      "--stride_c",
                      stride_c,
                      "--stride_d",
                      stride_s,
                      "--batch_count",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_rotg_name<T>,
                        "stride_a",
                        stride_a,
                        "stride_b",
                        stride_b,
                        "stride_c",
                        stride_c,
                        "stride_d",
                        stride_s,
                        "batch_count",
                        batch_count);

        if(batch_count <= 0)
            return rocblas_status_success;

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotg_check_numerics_status
                = rocblas_rotg_check_numerics_template(rocblas_rotg_name<T>,
                                                       handle,
                                                       1,
                                                       a,
                                                       0,
                                                       stride_a,
                                                       b,
                                                       0,
                                                       stride_b,
                                                       c,
                                                       0,
                                                       stride_c,
                                                       s,
                                                       0,
                                                       stride_s,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
        }

        rocblas_status status = rocblas_rotg_template(
            handle, a, 0, stride_a, b, 0, stride_b, c, 0, stride_c, s, 0, stride_s, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rotg_check_numerics_status
                = rocblas_rotg_check_numerics_template(rocblas_rotg_name<T>,
                                                       handle,
                                                       1,
                                                       a,
                                                       0,
                                                       stride_a,
                                                       b,
                                                       0,
                                                       stride_b,
                                                       c,
                                                       0,
                                                       stride_c,
                                                       s,
                                                       0,
                                                       stride_s,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
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

rocblas_status rocblas_srotg_strided_batched(rocblas_handle handle,
                                             float*         a,
                                             rocblas_stride stride_a,
                                             float*         b,
                                             rocblas_stride stride_b,
                                             float*         c,
                                             rocblas_stride stride_c,
                                             float*         s,
                                             rocblas_stride stride_s,
                                             rocblas_int    batch_count)
try
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_drotg_strided_batched(rocblas_handle handle,
                                             double*        a,
                                             rocblas_stride stride_a,
                                             double*        b,
                                             rocblas_stride stride_b,
                                             double*        c,
                                             rocblas_stride stride_c,
                                             double*        s,
                                             rocblas_stride stride_s,
                                             rocblas_int    batch_count)
try
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_crotg_strided_batched(rocblas_handle         handle,
                                             rocblas_float_complex* a,
                                             rocblas_stride         stride_a,
                                             rocblas_float_complex* b,
                                             rocblas_stride         stride_b,
                                             float*                 c,
                                             rocblas_stride         stride_c,
                                             rocblas_float_complex* s,
                                             rocblas_stride         stride_s,
                                             rocblas_int            batch_count)
try
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zrotg_strided_batched(rocblas_handle          handle,
                                             rocblas_double_complex* a,
                                             rocblas_stride          stride_a,
                                             rocblas_double_complex* b,
                                             rocblas_stride          stride_b,
                                             double*                 c,
                                             rocblas_stride          stride_c,
                                             rocblas_double_complex* s,
                                             rocblas_stride          stride_s,
                                             rocblas_int             batch_count)
try
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
