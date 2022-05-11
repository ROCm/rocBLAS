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
#include "rocblas_rotmg.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_rotmg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotmg_name<float>[] = "rocblas_srotmg_strided_batched";
    template <>
    constexpr char rocblas_rotmg_name<double>[] = "rocblas_drotmg_strided_batched";

    template <class T>
    rocblas_status rocblas_rotmg_strided_batched_impl(rocblas_handle handle,
                                                      T*             d1,
                                                      rocblas_stride stride_d1,
                                                      T*             d2,
                                                      rocblas_stride stride_d2,
                                                      T*             x1,
                                                      rocblas_stride stride_x1,
                                                      const T*       y1,
                                                      rocblas_stride stride_y1,
                                                      T*             param,
                                                      rocblas_stride stride_param,
                                                      rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_rotmg_name<T>,
                      d1,
                      stride_d1,
                      d2,
                      stride_d2,
                      x1,
                      stride_x1,
                      y1,
                      stride_y1,
                      param,
                      stride_param,
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotmg_strided_batched -r",
                      rocblas_precision_string<T>,
                      "--batch_count",
                      batch_count,
                      "--stride_a",
                      stride_d1,
                      "--stride_b",
                      stride_d2,
                      "--stride_c",
                      stride_param,
                      "--stride_x",
                      stride_x1,
                      "--stride_y",
                      stride_y1);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_rotmg_name<T>,
                        "stride_a",
                        stride_d1,
                        "stride_b",
                        stride_d2,
                        "stride_c",
                        stride_param,
                        "stride_x",
                        stride_x1,
                        "stride_y",
                        stride_y1,
                        "batch_count",
                        batch_count);

        if(batch_count <= 0)
            return rocblas_status_success;
        if(!d1 || !d2 || !x1 || !y1 || !param)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotmg_check_numerics_status
                = rocblas_rotmg_check_numerics_template(rocblas_rotmg_name<T>,
                                                        handle,
                                                        1,
                                                        d1,
                                                        0,
                                                        stride_d1,
                                                        d2,
                                                        0,
                                                        stride_d2,
                                                        x1,
                                                        0,
                                                        stride_x1,
                                                        y1,
                                                        0,
                                                        stride_y1,
                                                        batch_count,
                                                        check_numerics,
                                                        is_input);
            if(rotmg_check_numerics_status != rocblas_status_success)
                return rotmg_check_numerics_status;
        }
        rocblas_status status = rocblas_rotmg_template(handle,
                                                       d1,
                                                       0,
                                                       stride_d1,
                                                       d2,
                                                       0,
                                                       stride_d2,
                                                       x1,
                                                       0,
                                                       stride_x1,
                                                       y1,
                                                       0,
                                                       stride_y1,
                                                       param,
                                                       0,
                                                       stride_param,
                                                       batch_count);
        if(status != rocblas_status_success)
            return status;
        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rotm_check_numerics_status
                = rocblas_rotmg_check_numerics_template(rocblas_rotmg_name<T>,
                                                        handle,
                                                        1,
                                                        d1,
                                                        0,
                                                        stride_d1,
                                                        d2,
                                                        0,
                                                        stride_d2,
                                                        x1,
                                                        0,
                                                        stride_x1,
                                                        y1,
                                                        0,
                                                        stride_y1,
                                                        batch_count,
                                                        check_numerics,
                                                        is_input);
            if(rotm_check_numerics_status != rocblas_status_success)
                return rotm_check_numerics_status;
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

ROCBLAS_EXPORT rocblas_status rocblas_srotmg_strided_batched(rocblas_handle handle,
                                                             float*         d1,
                                                             rocblas_stride stride_d1,
                                                             float*         d2,
                                                             rocblas_stride stride_d2,
                                                             float*         x1,
                                                             rocblas_stride stride_x1,
                                                             const float*   y1,
                                                             rocblas_stride stride_y1,
                                                             float*         param,
                                                             rocblas_stride stride_param,
                                                             rocblas_int    batch_count)
try
{
    return rocblas_rotmg_strided_batched_impl(handle,
                                              d1,
                                              stride_d1,
                                              d2,
                                              stride_d2,
                                              x1,
                                              stride_x1,
                                              y1,
                                              stride_y1,
                                              param,
                                              stride_param,
                                              batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

ROCBLAS_EXPORT rocblas_status rocblas_drotmg_strided_batched(rocblas_handle handle,
                                                             double*        d1,
                                                             rocblas_stride stride_d1,
                                                             double*        d2,
                                                             rocblas_stride stride_d2,
                                                             double*        x1,
                                                             rocblas_stride stride_x1,
                                                             const double*  y1,
                                                             rocblas_stride stride_y1,
                                                             double*        param,
                                                             rocblas_stride stride_param,
                                                             rocblas_int    batch_count)
try
{
    return rocblas_rotmg_strided_batched_impl(handle,
                                              d1,
                                              stride_d1,
                                              d2,
                                              stride_d2,
                                              x1,
                                              stride_x1,
                                              y1,
                                              stride_y1,
                                              param,
                                              stride_param,
                                              batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
