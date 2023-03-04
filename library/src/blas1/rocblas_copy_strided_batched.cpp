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
#include "rocblas_copy.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_copy_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_copy_strided_batched_name<float>[] = "rocblas_scopy_strided_batched";
    template <>
    constexpr char rocblas_copy_strided_batched_name<double>[] = "rocblas_dcopy_strided_batched";
    template <>
    constexpr char rocblas_copy_strided_batched_name<rocblas_half>[]
        = "rocblas_hcopy_strided_batched";
    template <>
    constexpr char rocblas_copy_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_ccopy_strided_batched";
    template <>
    constexpr char rocblas_copy_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zcopy_strided_batched";

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_copy_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     T*             y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stridey,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_copy_strided_batched_name<T>,
                      n,
                      x,
                      incx,
                      stridex,
                      y,
                      incy,
                      stridey,
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f copy_strided_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--stride_x",
                      stridex,
                      "--incy",
                      incy,
                      "--stride_y",
                      stridey,
                      "--batch_count",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_copy_strided_batched_name<T>,
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
                        "batch_count",
                        batch_count);

        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;
        if(!x || !y)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status copy_check_numerics_status
                = rocblas_copy_check_numerics(rocblas_copy_strided_batched_name<T>,
                                              handle,
                                              n,
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
            if(copy_check_numerics_status != rocblas_status_success)
                return copy_check_numerics_status;
        }

        rocblas_status status = rocblas_copy_template<false, NB>(
            handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status copy_check_numerics_status
                = rocblas_copy_check_numerics(rocblas_copy_strided_batched_name<T>,
                                              handle,
                                              n,
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

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(name_, T_)                                                  \
    rocblas_status name_(rocblas_handle handle,                          \
                         rocblas_int    n,                               \
                         const T_*      x,                               \
                         rocblas_int    incx,                            \
                         rocblas_stride stridex,                         \
                         T_*            y,                               \
                         rocblas_int    incy,                            \
                         rocblas_stride stridey,                         \
                         rocblas_int    batch_count)                     \
    try                                                                  \
    {                                                                    \
        return rocblas_copy_strided_batched_impl<ROCBLAS_COPY_NB>(       \
            handle, n, x, incx, stridex, y, incy, stridey, batch_count); \
    }                                                                    \
    catch(...)                                                           \
    {                                                                    \
        return exception_to_rocblas_status();                            \
    }

IMPL(rocblas_scopy_strided_batched, float);
IMPL(rocblas_dcopy_strided_batched, double);
IMPL(rocblas_hcopy_strided_batched, rocblas_half);
IMPL(rocblas_ccopy_strided_batched, rocblas_float_complex);
IMPL(rocblas_zcopy_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
