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

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_reduction.hpp"
#include "utility.hpp"

template <bool ISBATCHED, typename Ti>
void rocblas_reduction_log_bench(rocblas_handle handle,
                                 rocblas_int    n,
                                 const Ti*      x,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 rocblas_int    batch_count,
                                 const char*    name)
{
    if(ISBATCHED)
    {
        log_bench(handle,
                  "./rocblas-bench",
                  "-f",
                  name,
                  "-r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx,
                  "--stride_x",
                  stridex,
                  "--batch_count",
                  batch_count);
    }
    else
    {
        log_bench(handle,
                  "./rocblas-bench",
                  "-f",
                  name,
                  "-r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);
    }
}

template <bool ISBATCHED, typename Ti>
void rocblas_reduction_log_bench(rocblas_handle   handle,
                                 rocblas_int      n,
                                 const Ti* const* x,
                                 rocblas_int      incx,
                                 rocblas_stride   stridex,
                                 rocblas_int      batch_count,
                                 const char*      name)
{
    log_bench(handle,
              "./rocblas-bench",
              "-f",
              name,
              "-r",
              rocblas_precision_string<Ti>,
              "-n",
              n,
              "--incx",
              incx,
              "--batch_count",
              batch_count);
}

template <bool ISBATCHED, typename Ti>
void rocblas_reduction_log_profile(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ti*      x,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   rocblas_int    batch_count,
                                   const char*    name)
{
    if(ISBATCHED)
    {
        log_profile(
            handle, name, "N", n, "incx", incx, "stride_x", stridex, "batch_count", batch_count);
    }
    else
    {
        log_profile(handle, name, "N", n, "incx", incx);
    }
}

template <bool ISBATCHED, typename Ti>
void rocblas_reduction_log_profile(rocblas_handle   handle,
                                   rocblas_int      n,
                                   const Ti* const* x,
                                   rocblas_int      incx,
                                   rocblas_stride   stridex,
                                   rocblas_int      batch_count,
                                   const char*      name)
{
    log_profile(handle, name, "N", n, "incx", incx, "batch_count", batch_count);
}

template <bool ISBATCHED, typename Ti>
void rocblas_reduction_log_trace(rocblas_handle handle,
                                 rocblas_int    n,
                                 const Ti*      x,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 rocblas_int    batch_count,
                                 const char*    name)
{
    if(ISBATCHED)
    {
        log_trace(handle, name, n, x, incx, stridex, batch_count);
    }
    else
    {
        log_trace(handle, name, n, x, incx);
    }
}

template <bool ISBATCHED, typename Ti>
void rocblas_reduction_log_trace(rocblas_handle   handle,
                                 rocblas_int      n,
                                 const Ti* const* x,
                                 rocblas_int      incx,
                                 rocblas_stride   stridex,
                                 rocblas_int      batch_count,
                                 const char*      name)
{
    log_trace(handle, name, n, x, incx, batch_count);
}

template <rocblas_int NB, bool ISBATCHED, typename Tw, typename U, typename Tr>
inline rocblas_status rocblas_reduction_setup(rocblas_handle handle,
                                              rocblas_int    n,
                                              U              x,
                                              rocblas_int    incx,
                                              rocblas_stride stridex,
                                              rocblas_int    batch_count,
                                              Tr*            results,
                                              const char*    name,
                                              const char*    name_bench,
                                              size_t&        work_size)
{
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    size_t dev_bytes = rocblas_reduction_kernel_workspace_size<rocblas_int, NB, Tw>(n, batch_count);

    if(handle->is_device_memory_size_query())
    {
        if(n <= 0 || incx <= 0 || (ISBATCHED && batch_count <= 0))
        {
            return rocblas_status_size_unchanged;
        }
        else
        {
            return handle->set_optimal_device_memory_size(dev_bytes);
        }
    }

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        rocblas_reduction_log_trace<ISBATCHED>(handle, n, x, incx, stridex, batch_count, name);
    }

    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        rocblas_reduction_log_bench<ISBATCHED>(
            handle, n, x, incx, stridex, batch_count, name_bench);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
    {
        rocblas_reduction_log_profile<ISBATCHED>(handle, n, x, incx, stridex, batch_count, name);
    }

    if(!results)
    {
        return rocblas_status_invalid_pointer;
    }

    // Quick return if possible.
    if(n <= 0 || incx <= 0 || (ISBATCHED && batch_count <= 0))
    {
        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            if(batch_count > 0)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(results, 0, batch_count * sizeof(Tr), handle->get_stream()));
            }
        }
        else
        {
            if(batch_count > 0)
                memset(results, 0, batch_count * sizeof(Tr));
        }
        return rocblas_status_success;
    }

    if(!x)
    {
        return rocblas_status_invalid_pointer;
    }

    work_size = dev_bytes;

    return rocblas_status_continue;
}
