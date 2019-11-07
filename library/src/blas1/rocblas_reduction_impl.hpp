/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "logging.h"
#include "rocblas_reduction_template.hpp"
#include "utility.h"

template <bool ISBATCHED, typename Ti>
void reduction_log_bench(rocblas_handle handle,
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
void reduction_log_bench(rocblas_handle   handle,
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
void reduction_log_profile(rocblas_handle handle,
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
void reduction_log_profile(rocblas_handle   handle,
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
void reduction_log_trace(rocblas_handle handle,
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
void reduction_log_trace(rocblas_handle   handle,
                         rocblas_int      n,
                         const Ti* const* x,
                         rocblas_int      incx,
                         rocblas_stride   stridex,
                         rocblas_int      batch_count,
                         const char*      name)
{
    log_trace(handle, name, n, x, incx, batch_count);
}

// allocate workspace inside this API
template <rocblas_int NB,
          bool        ISBATCHED,
          typename FETCH,
          typename REDUCE,
          typename FINALIZE,
          typename Tw,
          typename U,
          typename Tr>
rocblas_status rocblas_reduction_impl(rocblas_handle handle,
                                      rocblas_int    n,
                                      U              x,
                                      rocblas_int    incx,
                                      rocblas_stride stridex,
                                      rocblas_int    batch_count,
                                      Tr*            results,
                                      const char*    name,
                                      const char*    name_bench)
{
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        reduction_log_trace<ISBATCHED>(handle, n, x, incx, stridex, batch_count, name);
    }

    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        reduction_log_bench<ISBATCHED>(handle, n, x, incx, stridex, batch_count, name_bench);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
    {
        reduction_log_profile<ISBATCHED>(handle, n, x, incx, stridex, batch_count, name);
    }

    if(!x || !results)
    {
        return rocblas_status_invalid_pointer;
    }

    size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB, Tw>(n, batch_count);
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(dev_bytes);
    }

    auto mem = handle->device_malloc(dev_bytes);
    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    static constexpr rocblas_int shiftx_0 = 0;
    return rocblas_reduction_template<NB, ISBATCHED, FETCH, REDUCE, FINALIZE>(
        handle, n, x, shiftx_0, incx, stridex, batch_count, results, (Tw*)mem);
}
