/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_reduction_template.hpp"
#include "utility.hpp"

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

    size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB, Tw>(n, batch_count);

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
    size_t         dev_bytes     = 0;
    rocblas_status checks_status = rocblas_reduction_setup<NB, ISBATCHED, Tw>(
        handle, n, x, incx, stridex, batch_count, results, name, name_bench, dev_bytes);
    if(checks_status != rocblas_status_continue)
    {
        return checks_status;
    }

    auto check_numerics = handle->check_numerics;

    if(check_numerics)
    {
        bool           is_input              = true;
        rocblas_status check_numerics_status = rocblas_internal_check_numerics_vector_template(
            name, handle, n, x, 0, incx, stridex, batch_count, check_numerics, is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    auto w_mem = handle->device_malloc(dev_bytes);
    if(!w_mem)
    {
        return rocblas_status_memory_error;
    }

    static constexpr rocblas_stride shiftx_0 = 0;
    rocblas_status status = rocblas_reduction_template<NB, ISBATCHED, FETCH, REDUCE, FINALIZE>(
        handle, n, x, shiftx_0, incx, stridex, batch_count, results, (Tw*)w_mem);
    if(status != rocblas_status_success)
        return status;

    if(check_numerics)
    {
        bool           is_input              = false;
        rocblas_status check_numerics_status = rocblas_internal_check_numerics_vector_template(
            name, handle, n, x, 0, incx, stridex, batch_count, check_numerics, is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }
    return status;
}
