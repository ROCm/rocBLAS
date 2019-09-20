/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_nrm2_batched.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_nrm2_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_nrm2_batched_name<float>[] = "rocblas_snrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<double>[] = "rocblas_dnrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<rocblas_half>[] = "rocblas_hnrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<rocblas_float_complex>[] = "rocblas_scnrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<rocblas_double_complex>[] = "rocblas_dznrm2_batched";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_nrm2_batched_impl(rocblas_handle  handle,
                                             rocblas_int     n,
                                             const Ti* const x[],
                                             rocblas_int     incx,
                                             rocblas_int     batch_count,
                                             To*             results)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_nrm2_batched_name<Ti>, n, x, incx, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f nrm2_batched -r",
                      rocblas_precision_string<Ti>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--batch",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(
                handle, rocblas_nrm2_batched_name<Ti>, "N", n, "incx", incx, "batch", batch_count);

        if(!x || !results)
            return rocblas_status_invalid_pointer;

        if(batch_count < 0)
            return rocblas_status_invalid_size;

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB>(n, batch_count, results);

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_nrm2_batched_template<NB>(
            handle, n, x, 0, incx, batch_count, (To*)mem, results);
    }
}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_snrm2_batched(rocblas_handle     handle,
                                     rocblas_int        n,
                                     const float* const x[],
                                     rocblas_int        incx,
                                     rocblas_int        batch_count,
                                     float*             results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_batched_impl<NB>(handle, n, x, incx, batch_count, results);
}

rocblas_status rocblas_dnrm2_batched(rocblas_handle      handle,
                                     rocblas_int         n,
                                     const double* const x[],
                                     rocblas_int         incx,
                                     rocblas_int         batch_count,
                                     double*             results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_batched_impl<NB>(handle, n, x, incx, batch_count, results);
}

rocblas_status rocblas_scnrm2_batched(rocblas_handle                     handle,
                                      rocblas_int                        n,
                                      const rocblas_float_complex* const x[],
                                      rocblas_int                        incx,
                                      rocblas_int                        batch_count,
                                      float*                             results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_batched_impl<NB>(handle, n, x, incx, batch_count, results);
}

rocblas_status rocblas_dznrm2_batched(rocblas_handle                      handle,
                                      rocblas_int                         n,
                                      const rocblas_double_complex* const x[],
                                      rocblas_int                         incx,
                                      rocblas_int                         batch_count,
                                      double*                             results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_batched_impl<NB>(handle, n, x, incx, batch_count, results);
}

} // extern "C"
