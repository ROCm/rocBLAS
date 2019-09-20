/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_asum_strided_batched.hpp"
#include "logging.h"
#include "utility.h"

namespace
{

    template <typename>
    constexpr char rocblas_asum_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_strided_batched_name<float>[] = "rocblas_sasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<double>[] = "rocblas_dasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_scasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_dzasum_strided_batched";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_asum_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const Ti*      x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     rocblas_int    batch_count,
                                                     To*            results)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(
                handle, rocblas_asum_strided_batched_name<Ti>, n, x, incx, stridex, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f asum_strided_batched -r",
                      rocblas_precision_string<Ti>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--stride_x",
                      stridex,
                      "--batch",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_asum_strided_batched_name<Ti>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stride_x",
                        stridex,
                        "batch",
                        batch_count);

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

        return rocblas_asum_strided_batched_template<NB>(
            handle, n, x, 0, incx, stridex, batch_count, (To*)mem, results);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sasum_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   x,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             rocblas_int    batch_count,
                                             float*         results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_asum_strided_batched_impl<NB>(handle, n, x, incx, stridex, batch_count, results);
}

rocblas_status rocblas_dasum_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  x,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             rocblas_int    batch_count,
                                             double*        results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_asum_strided_batched_impl<NB>(handle, n, x, incx, stridex, batch_count, results);
}

rocblas_status rocblas_scasum_strided_batched(rocblas_handle               handle,
                                              rocblas_int                  n,
                                              const rocblas_float_complex* x,
                                              rocblas_int                  incx,
                                              rocblas_stride               stridex,
                                              rocblas_int                  batch_count,
                                              float*                       results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_asum_strided_batched_impl<NB>(handle, n, x, incx, stridex, batch_count, results);
}

rocblas_status rocblas_dzasum_strided_batched(rocblas_handle                handle,
                                              rocblas_int                   n,
                                              const rocblas_double_complex* x,
                                              rocblas_int                   incx,
                                              rocblas_stride                stridex,
                                              rocblas_int                   batch_count,
                                              double*                       results)
{
    constexpr rocblas_int NB = 512;
    return rocblas_asum_strided_batched_impl<NB>(handle, n, x, incx, stridex, batch_count, results);
}

} // extern "C"
