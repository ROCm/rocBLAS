/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_nrm2.hpp"
#include "logging.h"
#include "utility.h"

namespace
{

    template <typename>
    constexpr char rocblas_nrm2_name[] = "unknown";
    template <>
    constexpr char rocblas_nrm2_name<float>[] = "rocblas_snrm2";
    template <>
    constexpr char rocblas_nrm2_name<double>[] = "rocblas_dnrm2";
    template <>
    constexpr char rocblas_nrm2_name<rocblas_half>[] = "rocblas_hnrm2";
    template <>
    constexpr char rocblas_nrm2_name<rocblas_float_complex>[] = "rocblas_scnrm2";
    template <>
    constexpr char rocblas_nrm2_name<rocblas_double_complex>[] = "rocblas_dznrm2";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_nrm2_impl(
        rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, To* result)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_nrm2_name<Ti>, n, x, incx);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f nrm2 -r",
                      rocblas_precision_string<Ti>,
                      "-n",
                      n,
                      "--incx",
                      incx);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_nrm2_name<Ti>, "N", n, "incx", incx);

        if(!x || !result)
            return rocblas_status_invalid_pointer;

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB>(n, 1, result);

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_nrm2_template<NB>(handle, n, x, 0, incx, (To*)mem, result);
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_snrm2(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_impl<NB>(handle, n, x, incx, result);
}

rocblas_status rocblas_dnrm2(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_impl<NB>(handle, n, x, incx, result);
}

rocblas_status rocblas_scnrm2(rocblas_handle               handle,
                              rocblas_int                  n,
                              const rocblas_float_complex* x,
                              rocblas_int                  incx,
                              float*                       result)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_impl<NB>(handle, n, x, incx, result);
}

rocblas_status rocblas_dznrm2(rocblas_handle                handle,
                              rocblas_int                   n,
                              const rocblas_double_complex* x,
                              rocblas_int                   incx,
                              double*                       result)
{
    constexpr rocblas_int NB = 512;
    return rocblas_nrm2_impl<NB>(handle, n, x, incx, result);
}

} // extern "C"
