/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <class To>
    struct rocblas_fetch_asum
    {
        template <typename Ti>
        __forceinline__ __device__ To operator()(Ti x, ptrdiff_t)
        {
            return {fetch_asum(x)};
        }
    };

    template <typename>
    constexpr char rocblas_asum_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_name<float>[] = "rocblas_sasum";
    template <>
    constexpr char rocblas_asum_name<double>[] = "rocblas_dasum";
    template <>
    constexpr char rocblas_asum_name<rocblas_float_complex>[] = "rocblas_scasum";
    template <>
    constexpr char rocblas_asum_name<rocblas_double_complex>[] = "rocblas_dzasum";

    // HIP support up to 1024 threads/work itmes per thread block/work group
    constexpr int NB = 512;

    // allocate workspace inside this API
    template <typename Ti, typename To>
    rocblas_status rocblas_asum(
        rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, To* result)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_asum_name<Ti>, n, x, incx);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f asum -r",
                      rocblas_precision_string<Ti>,
                      "-n",
                      n,
                      "--incx",
                      incx);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_asum_name<Ti>, "N", n, "incx", incx);

        if(!x || !result)
            return rocblas_status_invalid_pointer;

        // Quick return if possible.
        if(n <= 0 || incx <= 0)
        {
            if(handle->is_device_memory_size_query())
                return rocblas_status_size_unchanged;
            else if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
            else
                *result = 0;
            return rocblas_status_success;
        }

        auto blocks = (n - 1) / NB + 1;
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(To) * blocks);

        auto mem = handle->device_malloc(sizeof(To) * blocks);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_reduction_kernel<NB, rocblas_fetch_asum<To>>(
            handle, n, x, incx, result, (To*)mem, blocks);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sasum(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

rocblas_status rocblas_dasum(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

rocblas_status rocblas_scasum(rocblas_handle               handle,
                              rocblas_int                  n,
                              const rocblas_float_complex* x,
                              rocblas_int                  incx,
                              float*                       result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

rocblas_status rocblas_dzasum(rocblas_handle                handle,
                              rocblas_int                   n,
                              const rocblas_double_complex* x,
                              rocblas_int                   incx,
                              double*                       result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

} // extern "C"
