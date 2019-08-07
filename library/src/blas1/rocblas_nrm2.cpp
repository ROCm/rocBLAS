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
    // HIP support up to 1024 threads/work itmes per thread block/work group
    constexpr int NB = 512;

    template <class To>
    struct rocblas_fetch_nrm2
    {
        template <class Ti>
        __forceinline__ __device__ To operator()(Ti x, ptrdiff_t tid)
        {
            return {fetch_abs2(x)};
        }
    };

    struct rocblas_finalize_nrm2
    {
        template <class To>
        __forceinline__ __host__ __device__ To operator()(To x)
        {
            return sqrt(x);
        }
    };

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
    template <typename Ti, typename To>
    rocblas_status rocblas_nrm2(
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

        return rocblas_reduction_kernel<NB,
                                        rocblas_fetch_nrm2<To>,
                                        rocblas_reduce_sum,
                                        rocblas_finalize_nrm2>(
            handle, n, x, incx, result, (To*)mem, blocks);
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
    return rocblas_nrm2(handle, n, x, incx, result);
}

rocblas_status rocblas_dnrm2(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    return rocblas_nrm2(handle, n, x, incx, result);
}

rocblas_status rocblas_scnrm2(rocblas_handle               handle,
                              rocblas_int                  n,
                              const rocblas_float_complex* x,
                              rocblas_int                  incx,
                              float*                       result)
{
    return rocblas_nrm2(handle, n, x, incx, result);
}

rocblas_status rocblas_dznrm2(rocblas_handle                handle,
                              rocblas_int                   n,
                              const rocblas_double_complex* x,
                              rocblas_int                   incx,
                              double*                       result)
{
    return rocblas_nrm2(handle, n, x, incx, result);
}

} // extern "C"
