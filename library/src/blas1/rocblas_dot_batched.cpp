/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_dot.hpp"
#include "utility.h"

namespace
{

    // HIP support up to 1024 threads/work itemes per thread block/work group
    // setting to 512 for gfx803.
    constexpr int NB = 512;

    template <bool, typename>
    constexpr char rocblas_dot_batched_name[] = "unknown";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, float>[] = "rocblas_sdot_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, double>[] = "rocblas_ddot_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, rocblas_half>[] = "rocblas_hdot_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, rocblas_bfloat16>[] = "rocblas_bfdot_batched";
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_float_complex>[]
        = "rocblas_cdotc_batched";
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_float_complex>[]
        = "rocblas_cdotu_batched";
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_double_complex>[]
        = "rocblas_zdotc_batched";
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_double_complex>[]
        = "rocblas_zdotu_batched";

    // allocate workspace inside this API
    template <bool CONJ, typename T, typename T2 = T>
    inline rocblas_status rocblas_dot_batched_impl(rocblas_handle handle,
                                                   rocblas_int    n,
                                                   const T* const x[],
                                                   rocblas_int    incx,
                                                   const T* const y[],
                                                   rocblas_int    incy,
                                                   rocblas_int    batch_count,
                                                   T*             results)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB * WIN, T2>(n, batch_count);
        if(handle->is_device_memory_size_query())
        {
            if(n <= 0 || batch_count <= 0)
                return rocblas_status_size_unchanged;
            else
                return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_dot_batched_name<CONJ, T>, n, x, incx, y, incy, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f dot_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch_count",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_dot_batched_name<CONJ, T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch_count",
                        batch_count);

        // Quick return if possible.
        if(n <= 0 || batch_count <= 0)
        {
            if(!results)
                return rocblas_status_invalid_pointer;
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(results, 0, sizeof(*results), handle->rocblas_stream));
            else
                *results = T(0);
            return rocblas_status_success;
        }

        if(!x || !y || !results)
            return rocblas_status_invalid_pointer;

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_dot_template<NB, CONJ, T>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count, results, (T2*)mem);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sdot_batched(rocblas_handle     handle,
                                    rocblas_int        n,
                                    const float* const x[],
                                    rocblas_int        incx,
                                    const float* const y[],
                                    rocblas_int        incy,
                                    rocblas_int        batch_count,
                                    float*             results)
try
{
    return rocblas_dot_batched_impl<false>(handle, n, x, incx, y, incy, batch_count, results);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ddot_batched(rocblas_handle      handle,
                                    rocblas_int         n,
                                    const double* const x[],
                                    rocblas_int         incx,
                                    const double* const y[],
                                    rocblas_int         incy,
                                    rocblas_int         batch_count,
                                    double*             results)
try
{
    return rocblas_dot_batched_impl<false>(handle, n, x, incx, y, incy, batch_count, results);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_hdot_batched(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const rocblas_half* const x[],
                                    rocblas_int               incx,
                                    const rocblas_half* const y[],
                                    rocblas_int               incy,
                                    rocblas_int               batch_count,
                                    rocblas_half*             result)
try
{
    return rocblas_dot_batched_impl<false>(handle,
                                           n,
                                           (const rocblas_half* const*)x,
                                           incx,
                                           (const rocblas_half* const*)y,
                                           incy,
                                           batch_count,
                                           (rocblas_half*)result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_bfdot_batched(rocblas_handle                handle,
                                     rocblas_int                   n,
                                     const rocblas_bfloat16* const x[],
                                     rocblas_int                   incx,
                                     const rocblas_bfloat16* const y[],
                                     rocblas_int                   incy,
                                     rocblas_int                   batch_count,
                                     rocblas_bfloat16*             result)
try
{
    return rocblas_dot_batched_impl<false, rocblas_bfloat16, float>(
        handle, n, x, incx, y, incy, batch_count, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cdotu_batched(rocblas_handle                     handle,
                                     rocblas_int                        n,
                                     const rocblas_float_complex* const x[],
                                     rocblas_int                        incx,
                                     const rocblas_float_complex* const y[],
                                     rocblas_int                        incy,
                                     rocblas_int                        batch_count,
                                     rocblas_float_complex*             results)
try
{
    return rocblas_dot_batched_impl<false>(handle, n, x, incx, y, incy, batch_count, results);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zdotu_batched(rocblas_handle                      handle,
                                     rocblas_int                         n,
                                     const rocblas_double_complex* const x[],
                                     rocblas_int                         incx,
                                     const rocblas_double_complex* const y[],
                                     rocblas_int                         incy,
                                     rocblas_int                         batch_count,
                                     rocblas_double_complex*             results)
try
{
    return rocblas_dot_batched_impl<false>(handle, n, x, incx, y, incy, batch_count, results);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cdotc_batched(rocblas_handle                     handle,
                                     rocblas_int                        n,
                                     const rocblas_float_complex* const x[],
                                     rocblas_int                        incx,
                                     const rocblas_float_complex* const y[],
                                     rocblas_int                        incy,
                                     rocblas_int                        batch_count,
                                     rocblas_float_complex*             results)
try
{
    return rocblas_dot_batched_impl<true>(handle, n, x, incx, y, incy, batch_count, results);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zdotc_batched(rocblas_handle                      handle,
                                     rocblas_int                         n,
                                     const rocblas_double_complex* const x[],
                                     rocblas_int                         incx,
                                     const rocblas_double_complex* const y[],
                                     rocblas_int                         incy,
                                     rocblas_int                         batch_count,
                                     rocblas_double_complex*             results)
try
{
    return rocblas_dot_batched_impl<true>(handle, n, x, incx, y, incy, batch_count, results);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
