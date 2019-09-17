/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "reduction.h"
#include "rocblas.h"
#include "utility.h"
#include "rocblas_dot.hpp"

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
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_float_complex>[] = "rocblas_cdotc_batched";
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_float_complex>[] = "rocblas_cdotu_batched";
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_double_complex>[] = "rocblas_zdotc_batched";
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_double_complex>[] = "rocblas_zdotu_batched";

    // allocate workspace inside this API
    template <bool CONJ, typename T>
    rocblas_status rocblas_dot_batched(rocblas_handle handle,
                               rocblas_int    n,
                               const T*       const x[],
                               rocblas_int    incx,
                               const T*       const y[],
                               rocblas_int    incy,
                               rocblas_int    batch_count,
                               T*             result)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

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
            log_profile(handle, rocblas_dot_batched_name<CONJ, T>, "N", n, "incx", incx, "incy", incy, "batch_count", batch_count);

        if(!x || !y || !result)
            return rocblas_status_invalid_pointer;
        
        if(batch_count <= 0)
            return rocblas_status_invalid_size;

        auto blocks = (n - 1) / NB + 1;
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(T) * blocks * batch_count);

        auto mem = handle->device_malloc(sizeof(T) * blocks * batch_count);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_dot_template<NB, CONJ, T>(handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count, result, (T*)mem, blocks);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sdot_batched(rocblas_handle handle,
                            rocblas_int    n,
                            const float*    const x[],
                            rocblas_int    incx,
                            const float*   const y[],
                            rocblas_int    incy,
                            rocblas_int    batch_count,
                            float*         result)
{
    return rocblas_dot_batched<false>(handle, n, x, incx, y, incy, batch_count, result);
}

rocblas_status rocblas_ddot_batched(rocblas_handle handle,
                            rocblas_int    n,
                            const double*   const x[],
                            rocblas_int    incx,
                            const double*  const y[],
                            rocblas_int    incy,
                            rocblas_int    batch_count,
                            double*        result)
{
    return rocblas_dot_batched<false>(handle, n, x, incx, y, incy, batch_count, result);
}

rocblas_status rocblas_cdotu_batched(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* const x[],
                             rocblas_int                  incx,
                             const rocblas_float_complex* const y[],
                             rocblas_int                  incy,
                            rocblas_int    batch_count,
                             rocblas_float_complex*       result)
{
    return rocblas_dot_batched<false>(handle, n, x, incx, y, incy, batch_count, result);
}

rocblas_status rocblas_zdotu_batched(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* const x[],
                             rocblas_int                   incx,
                             const rocblas_double_complex* const y[],
                             rocblas_int                   incy,
                            rocblas_int    batch_count,
                             rocblas_double_complex*       result)
{
    return rocblas_dot_batched<false>(handle, n, x, incx, y, incy, batch_count, result);
}

rocblas_status rocblas_cdotc_batched(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* const x[],
                             rocblas_int                  incx,
                             const rocblas_float_complex* const y[],
                             rocblas_int                  incy,
                            rocblas_int    batch_count,
                             rocblas_float_complex*       result)
{
    return rocblas_dot_batched<true>(handle, n, x, incx, y, incy, batch_count, result);
}

rocblas_status rocblas_zdotc_batched(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* const x[],
                             rocblas_int                   incx,
                             const rocblas_double_complex*  const y[],
                             rocblas_int                   incy,
                            rocblas_int    batch_count,
                             rocblas_double_complex*       result)
{
    return rocblas_dot_batched<true>(handle, n, x, incx, y, incy, batch_count, result);
}

} // extern "C"
