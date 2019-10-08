/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
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
    constexpr char rocblas_dot_strided_batched_name[] = "unknown";
    template <bool CONJ>
    constexpr char rocblas_dot_strided_batched_name<CONJ, float>[] = "rocblas_sdot_strided_batched";
    template <bool CONJ>
    constexpr char rocblas_dot_strided_batched_name<CONJ, double>[]
        = "rocblas_ddot_strided_batched";
    template <>
    constexpr char rocblas_dot_strided_batched_name<true, rocblas_float_complex>[]
        = "rocblas_cdotc_strided_batched";
    template <>
    constexpr char rocblas_dot_strided_batched_name<false, rocblas_float_complex>[]
        = "rocblas_cdot_ustrided_batched";
    template <>
    constexpr char rocblas_dot_strided_batched_name<true, rocblas_double_complex>[]
        = "rocblas_zdotc_strided_batched";
    template <>
    constexpr char rocblas_dot_strided_batched_name<false, rocblas_double_complex>[]
        = "rocblas_zdotu_strided_batched";

    // allocate workspace inside this API
    template <bool CONJ, typename T, typename T2 = T>
    rocblas_status rocblas_dot_strided_batched_impl(rocblas_handle handle,
                                                    rocblas_int    n,
                                                    const T*       x,
                                                    rocblas_int    incx,
                                                    rocblas_stride stridex,
                                                    const T*       y,
                                                    rocblas_int    incy,
                                                    rocblas_stride stridey,
                                                    rocblas_int    batch_count,
                                                    T*             results)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_dot_strided_batched_name<CONJ, T>,
                      n,
                      x,
                      incx,
                      stridex,
                      y,
                      incy,
                      stridey,
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f dot_strided_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--stridex",
                      stridex,
                      "--incy",
                      incy,
                      "--stridey",
                      stridey,
                      "--batch",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_dot_strided_batched_name<CONJ, T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stridex",
                        stridex,
                        "incy",
                        incy,
                        "stridey",
                        stridey,
                        "batch_count",
                        batch_count);

        if(!x || !y || !results)
            return rocblas_status_invalid_pointer;

        if(batch_count < 0)
            return rocblas_status_invalid_size;

        size_t dev_bytes
            = rocblas_reduction_kernel_workspace_size<NB>(n, batch_count, (T2*)results);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_dot_template<NB, CONJ, T>(
            handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count, results, (T2*)mem);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sdot_strided_batched(rocblas_handle handle,
                                            rocblas_int    n,
                                            const float*   x,
                                            rocblas_int    incx,
                                            rocblas_stride stridex,
                                            const float*   y,
                                            rocblas_int    incy,
                                            rocblas_stride stridey,
                                            rocblas_int    batch_count,
                                            float*         results)
{
    return rocblas_dot_strided_batched_impl<false>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, results);
}

rocblas_status rocblas_ddot_strided_batched(rocblas_handle handle,
                                            rocblas_int    n,
                                            const double*  x,
                                            rocblas_int    incx,
                                            rocblas_stride stridex,
                                            const double*  y,
                                            rocblas_int    incy,
                                            rocblas_stride stridey,
                                            rocblas_int    batch_count,
                                            double*        results)
{
    return rocblas_dot_strided_batched_impl<false>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, results);
}

rocblas_status rocblas_hdot_strided_batched(rocblas_handle      handle,
                                            rocblas_int         n,
                                            const rocblas_half* x,
                                            rocblas_int         incx,
                                            rocblas_stride      stridex,
                                            const rocblas_half* y,
                                            rocblas_int         incy,
                                            rocblas_stride      stridey,
                                            rocblas_int         batch_count,
                                            rocblas_half*       result)
{
    return rocblas_dot_strided_batched_impl<false>(handle,
                                                   n,
                                                   (const _Float16*)x,
                                                   incx,
                                                   stridex,
                                                   (const _Float16*)y,
                                                   incy,
                                                   stridey,
                                                   batch_count,
                                                   (_Float16*)result);
}

rocblas_status rocblas_bfdot_strided_batched(rocblas_handle          handle,
                                             rocblas_int             n,
                                             const rocblas_bfloat16* x,
                                             rocblas_int             incx,
                                             rocblas_stride          stridex,
                                             const rocblas_bfloat16* y,
                                             rocblas_int             incy,
                                             rocblas_stride          stridey,
                                             rocblas_int             batch_count,
                                             rocblas_bfloat16*       result)
{
    return rocblas_dot_strided_batched_impl<false, rocblas_bfloat16, float>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

rocblas_status rocblas_cdotu_strided_batched(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             const rocblas_float_complex* y,
                                             rocblas_int                  incy,
                                             rocblas_stride               stridey,
                                             rocblas_int                  batch_count,
                                             rocblas_float_complex*       results)
{
    return rocblas_dot_strided_batched_impl<false>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, results);
}

rocblas_status rocblas_zdotu_strided_batched(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             const rocblas_double_complex* y,
                                             rocblas_int                   incy,
                                             rocblas_stride                stridey,
                                             rocblas_int                   batch_count,
                                             rocblas_double_complex*       results)
{
    return rocblas_dot_strided_batched_impl<false>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, results);
}

rocblas_status rocblas_cdotc_strided_batched(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             const rocblas_float_complex* y,
                                             rocblas_int                  incy,
                                             rocblas_stride               stridey,
                                             rocblas_int                  batch_count,
                                             rocblas_float_complex*       results)
{
    return rocblas_dot_strided_batched_impl<true>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, results);
}

rocblas_status rocblas_zdotc_strided_batched(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             const rocblas_double_complex* y,
                                             rocblas_int                   incy,
                                             rocblas_stride                stridey,
                                             rocblas_int                   batch_count,
                                             rocblas_double_complex*       results)
{
    return rocblas_dot_strided_batched_impl<true>(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, results);
}

} // extern "C"
