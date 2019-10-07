/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_rot.hpp"
#include "utility.h"

namespace
{
    constexpr int NB = 512;

    template <typename T, typename = T>
    constexpr char rocblas_rot_name[] = "unknown";
    template <>
    constexpr char rocblas_rot_name<float>[] = "rocblas_srot_batched";
    template <>
    constexpr char rocblas_rot_name<double>[] = "rocblas_drot";
    template <>
    constexpr char rocblas_rot_name<rocblas_float_complex>[] = "rocblas_crot_batched";
    template <>
    constexpr char rocblas_rot_name<rocblas_double_complex>[] = "rocblas_zrot_batched";
    template <>
    constexpr char rocblas_rot_name<rocblas_float_complex, float>[] = "rocblas_csrot_batched";
    template <>
    constexpr char rocblas_rot_name<rocblas_double_complex, double>[] = "rocblas_zdrot_batched";

    template <class T, class U, class V>
    rocblas_status rocblas_rot_batched_impl(rocblas_handle handle,
                                            rocblas_int    n,
                                            T* const       x[],
                                            rocblas_int    incx,
                                            T* const       y[],
                                            rocblas_int    incy,
                                            const U*       c,
                                            const V*       s,
                                            rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rot_name<T, V>, n, x, incx, y, incy, c, s, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rot_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "--c_type",
                      rocblas_precision_string<V>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_rot_name<T, V>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch",
                        batch_count);

        if(!x || !y || !c || !s)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rot_template<NB, T>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, c, 0, s, 0, batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_srot_batched(rocblas_handle handle,
                                    rocblas_int    n,
                                    float* const   x[],
                                    rocblas_int    incx,
                                    float* const   y[],
                                    rocblas_int    incy,
                                    const float*   c,
                                    const float*   s,
                                    rocblas_int    batch_count)
{
    return rocblas_rot_batched_impl(handle, n, x, incx, y, incy, c, s, batch_count);
}

rocblas_status rocblas_drot_batched(rocblas_handle handle,
                                    rocblas_int    n,
                                    double* const  x[],
                                    rocblas_int    incx,
                                    double* const  y[],
                                    rocblas_int    incy,
                                    const double*  c,
                                    const double*  s,
                                    rocblas_int    batch_count)
{
    return rocblas_rot_batched_impl(handle, n, x, incx, y, incy, c, s, batch_count);
}

rocblas_status rocblas_crot_batched(rocblas_handle               handle,
                                    rocblas_int                  n,
                                    rocblas_float_complex* const x[],
                                    rocblas_int                  incx,
                                    rocblas_float_complex* const y[],
                                    rocblas_int                  incy,
                                    const float*                 c,
                                    const rocblas_float_complex* s,
                                    rocblas_int                  batch_count)
{
    return rocblas_rot_batched_impl(handle, n, x, incx, y, incy, c, s, batch_count);
}

rocblas_status rocblas_csrot_batched(rocblas_handle               handle,
                                     rocblas_int                  n,
                                     rocblas_float_complex* const x[],
                                     rocblas_int                  incx,
                                     rocblas_float_complex* const y[],
                                     rocblas_int                  incy,
                                     const float*                 c,
                                     const float*                 s,
                                     rocblas_int                  batch_count)
{
    return rocblas_rot_batched_impl(handle, n, x, incx, y, incy, c, s, batch_count);
}

rocblas_status rocblas_zrot_batched(rocblas_handle                handle,
                                    rocblas_int                   n,
                                    rocblas_double_complex* const x[],
                                    rocblas_int                   incx,
                                    rocblas_double_complex* const y[],
                                    rocblas_int                   incy,
                                    const double*                 c,
                                    const rocblas_double_complex* s,
                                    rocblas_int                   batch_count)
{
    return rocblas_rot_batched_impl(handle, n, x, incx, y, incy, c, s, batch_count);
}

rocblas_status rocblas_zdrot_batched(rocblas_handle                handle,
                                     rocblas_int                   n,
                                     rocblas_double_complex* const x[],
                                     rocblas_int                   incx,
                                     rocblas_double_complex* const y[],
                                     rocblas_int                   incy,
                                     const double*                 c,
                                     const double*                 s,
                                     rocblas_int                   batch_count)
{
    return rocblas_rot_batched_impl(handle, n, x, incx, y, incy, c, s, batch_count);
}

} // extern "C"
