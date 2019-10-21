/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_scal.hpp"
#include "utility.h"

namespace
{

    template <typename T, typename = T>
    constexpr char rocblas_scal_name[] = "unknown";
    template <>
    constexpr char rocblas_scal_name<float>[] = "rocblas_sscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<double>[] = "rocblas_dscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex>[] = "rocblas_cscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex>[] = "rocblas_zscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex, float>[]
        = "rocblas_csscal_strided_batched";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex, double>[]
        = "rocblas_zdscal_strided_batched";

    template <rocblas_int NB, typename T, typename U>
    rocblas_status rocblas_scal_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const U*       alpha,
                                                     T*             x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        if(!alpha)
            return rocblas_status_invalid_pointer;

        auto layer_mode = handle->layer_mode;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(
                    handle, rocblas_scal_name<T, U>, n, *alpha, x, incx, stridex, batch_count);

            // there are an extra 2 scal functions, thus
            // the -r mode will not work correctly. Substitute
            // with --a_type and --b_type (?)
            // ANSWER: -r is syntatic sugar; the types can be specified separately
            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                std::stringstream alphass;
                alphass << "--alpha " << std::real(*alpha)
                        << (std::imag(*alpha) != 0
                                ? (" --alphai " + std::to_string(std::imag(*alpha)))
                                : "");
                log_bench(handle,
                          "./rocblas-bench -f scal_strided_batched --a_type",
                          rocblas_precision_string<T>,
                          "--b_type",
                          rocblas_precision_string<U>,
                          "-n",
                          n,
                          "--incx",
                          incx,
                          "--stride_x",
                          stridex,
                          alphass.str(),
                          "--batch",
                          batch_count);
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_scal_name<T, U>, n, alpha, x, incx, stridex, batch_count);
        }
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_scal_name<T, U>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stride_x",
                        stridex,
                        "batch",
                        batch_count);

        if(!x)
            return rocblas_status_invalid_pointer;

        if(batch_count < 0)
            return rocblas_status_invalid_size;

        return rocblas_scal_template<NB, T>(handle, n, alpha, 0, x, 0, incx, stridex, batch_count);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sscal_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   alpha,
                                             float*         x,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             rocblas_int    batch_count)
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_dscal_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  alpha,
                                             double*        x,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             rocblas_int    batch_count)
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_cscal_strided_batched(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             rocblas_float_complex*       x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stridex,
                                             rocblas_int                  batch_count)
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_zscal_strided_batched(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             rocblas_double_complex*       x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stridex,
                                             rocblas_int                   batch_count)
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}

// Scal with a real alpha & complex vector
rocblas_status rocblas_csscal_strided_batched(rocblas_handle         handle,
                                              rocblas_int            n,
                                              const float*           alpha,
                                              rocblas_float_complex* x,
                                              rocblas_int            incx,
                                              rocblas_stride         stridex,
                                              rocblas_int            batch_count)
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_zdscal_strided_batched(rocblas_handle          handle,
                                              rocblas_int             n,
                                              const double*           alpha,
                                              rocblas_double_complex* x,
                                              rocblas_int             incx,
                                              rocblas_stride          stridex,
                                              rocblas_int             batch_count)
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_strided_batched_impl<NB>(handle, n, alpha, x, incx, stridex, batch_count);
}

} // extern "C"
