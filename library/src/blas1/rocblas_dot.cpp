/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_dot.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas/rocblas.h"
#include "utility.hpp"

namespace
{
    // HIP support up to 1024 threads/work itemes per thread block/work group
    // setting to 512 for gfx803.
    constexpr int NB = 512;

    template <bool, typename>
    constexpr char rocblas_dot_name[] = "unknown";
    template <bool CONJ>
    constexpr char rocblas_dot_name<CONJ, float>[] = "rocblas_sdot";
    template <bool CONJ>
    constexpr char rocblas_dot_name<CONJ, double>[] = "rocblas_ddot";
    template <bool CONJ>
    constexpr char rocblas_dot_name<CONJ, rocblas_half>[] = "rocblas_hdot";
    template <bool CONJ>
    constexpr char rocblas_dot_name<CONJ, rocblas_bfloat16>[] = "rocblas_bfdot";
    template <>
    constexpr char rocblas_dot_name<true, rocblas_float_complex>[] = "rocblas_cdotc";
    template <>
    constexpr char rocblas_dot_name<false, rocblas_float_complex>[] = "rocblas_cdotu";
    template <>
    constexpr char rocblas_dot_name<true, rocblas_double_complex>[] = "rocblas_zdotc";
    template <>
    constexpr char rocblas_dot_name<false, rocblas_double_complex>[] = "rocblas_zdotu";

    // allocate workspace inside this API
    template <bool CONJ, typename T, typename T2 = T>
    inline rocblas_status rocblas_dot_impl(rocblas_handle handle,
                                           rocblas_int    n,
                                           const T*       x,
                                           rocblas_int    incx,
                                           const T*       y,
                                           rocblas_int    incy,
                                           T*             result)
    {
        static constexpr int WIN = rocblas_dot_WIN<T>();

        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB * WIN, T2>(n);
        if(handle->is_device_memory_size_query())
        {
            if(n <= 0)
                return rocblas_status_size_unchanged;
            else
                return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_dot_name<CONJ, T>, n, x, incx, y, incy);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f dot -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_dot_name<CONJ, T>, "N", n, "incx", incx, "incy", incy);

        // Quick return if possible.
        if(n <= 0)
        {
            if(!result)
                return rocblas_status_invalid_pointer;
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(result, 0, sizeof(*result), handle->get_stream()));
            else
                *result = T(0);
            return rocblas_status_success;
        }

        if(!x || !y || !result)
            return rocblas_status_invalid_pointer;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status dot_check_numerics_status
                = rocblas_dot_check_numerics(rocblas_dot_name<CONJ, T>,
                                             handle,
                                             n,
                                             x,
                                             0,
                                             incx,
                                             0,
                                             y,
                                             0,
                                             incy,
                                             0,
                                             1,
                                             check_numerics,
                                             is_input);
            if(dot_check_numerics_status != rocblas_status_success)
                return dot_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_dot_template<NB, CONJ, T>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, 1, result, (T2*)w_mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status dot_check_numerics_status
                = rocblas_dot_check_numerics(rocblas_dot_name<CONJ, T>,
                                             handle,
                                             n,
                                             x,
                                             0,
                                             incx,
                                             0,
                                             y,
                                             0,
                                             incy,
                                             0,
                                             1,
                                             check_numerics,
                                             is_input);
            if(dot_check_numerics_status != rocblas_status_success)
                return dot_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sdot(rocblas_handle handle,
                            rocblas_int    n,
                            const float*   x,
                            rocblas_int    incx,
                            const float*   y,
                            rocblas_int    incy,
                            float*         result)
try
{
    return rocblas_dot_impl<false>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ddot(rocblas_handle handle,
                            rocblas_int    n,
                            const double*  x,
                            rocblas_int    incx,
                            const double*  y,
                            rocblas_int    incy,
                            double*        result)
try
{
    return rocblas_dot_impl<false>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_hdot(rocblas_handle      handle,
                            rocblas_int         n,
                            const rocblas_half* x,
                            rocblas_int         incx,
                            const rocblas_half* y,
                            rocblas_int         incy,
                            rocblas_half*       result)
try
{
    return rocblas_dot_impl<false>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_bfdot(rocblas_handle          handle,
                             rocblas_int             n,
                             const rocblas_bfloat16* x,
                             rocblas_int             incx,
                             const rocblas_bfloat16* y,
                             rocblas_int             incy,
                             rocblas_bfloat16*       result)
try
{
    return rocblas_dot_impl<false, rocblas_bfloat16, float>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cdotu(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* y,
                             rocblas_int                  incy,
                             rocblas_float_complex*       result)
try
{
    return rocblas_dot_impl<false>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zdotu(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* y,
                             rocblas_int                   incy,
                             rocblas_double_complex*       result)
try
{
    return rocblas_dot_impl<false>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cdotc(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* y,
                             rocblas_int                  incy,
                             rocblas_float_complex*       result)
try
{
    return rocblas_dot_impl<true>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zdotc(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* y,
                             rocblas_int                   incy,
                             rocblas_double_complex*       result)
try
{
    return rocblas_dot_impl<true>(handle, n, x, incx, y, incy, result);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
