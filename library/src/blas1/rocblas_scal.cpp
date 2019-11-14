/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    constexpr int NB = 256;

    template <typename T, typename U>
    __global__ void scal_kernel(rocblas_int n, U alpha_device_host, T* x, rocblas_int incx)
    {
        auto      alpha = load_scalar(alpha_device_host);
        ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        // bound
        if(tid < n)
            x[tid * incx] *= alpha;
    }

    template <typename T, typename = T>
    constexpr char rocblas_scal_name[] = "unknown";
    template <>
    constexpr char rocblas_scal_name<float>[] = "rocblas_sscal";
    template <>
    constexpr char rocblas_scal_name<double>[] = "rocblas_dscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex>[] = "rocblas_cscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex>[] = "rocblas_zscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_float_complex, float>[] = "rocblas_csscal";
    template <>
    constexpr char rocblas_scal_name<rocblas_double_complex, double>[] = "rocblas_zdscal";

    template <typename T, typename U>
    rocblas_status
        rocblas_scal(rocblas_handle handle, rocblas_int n, const U* alpha, T* x, rocblas_int incx)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        if(!alpha)
            return rocblas_status_invalid_pointer;

        auto layer_mode = handle->layer_mode;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_scal_name<T, U>, n, *alpha, x, incx);

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
                          "./rocblas-bench -f scal --a_type",
                          rocblas_precision_string<T>,
                          "--b_type",
                          rocblas_precision_string<U>,
                          "-n",
                          n,
                          "--incx",
                          incx,
                          alphass.str());
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_scal_name<T, U>, n, alpha, x, incx);
        }
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_scal_name<T, U>, "N", n, "incx", incx);

        if(!x)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Quick return if possible. Not Argument error
        if(n <= 0 || incx <= 0)
            return rocblas_status_success;

        rocblas_int blocks = (n - 1) / NB + 1;
        dim3        threads(NB);
        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL(scal_kernel, blocks, threads, 0, rocblas_stream, n, alpha, x, incx);
        else // alpha is on host
            hipLaunchKernelGGL(scal_kernel, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx);

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sscal(
    rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

rocblas_status rocblas_cscal(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             rocblas_float_complex*       x,
                             rocblas_int                  incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

rocblas_status rocblas_zscal(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             rocblas_double_complex*       x,
                             rocblas_int                   incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

// Scal with a real alpha & complex vector
rocblas_status rocblas_csscal(rocblas_handle         handle,
                              rocblas_int            n,
                              const float*           alpha,
                              rocblas_float_complex* x,
                              rocblas_int            incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

rocblas_status rocblas_zdscal(rocblas_handle          handle,
                              rocblas_int             n,
                              const double*           alpha,
                              rocblas_double_complex* x,
                              rocblas_int             incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

} // extern "C"
