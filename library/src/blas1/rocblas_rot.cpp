/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    constexpr int NB = 512;

    template <typename T,
              typename U,
              typename V,
              typename std::enable_if<!is_complex<V>, int>::type = 0>
    __global__ void rot_kernel(rocblas_int n,
                               T*          x,
                               rocblas_int incx,
                               T*          y,
                               rocblas_int incy,
                               U           c_device_host,
                               V           s_device_host)
    {
        auto      c   = load_scalar(c_device_host);
        auto      s   = load_scalar(s_device_host);
        ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(tid < n)
        {
            auto ix   = tid * incx;
            auto iy   = tid * incy;
            auto temp = c * x[ix] + s * y[iy];
            y[iy]     = c * y[iy] - s * x[ix];
            x[ix]     = temp;
        }
    }

    template <typename T,
              typename U,
              typename V,
              typename std::enable_if<is_complex<V>, int>::type = 0>
    __global__ void rot_kernel(rocblas_int n,
                               T*          x,
                               rocblas_int incx,
                               T*          y,
                               rocblas_int incy,
                               U           c_device_host,
                               V           s_device_host)
    {
        auto      c   = load_scalar(c_device_host);
        auto      s   = load_scalar(s_device_host);
        ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(tid < n)
        {
            auto ix   = tid * incx;
            auto iy   = tid * incy;
            auto temp = c * x[ix] + s * y[iy];
            y[iy]     = c * y[iy] - conj(s) * x[ix];
            x[ix]     = temp;
        }
    }

    template <typename T, typename = T>
    constexpr char rocblas_rot_name[] = "unknown";
    template <>
    constexpr char rocblas_rot_name<float>[] = "rocblas_srot";
    template <>
    constexpr char rocblas_rot_name<double>[] = "rocblas_drot";
    template <>
    constexpr char rocblas_rot_name<rocblas_float_complex>[] = "rocblas_crot";
    template <>
    constexpr char rocblas_rot_name<rocblas_double_complex>[] = "rocblas_zrot";
    template <>
    constexpr char rocblas_rot_name<rocblas_float_complex, float>[] = "rocblas_csrot";
    template <>
    constexpr char rocblas_rot_name<rocblas_double_complex, double>[] = "rocblas_zdrot";

    template <class T, class U, class V>
    rocblas_status rocblas_rot(rocblas_handle handle,
                               rocblas_int    n,
                               T*             x,
                               rocblas_int    incx,
                               T*             y,
                               rocblas_int    incy,
                               const U*       c,
                               const V*       s)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rot_name<T, V>, n, x, incx, y, incy, c, s);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rot -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rot_name<T, V>, "N", n, "incx", incx, "incy", incy);

        if(!x || !y || !c || !s)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Quick return if possible
        if(n <= 0 || incx <= 0 || incy <= 0)
            return rocblas_status_success;

        dim3        blocks((n - 1) / NB + 1);
        dim3        threads(NB);
        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL(
                rot_kernel, blocks, threads, 0, rocblas_stream, n, x, incx, y, incy, c, s);
        else // c and s are on host
            hipLaunchKernelGGL(
                rot_kernel, blocks, threads, 0, rocblas_stream, n, x, incx, y, incy, *c, *s);

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_srot(rocblas_handle handle,
                            rocblas_int    n,
                            float*         x,
                            rocblas_int    incx,
                            float*         y,
                            rocblas_int    incy,
                            const float*   c,
                            const float*   s)
{
    return rocblas_rot(handle, n, x, incx, y, incy, c, s);
}

rocblas_status rocblas_drot(rocblas_handle handle,
                            rocblas_int    n,
                            double*        x,
                            rocblas_int    incx,
                            double*        y,
                            rocblas_int    incy,
                            const double*  c,
                            const double*  s)
{
    return rocblas_rot(handle, n, x, incx, y, incy, c, s);
}

rocblas_status rocblas_crot(rocblas_handle               handle,
                            rocblas_int                  n,
                            rocblas_float_complex*       x,
                            rocblas_int                  incx,
                            rocblas_float_complex*       y,
                            rocblas_int                  incy,
                            const float*                 c,
                            const rocblas_float_complex* s)
{
    return rocblas_rot(handle, n, x, incx, y, incy, c, s);
}

rocblas_status rocblas_csrot(rocblas_handle         handle,
                             rocblas_int            n,
                             rocblas_float_complex* x,
                             rocblas_int            incx,
                             rocblas_float_complex* y,
                             rocblas_int            incy,
                             const float*           c,
                             const float*           s)
{
    return rocblas_rot(handle, n, x, incx, y, incy, c, s);
}

rocblas_status rocblas_zrot(rocblas_handle                handle,
                            rocblas_int                   n,
                            rocblas_double_complex*       x,
                            rocblas_int                   incx,
                            rocblas_double_complex*       y,
                            rocblas_int                   incy,
                            const double*                 c,
                            const rocblas_double_complex* s)
{
    return rocblas_rot(handle, n, x, incx, y, incy, c, s);
}

rocblas_status rocblas_zdrot(rocblas_handle          handle,
                             rocblas_int             n,
                             rocblas_double_complex* x,
                             rocblas_int             incx,
                             rocblas_double_complex* y,
                             rocblas_int             incy,
                             const double*           c,
                             const double*           s)
{
    return rocblas_rot(handle, n, x, incx, y, incy, c, s);
}

} // extern "C"
