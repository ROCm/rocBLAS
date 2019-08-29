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

    template <typename T, typename U>
    __global__ void rotm_kernel(rocblas_int n,
                                T*          x,
                                rocblas_int incx,
                                T*          y,
                                rocblas_int incy,
                                U           flag_device_host,
                                U           h11_device_host,
                                U           h21_device_host,
                                U           h12_device_host,
                                U           h22_device_host)
    {
        auto      flag = load_scalar(flag_device_host);
        auto      h11  = load_scalar(h11_device_host);
        auto      h21  = load_scalar(h21_device_host);
        auto      h12  = load_scalar(h12_device_host);
        auto      h22  = load_scalar(h22_device_host);
        ptrdiff_t tid  = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(tid < n && flag != -2)
        {
            auto ix = tid * incx;
            auto iy = tid * incy;
            auto w  = x[ix];
            auto z  = y[iy];
            if(flag < 0)
            {
                x[ix] = w * h11 + z * h12;
                y[iy] = w * h21 + z * h22;
            }
            else if(flag == 0)
            {
                x[ix] = w + z * h12;
                y[iy] = w * h21 + z;
            }
            else
            {
                x[ix] = w * h11 + z;
                y[iy] = -w + z * h22;
            }
        }
    }

    template <typename>
    constexpr char rocblas_rotm_name[] = "unknown";
    template <>
    constexpr char rocblas_rotm_name<float>[] = "rocblas_srotm";
    template <>
    constexpr char rocblas_rotm_name<double>[] = "rocblas_drotm";

    template <class T>
    rocblas_status rocblas_rotm(rocblas_handle handle,
                                rocblas_int    n,
                                T*             x,
                                rocblas_int    incx,
                                T*             y,
                                rocblas_int    incy,
                                const T*       param)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotm_name<T>, n, x, incx, y, incy, param);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotm -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotm_name<T>, "N", n, "incx", incx, "incy", incy);

        if(!x || !y || !param)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Quick return if possible
        if(n <= 0 || incx <= 0 || incy <= 0)
            return rocblas_status_success;
        if(rocblas_pointer_mode_host == handle->pointer_mode && param[0] == -2)
            return rocblas_status_success;

        dim3        blocks((n - 1) / NB + 1);
        dim3        threads(NB);
        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            hipLaunchKernelGGL(rotm_kernel,
                               blocks,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               x,
                               incx,
                               y,
                               incy,
                               param,
                               param + 1,
                               param + 2,
                               param + 3,
                               param + 4);
        else // c and s are on host
            hipLaunchKernelGGL(rotm_kernel,
                               blocks,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               x,
                               incx,
                               y,
                               incy,
                               param[0],
                               param[1],
                               param[2],
                               param[3],
                               param[4]);

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotm(rocblas_handle handle,
                                            rocblas_int    n,
                                            float*         x,
                                            rocblas_int    incx,
                                            float*         y,
                                            rocblas_int    incy,
                                            const float*   param)
{
    return rocblas_rotm(handle, n, x, incx, y, incy, param);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotm(rocblas_handle handle,
                                            rocblas_int    n,
                                            double*        x,
                                            rocblas_int    incx,
                                            double*        y,
                                            rocblas_int    incy,
                                            const double*  param)
{
    return rocblas_rotm(handle, n, x, incx, y, incy, param);
}

} // extern "C"
