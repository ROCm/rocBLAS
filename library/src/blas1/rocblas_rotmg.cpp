/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename T>
    __device__ __host__ void rotmg_calc(T& d1, T& d2, T& x1, const T& y1, T* param)
    {
        const T gam    = 4096;
        const T gamsq  = gam * gam;
        const T rgamsq = 1 / gamsq;

        T flag = -1;
        T h11 = 0, h21 = 0, h12 = 0, h22 = 0;

        if(d1 < 0)
        {
            d1 = d2 = x1 = 0;
        }
        else
        {
            T p2 = d2 * y1;
            if(p2 == 0)
            {
                flag     = -2;
                param[0] = flag;
                return;
            }
            T p1 = d1 * x1;
            T q2 = p2 * y1;
            T q1 = p1 * x1;
            if(rocblas_abs(q1) > rocblas_abs(q2))
            {
                h21 = -y1 / x1;
                h12 = p2 / p1;
                T u = 1 - h12 * h21;
                if(u > 0)
                {
                    flag = 0;
                    d1 /= u;
                    d2 /= u;
                    x1 *= u;
                }
            }
            else
            {
                if(q2 < 0)
                {
                    d1 = d2 = x1 = 0;
                }
                else
                {
                    flag   = 1;
                    h11    = p1 / p2;
                    h22    = x1 / y1;
                    T u    = 1 + h11 * h22;
                    T temp = d2 / u;
                    d2     = d1 / u;
                    d1     = temp;
                    x1     = y1 * u;
                }
            }

            if(d1 != 0)
            {
                while((d1 <= rgamsq) || (d1 >= gamsq))
                {
                    if(flag == 0)
                    {
                        h11 = h22 = 1;
                        flag      = -1;
                    }
                    else
                    {
                        h21  = -1;
                        h12  = 1;
                        flag = -1;
                    }
                    if(d1 <= rgamsq)
                    {
                        d1 *= gamsq;
                        x1 /= gam;
                        h11 /= gam;
                        h12 /= gam;
                    }
                    else
                    {
                        d1 /= gamsq;
                        x1 *= gam;
                        h11 *= gam;
                        h12 *= gam;
                    }
                }
            }

            if(d2 != 0)
            {
                while((rocblas_abs(d2) <= rgamsq) || (rocblas_abs(d2) >= gamsq))
                {
                    if(flag == 0)
                    {
                        h11 = h22 = 1;
                        flag      = -1;
                    }
                    else
                    {
                        h21  = -1;
                        h12  = 1;
                        flag = -1;
                    }
                    if(rocblas_abs(d2) <= rgamsq)
                    {
                        d2 *= gamsq;
                        h21 /= gam;
                        h22 /= gam;
                    }
                    else
                    {
                        d2 /= gamsq;
                        h21 *= gam;
                        h22 *= gam;
                    }
                }
            }
        }

        if(flag < 0)
        {
            param[1] = h11;
            param[2] = h21;
            param[3] = h12;
            param[4] = h22;
        }
        else if(flag == 0)
        {
            param[2] = h21;
            param[3] = h12;
        }
        else
        {
            param[1] = h11;
            param[4] = h22;
        }
        param[0] = flag;
    }

    template <typename T>
    __global__ void rotmg_kernel(T* d1, T* d2, T* x1, const T* y1, T* param)
    {
        rotmg_calc(*d1, *d2, *x1, *y1, param);
    }

    template <typename>
    constexpr char rocblas_rotmg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotmg_name<float>[] = "rocblas_srotmg";
    template <>
    constexpr char rocblas_rotmg_name<double>[] = "rocblas_drotmg";

    template <class T>
    rocblas_status rocblas_rotmg(rocblas_handle handle, T* d1, T* d2, T* x1, const T* y1, T* param)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotmg_name<T>, d1, d2, x1, y1, param);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_trace(handle, "./rocblas-bench -f rotmg -r", rocblas_precision_string<T>);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotmg_name<T>);

        if(!d1 || !d2 || !x1 || !y1 || !param)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            hipLaunchKernelGGL(rotmg_kernel, 1, 1, 0, rocblas_stream, d1, d2, x1, y1, param);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));
            rotmg_calc(*d1, *d2, *x1, *y1, param);
        }

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotmg(
    rocblas_handle handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    return rocblas_rotmg(handle, d1, d2, x1, y1, param);
}

ROCBLAS_EXPORT rocblas_status rocblas_drotmg(
    rocblas_handle handle, double* d1, double* d2, double* x1, const double* y1, double* param)
{
    return rocblas_rotmg(handle, d1, d2, x1, y1, param);
}

} // extern "C"
