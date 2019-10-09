/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T>
__device__ __host__ void rocblas_rotmg_calc(T& d1, T& d2, T& x1, const T& y1, T* param)
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

template <typename T, typename U>
__global__ void rocblas_rotmg_kernel(T              d1_in,
                                     rocblas_int    offset_d1,
                                     rocblas_stride stride_d1,
                                     T              d2_in,
                                     rocblas_int    offset_d2,
                                     rocblas_stride stride_d2,
                                     T              x1_in,
                                     rocblas_int    offset_x1,
                                     rocblas_stride stride_x1,
                                     U              y1_in,
                                     rocblas_int    offset_y1,
                                     rocblas_stride stride_y1,
                                     T              param,
                                     rocblas_int    offset_param,
                                     rocblas_stride stride_param,
                                     rocblas_int    batch_count)
{
    auto d1 = load_ptr_batch(d1_in, hipBlockIdx_x, offset_d1, stride_d1);
    auto d2 = load_ptr_batch(d2_in, hipBlockIdx_x, offset_d2, stride_d2);
    auto x1 = load_ptr_batch(x1_in, hipBlockIdx_x, offset_x1, stride_x1);
    auto y1 = load_ptr_batch(y1_in, hipBlockIdx_x, offset_y1, stride_y1);
    auto p  = load_ptr_batch(param, hipBlockIdx_x, offset_param, stride_param);
    rocblas_rotmg_calc(*d1, *d2, *x1, *y1, p);
}

template <typename T, typename U>
rocblas_status rocblas_rotmg_template(rocblas_handle handle,
                                      T              d1_in,
                                      rocblas_int    offset_d1,
                                      rocblas_stride stride_d1,
                                      T              d2_in,
                                      rocblas_int    offset_d2,
                                      rocblas_stride stride_d2,
                                      T              x1_in,
                                      rocblas_int    offset_x1,
                                      rocblas_stride stride_x1,
                                      U              y1_in,
                                      rocblas_int    offset_y1,
                                      rocblas_stride stride_y1,
                                      T              param,
                                      rocblas_int    offset_param,
                                      rocblas_stride stride_param,
                                      rocblas_int    batch_count)
{
    if(!batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;
    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL(rocblas_rotmg_kernel,
                           batch_count,
                           1,
                           0,
                           rocblas_stream,
                           d1_in,
                           offset_d1,
                           stride_d1,
                           d2_in,
                           offset_d2,
                           stride_d2,
                           x1_in,
                           offset_x1,
                           stride_x1,
                           y1_in,
                           offset_y1,
                           stride_y1,
                           param,
                           offset_param,
                           stride_param,
                           batch_count);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));
        // TODO: make this faster for a large number of batches.
        for(int i = 0; i < batch_count; i++)
        {
            auto d1 = load_ptr_batch(d1_in, i, offset_d1, stride_d1);
            auto d2 = load_ptr_batch(d2_in, i, offset_d2, stride_d2);
            auto x1 = load_ptr_batch(x1_in, i, offset_x1, stride_x1);
            auto y1 = load_ptr_batch(y1_in, i, offset_y1, stride_y1);
            auto p  = load_ptr_batch(param, i, offset_param, stride_param);

            rocblas_rotmg_calc(*d1, *d2, *x1, *y1, p);
        }
    }

    return rocblas_status_success;
}