/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U>
__global__ void rotm_kernel(rocblas_int    n,
                            T              x_in,
                            rocblas_int    offset_x,
                            rocblas_int    incx,
                            rocblas_stride stride_x,
                            T              y_in,
                            rocblas_int    offset_y,
                            rocblas_int    incy,
                            rocblas_stride stride_y,
                            U              flag_device_host,
                            U              h11_device_host,
                            U              h21_device_host,
                            U              h12_device_host,
                            U              h22_device_host,
                            rocblas_stride stride_param)
{
    auto      flag = load_scalar(flag_device_host, hipBlockIdx_y, stride_param);
    auto      h11  = load_scalar(h11_device_host, hipBlockIdx_y, stride_param);
    auto      h21  = load_scalar(h21_device_host, hipBlockIdx_y, stride_param);
    auto      h12  = load_scalar(h12_device_host, hipBlockIdx_y, stride_param);
    auto      h22  = load_scalar(h22_device_host, hipBlockIdx_y, stride_param);
    auto      x    = load_ptr_batch(x_in, hipBlockIdx_y, offset_x, stride_x);
    auto      y    = load_ptr_batch(y_in, hipBlockIdx_y, offset_y, stride_y);
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

template <rocblas_int NB, typename T, typename U>
rocblas_status rocblas_rotm_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     U              x,
                                     rocblas_int    offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     U              y,
                                     rocblas_int    offset_y,
                                     rocblas_int    incy,
                                     rocblas_stride stride_y,
                                     const T*       param,
                                     rocblas_stride stride_param,
                                     rocblas_int    batch_count)
{
    // Quick return if possible
    if(n <= 0 || incx <= 0 || incy <= 0 || batch_count == 0)
        return rocblas_status_success;
    if(rocblas_pointer_mode_host == handle->pointer_mode && param[0] == -2)
        return rocblas_status_success;

    dim3        blocks((n - 1) / NB + 1, batch_count);
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
                           offset_x,
                           incx,
                           stride_x,
                           y,
                           offset_y,
                           incy,
                           stride_y,
                           param,
                           param + 1,
                           param + 2,
                           param + 3,
                           param + 4,
                           stride_param);
    else // c and s are on host
        hipLaunchKernelGGL(rotm_kernel,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           x,
                           offset_x,
                           incx,
                           stride_x,
                           y,
                           offset_y,
                           incy,
                           stride_y,
                           param[0],
                           param[1],
                           param[2],
                           param[3],
                           param[4],
                           stride_param);

    return rocblas_status_success;
}