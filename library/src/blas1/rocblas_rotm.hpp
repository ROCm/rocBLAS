/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U>
__device__ void rotm_kernel_calc(rocblas_int    n,
                                 T              x_in,
                                 rocblas_int    offset_x,
                                 rocblas_int    incx,
                                 rocblas_stride stride_x,
                                 T              y_in,
                                 rocblas_int    offset_y,
                                 rocblas_int    incy,
                                 rocblas_stride stride_y,
                                 U              flag,
                                 U              h11,
                                 U              h21,
                                 U              h12,
                                 U              h22)
{
    auto      x   = load_ptr_batch(x_in, hipBlockIdx_y, offset_x, stride_x);
    auto      y   = load_ptr_batch(y_in, hipBlockIdx_y, offset_y, stride_y);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

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

template <typename T, typename U>
__global__ void rotm_kernel_batched(rocblas_int    n,
                                    T              x_in,
                                    rocblas_int    offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    T              y_in,
                                    rocblas_int    offset_y,
                                    rocblas_int    incy,
                                    rocblas_stride stride_y,
                                    U              param,
                                    rocblas_int    offset_param,
                                    rocblas_stride stride_param)
{
    auto p    = load_ptr_batch(param, hipBlockIdx_y, offset_param, stride_param);
    auto flag = p[0];
    auto h11  = p[1];
    auto h21  = p[2];
    auto h12  = p[3];
    auto h22  = p[4];
    rotm_kernel_calc(n,
                     x_in,
                     offset_x,
                     incx,
                     stride_x,
                     y_in,
                     offset_y,
                     incy,
                     stride_y,
                     flag,
                     h11,
                     h21,
                     h12,
                     h22);
}

template <typename T>
__global__ void rotm_kernel_regular(rocblas_int    n,
                                    T*             x_in,
                                    rocblas_int    offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    T*             y_in,
                                    rocblas_int    offset_y,
                                    rocblas_int    incy,
                                    rocblas_stride stride_y,
                                    T              flag,
                                    T              h11,
                                    T              h21,
                                    T              h12,
                                    T              h22)
{
    rotm_kernel_calc(n,
                     x_in,
                     offset_x,
                     incx,
                     stride_x,
                     y_in,
                     offset_y,
                     incy,
                     stride_y,
                     flag,
                     h11,
                     h21,
                     h12,
                     h22);
}

// Workaround to avoid constexpr if - Helper function to quick return when param[0] == -2
template <typename T>
bool quick_return_param(rocblas_handle handle, const T* param, rocblas_stride stride_param)
{
    if(rocblas_pointer_mode_host == handle->pointer_mode)
        if(param[0] == -2 && stride_param == 0)
            return true;
    return false;
}

template <typename T>
bool quick_return_param(rocblas_handle handle, const T* const param[], rocblas_stride stride_param)
{
    return false;
}

template <rocblas_int NB, bool BATCHED_OR_STRIDED, typename T, typename U>
rocblas_status rocblas_rotm_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     T              x,
                                     rocblas_int    offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     T              y,
                                     rocblas_int    offset_y,
                                     rocblas_int    incy,
                                     rocblas_stride stride_y,
                                     U              param,
                                     rocblas_int    offset_param,
                                     rocblas_stride stride_param,
                                     rocblas_int    batch_count)
{
    // Quick return if possible
    if(n <= 0 || incx <= 0 || incy <= 0 || batch_count <= 0)
        return rocblas_status_success;

    if(quick_return_param(handle, param, stride_param))
        return rocblas_status_success;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rotm_kernel_batched,
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
                           offset_param,
                           stride_param);
    else if(!BATCHED_OR_STRIDED)
        hipLaunchKernelGGL(rotm_kernel_regular,
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
                           param[4]);
    else // host mode not implemented for (strided_)batched functions
    {
        // TODO: if desired we can use a host for loop to iterate through
        //       batches in this scenario. Currently simply not implemented.
        return rocblas_status_not_implemented;
    }

    return rocblas_status_success;
}