/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"

template <typename T,
          typename T2,
          typename U,
          typename V,
          std::enable_if_t<!is_complex<V>, int> = 0>
__global__ void rot_kernel(rocblas_int    n,
                           T2             x_in,
                           rocblas_int    offset_x,
                           rocblas_int    incx,
                           rocblas_stride stride_x,
                           T2             y_in,
                           rocblas_int    offset_y,
                           rocblas_int    incy,
                           rocblas_stride stride_y,
                           U              c_device_host,
                           rocblas_stride c_stride,
                           V              s_device_host,
                           rocblas_stride s_stride)
{
    auto      c   = load_scalar(c_device_host, hipBlockIdx_y, c_stride);
    auto      s   = load_scalar(s_device_host, hipBlockIdx_y, s_stride);
    auto      x   = load_ptr_batch(x_in, hipBlockIdx_y, offset_x, stride_x);
    auto      y   = load_ptr_batch(y_in, hipBlockIdx_y, offset_y, stride_y);
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

template <typename T, typename T2, typename U, typename V, std::enable_if_t<is_complex<V>, int> = 0>
__global__ void rot_kernel(rocblas_int    n,
                           T2             x_in,
                           rocblas_int    offset_x,
                           rocblas_int    incx,
                           rocblas_stride stride_x,
                           T2             y_in,
                           rocblas_int    offset_y,
                           rocblas_int    incy,
                           rocblas_stride stride_y,
                           U              c_device_host,
                           rocblas_stride c_stride,
                           V              s_device_host,
                           rocblas_stride s_stride)
{
    auto      c   = load_scalar(c_device_host, hipBlockIdx_y, c_stride);
    auto      s   = load_scalar(s_device_host, hipBlockIdx_y, s_stride);
    auto      x   = load_ptr_batch(x_in, hipBlockIdx_y, offset_x, stride_x);
    auto      y   = load_ptr_batch(y_in, hipBlockIdx_y, offset_y, stride_y);
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

template <rocblas_int NB, typename T, typename T2, typename U, typename V>
rocblas_status rocblas_rot_template(rocblas_handle handle,
                                    rocblas_int    n,
                                    T2             x,
                                    rocblas_int    offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    T2             y,
                                    rocblas_int    offset_y,
                                    rocblas_int    incy,
                                    rocblas_stride stride_y,
                                    U*             c,
                                    rocblas_stride c_stride,
                                    V*             s,
                                    rocblas_stride s_stride,
                                    rocblas_int    batch_count)
{
    // Quick return if possible
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    auto shiftx = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    auto shifty = incy < 0 ? offset_y - ptrdiff_t(incy) * (n - 1) : offset_y;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rot_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           x,
                           shiftx,
                           incx,
                           stride_x,
                           y,
                           shifty,
                           incy,
                           stride_y,
                           c,
                           c_stride,
                           s,
                           s_stride);
    else // c and s are on host
        hipLaunchKernelGGL(rot_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           x,
                           shiftx,
                           incx,
                           stride_x,
                           y,
                           shifty,
                           incy,
                           stride_y,
                           *c,
                           c_stride,
                           *s,
                           s_stride);

    return rocblas_status_success;
}
