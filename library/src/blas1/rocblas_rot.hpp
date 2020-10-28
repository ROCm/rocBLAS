/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.hpp"

template <typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts,
          std::enable_if_t<!is_complex<Ts>, int> = 0>
__device__ void
    rot_kernel_calc(rocblas_int n, Tx* x, rocblas_int incx, Ty* y, rocblas_int incy, Tc c, Ts s)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        auto ix    = tid * incx;
        auto iy    = tid * incy;
        Tex  tempx = Tex(c * x[ix]) + Tex(s * y[iy]);
        Tex  tempy = Tex(c * y[iy]) - Tex((s)*x[ix]);
        y[iy]      = Ty(tempy);
        x[ix]      = Tx(tempx);
    }
}

template <typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts,
          std::enable_if_t<is_complex<Ts>, int> = 0>
__device__ void
    rot_kernel_calc(rocblas_int n, Tx* x, rocblas_int incx, Ty* y, rocblas_int incy, Tc c, Ts s)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        auto ix    = tid * incx;
        auto iy    = tid * incy;
        Tex  tempx = Tex(c * x[ix]) + Tex(s * y[iy]);
        Tex  tempy = Tex(c * y[iy]) - Tex(conj(s) * x[ix]);
        y[iy]      = Ty(tempy);
        x[ix]      = Tx(tempx);
    }
}

template <typename Tex, typename Tx, typename Ty, typename Tc, typename Ts>
__global__ void rot_kernel(rocblas_int    n,
                           Tx             x_in,
                           rocblas_int    offset_x,
                           rocblas_int    incx,
                           rocblas_stride stride_x,
                           Ty             y_in,
                           rocblas_int    offset_y,
                           rocblas_int    incy,
                           rocblas_stride stride_y,
                           Tc             c_in,
                           rocblas_stride c_stride,
                           Ts             s_in,
                           rocblas_stride s_stride)
{
    auto c = std::real(load_scalar(c_in, hipBlockIdx_y, c_stride));
    auto s = load_scalar(s_in, hipBlockIdx_y, s_stride);
    auto x = load_ptr_batch(x_in, hipBlockIdx_y, offset_x, stride_x);
    auto y = load_ptr_batch(y_in, hipBlockIdx_y, offset_y, stride_y);

    rot_kernel_calc<Tex>(n, x, incx, y, incy, c, s);
}

template <rocblas_int NB, typename Tex, typename Tx, typename Ty, typename Tc, typename Ts>
rocblas_status rocblas_rot_template(rocblas_handle handle,
                                    rocblas_int    n,
                                    Tx             x,
                                    rocblas_int    offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    Ty             y,
                                    rocblas_int    offset_y,
                                    rocblas_int    incy,
                                    rocblas_stride stride_y,
                                    Tc*            c,
                                    rocblas_stride c_stride,
                                    Ts*            s,
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

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rot_kernel<Tex>,
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
        hipLaunchKernelGGL(rot_kernel<Tex>,
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
