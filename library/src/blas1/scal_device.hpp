/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __SCAL_DEVICE_HPP__
#define __SCAL_DEVICE_HPP__

#include "utility.h"

template <typename T, typename U>
__device__ void scal_kernel(rocblas_int n, U alpha_device_host, T* x, rocblas_int incx)
{
    auto      alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= alpha;
}

template <typename T, typename U>
__global__ void scal_kernel_batched(
    rocblas_int n, U alpha_device_host, T* xa[], rocblas_int offsetx, rocblas_int incx)
{
    T* x = xa[hipBlockIdx_y] + offsetx;
    scal_kernel<T, U>(n, alpha_device_host, x, incx);
}

template <typename T, typename U>
__global__ void scal_kernel_strided_batched(rocblas_int n,
                                            U           alpha_device_host,
                                            T*          xa,
                                            rocblas_int offsetx,
                                            rocblas_int incx,
                                            rocblas_int stridex)
{
    T* x = xa + hipBlockIdx_y * stridex + offsetx;
    scal_kernel<T, U>(n, alpha_device_host, x, incx);
}

#endif // \IncludeGuard