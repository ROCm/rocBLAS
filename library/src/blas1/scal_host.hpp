/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

///////////
// DEVICE//
///////////
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
                                            rocblas_int incx,
                                            rocblas_int stridex)
{
    T* x = xa + hipBlockIdx_y * stridex;
    scal_kernel<T, U>(n, alpha_device_host, x, incx);
}


///////////
/// HOST///
///////////
template <typename T, typename U>
rocblas_status rocblas_scal_batched_template(rocblas_handle handle,
                                             rocblas_int    n,
                                             const U*       alpha,
                                             T*             x[],
                                             rocblas_int    offsetx,
                                             rocblas_int    incx,
                                             rocblas_int    batch_count)
{
    static constexpr int NB = 256;
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
        return rocblas_status_success;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(
            scal_kernel_batched, blocks, threads, 0, rocblas_stream, n, alpha, x, offsetx, incx);
    else // alpha is on host
        hipLaunchKernelGGL(
            scal_kernel_batched, blocks, threads, 0, rocblas_stream, n, *alpha, x, offsetx, incx);

    return rocblas_status_success;
}

template <typename T, typename U>
rocblas_status rocblas_scal_strided_batched_template(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const U*       alpha,
                                                     T*             x,
                                                     rocblas_int    incx,
                                                     rocblas_int    stridex,
                                                     rocblas_int    batch_count)
{
    static constexpr int NB = 256;
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count == 0)
        return rocblas_status_success;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(scal_kernel_strided_batched,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           alpha,
                           x,
                           incx,
                           stridex);
    else // alpha is on host
        hipLaunchKernelGGL(scal_kernel_strided_batched,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           *alpha,
                           x,
                           incx,
                           stridex);

    return rocblas_status_success;
}