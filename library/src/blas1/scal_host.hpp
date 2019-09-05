/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "rocblas.h"
#include "scal_device.hpp"
#include "utility.h"

template <typename T, typename U>
rocblas_status rocblas_scal_batched_template(rocblas_handle handle,
                                             rocblas_int    n,
                                             const U*       alpha,
                                             T*             x[],
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
            scal_kernel_batched, blocks, threads, 0, rocblas_stream, n, alpha, x, incx);
    else // alpha is on host
        hipLaunchKernelGGL(
            scal_kernel_batched, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx);

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