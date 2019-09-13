/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U, typename V>
__global__ void scal_kernel(rocblas_int n,
                            V           alpha_device_host,
                            rocblas_int inca,
                            U           xa,
                            rocblas_int offsetx,
                            rocblas_int incx,
                            rocblas_int stridex)
{
    T*        x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto      alpha = load_scalar(alpha_device_host, hipBlockIdx_y, inca);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= alpha;
}

template <rocblas_int NB, typename T, typename U, typename V>
rocblas_status rocblas_scal_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const V*       alpha,
                                     rocblas_int    inca,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_int    stridex,
                                     rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        return rocblas_status_success;
    }

    if(handle->is_device_memory_size_query())
    {
        if(rocblas_pointer_mode_host == handle->pointer_mode && inca != 0)
            return handle->set_optimal_device_memory_size(sizeof(V) * batch_count * inca);
        else
            return rocblas_status_size_unchanged;
    }

    if(n <= 0 || incx <= 0 || batch_count <= 0)
        return rocblas_status_success;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           alpha,
                           inca,
                           x,
                           offsetx,
                           incx,
                           stridex);
    else if(!inca) // single alpha is on host
    {
        hipLaunchKernelGGL(scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           *alpha,
                           inca,
                           x,
                           offsetx,
                           incx,
                           stridex);
    }
    else // array of alphas on host - copy to device
    {
        auto mem = handle->device_malloc(sizeof(V) * batch_count * inca);
        if(!mem)
            return rocblas_status_memory_error;
        RETURN_IF_HIP_ERROR(
            hipMemcpy((V*)mem, alpha, sizeof(V) * batch_count * inca, hipMemcpyHostToDevice));

        hipLaunchKernelGGL(scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           (V*)mem,
                           inca,
                           x,
                           offsetx,
                           incx,
                           stridex);
    }

    return rocblas_status_success;
}
