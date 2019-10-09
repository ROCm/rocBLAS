/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

template <typename T, typename U, typename V>
__global__ void rocblas_scal_kernel(rocblas_int    n,
                                    V              alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    U              xa,
                                    ptrdiff_t      offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex)
{
    T*        x     = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto      alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= alpha;
}

template <rocblas_int NB, typename T, typename U, typename V>
rocblas_status rocblas_scal_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const V*       alpha,
                                     rocblas_stride stride_alpha,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     rocblas_int    batch_count,
                                     V*             mem)
{
    // Memory queries must be in template as _impl doesn't have stride_alpha parameter (for calls from
    // outside of rocblas)
    if(handle->is_device_memory_size_query())
    {
        if(stride_alpha && rocblas_pointer_mode_host == handle->pointer_mode && n > 0 && incx > 0
           && batch_count > 0)
            return handle->set_optimal_device_memory_size(sizeof(V) * batch_count * stride_alpha);
        else
            return rocblas_status_size_unchanged;
    }
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
    {
        return rocblas_status_success;
    }

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rocblas_scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           alpha,
                           stride_alpha,
                           x,
                           offsetx,
                           incx,
                           stridex);
    else if(!stride_alpha) // single alpha is on host
    {
        hipLaunchKernelGGL(rocblas_scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           *alpha,
                           stride_alpha,
                           x,
                           offsetx,
                           incx,
                           stridex);
    }
    else // array of alphas on host - copy to device
    {
        // This should NOT happen from calls from the API currently.
        RETURN_IF_HIP_ERROR(
            hipMemcpy(mem, alpha, sizeof(V) * batch_count * stride_alpha, hipMemcpyHostToDevice));

        hipLaunchKernelGGL(rocblas_scal_kernel<T>,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           mem,
                           stride_alpha,
                           x,
                           offsetx,
                           incx,
                           stridex);
    }

    return rocblas_status_success;
}
