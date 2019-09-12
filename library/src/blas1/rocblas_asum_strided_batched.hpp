/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"

template <class To>
struct rocblas_fetch_asum_strided_batched
{
    template <typename Ti>
    __forceinline__ __device__ To operator()(Ti x, ptrdiff_t)
    {
        return {fetch_asum(x)};
    }
};

template <rocblas_int NB, typename To>
size_t rocblas_asum_strided_batched_template_workspace_size(rocblas_int n,
                                                            rocblas_int batch_count,
                                                            To*         output_type)
{
    return rocblas_reduction_batched_kernel_workspace_size<NB>(n, batch_count, output_type);
}

template <rocblas_int NB, typename Ti, typename To>
rocblas_status rocblas_asum_strided_batched_template(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const Ti*      x,
                                                     rocblas_int    shiftx,
                                                     rocblas_int    incx,
                                                     rocblas_int    stridex,
                                                     rocblas_int    batch_count,
                                                     To*            workspace,
                                                     To*            results)
{

    // Quick return if possible.
    if(n <= 0 || incx <= 0 || batch_count == 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode && batch_count > 0)
            RETURN_IF_HIP_ERROR(hipMemset(results, 0, batch_count * sizeof(To)));
        else
        {
            for(int i = 0; i < batch_count; i++)
            {
                results[i] = 0;
            }
        }
        return rocblas_status_success;
    }

    return rocblas_reduction_strided_batched_kernel<NB, rocblas_fetch_asum_strided_batched<To>>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, results);
}
