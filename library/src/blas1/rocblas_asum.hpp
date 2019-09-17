/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"

template <class To>
struct rocblas_fetch_asum
{
    template <typename Ti>
    __forceinline__ __device__ To operator()(Ti x, ptrdiff_t)
    {
        return {fetch_asum(x)};
    }
};

// allocate workspace inside this API
template <rocblas_int NB, typename Ti, typename To>
rocblas_status rocblas_asum_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const Ti*      x,
                                     rocblas_int    shiftx,
                                     rocblas_int    incx,
                                     To*            workspace,
                                     To*            result)
{
    // Quick return if possible.
    if(n <= 0 || incx <= 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode)
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        else
            *result = 0;
        return rocblas_status_success;
    }

    return rocblas_reduction_strided_batched_kernel<NB, Ti, rocblas_fetch_asum<To>>(
        handle, n, x, shiftx, incx, 0, 1, workspace, result);
}
