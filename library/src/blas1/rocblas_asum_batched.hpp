/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"
#include "rocblas_asum.hpp"

template <rocblas_int NB, typename U, typename To>
rocblas_status rocblas_asum_batched_template(rocblas_handle handle,
                                             rocblas_int    n,
                                             U              x,
                                             rocblas_int    shiftx,
                                             rocblas_int    incx,
                                             rocblas_int    batch_count,
                                             To*            workspace,
                                             To*            results)
{
    // Quick return if possible.
    if(!batch_count)
        return rocblas_status_success;
    if(n <= 0 || incx <= 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode)
            RETURN_IF_HIP_ERROR(hipMemsetAsync(results, 0, batch_count * sizeof(To)));
        else
        {
            for(int i = 0; i < batch_count; i++)
                results[i] = 0;
        }
        return rocblas_status_success;
    }

    return rocblas_reduction_strided_batched_kernel<NB, rocblas_fetch_asum<To>>(
        handle, n, x, shiftx, incx, 0, batch_count, workspace, results);
}
