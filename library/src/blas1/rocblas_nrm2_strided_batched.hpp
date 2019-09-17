/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"
#include "rocblas_nrm2.hpp"

template <rocblas_int NB, typename Ti, typename To>
rocblas_status rocblas_nrm2_strided_batched_template(rocblas_handle handle,
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

    return rocblas_reduction_strided_batched_kernel<NB,
                                                    Ti,
                                                    rocblas_fetch_nrm2<To>,
                                                    rocblas_reduce_sum,
                                                    rocblas_finalize_nrm2>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, results);
}
