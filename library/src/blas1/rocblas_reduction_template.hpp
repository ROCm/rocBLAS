/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"

// allocate workspace inside this API
template <rocblas_int NB,
          bool        ISBATCHED,
          typename FETCH,
          typename REDUCE,
          typename FINALIZE,
          typename U,
          typename Tr,
          typename Tw>
rocblas_status rocblas_reduction_template(rocblas_handle handle,
                                          rocblas_int    n,
                                          U              x,
                                          rocblas_int    shiftx,
                                          rocblas_int    incx,
                                          rocblas_stride stridex,
                                          rocblas_int    batch_count,
                                          Tr*            results,
                                          Tw*            workspace)
{

    // Quick return if possible.
    if(n <= 0 || incx <= 0 || (ISBATCHED && batch_count <= 0))
    {
        if(n > 0 && incx > 0 && batch_count < 0)
        {
            return rocblas_status_invalid_size;
        }
        else
        {
            if(handle->is_device_memory_size_query())
            {
                return rocblas_status_size_unchanged;
            }
            else if(rocblas_pointer_mode_device == handle->pointer_mode && batch_count > 0)
            {
                RETURN_IF_HIP_ERROR(hipMemset(results, 0, batch_count * sizeof(Tr)));
            }
            else
            {
                for(rocblas_int batch_index = 0; batch_index < batch_count; batch_index++)
                {
                    results[batch_index] = Tr(0);
                }
            }
        }
        return rocblas_status_success;
    }

    return rocblas_reduction_strided_batched_kernel<NB, FETCH, REDUCE, FINALIZE>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, results);
}
