/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched2.h"
#include "rocblas.h"

// same as non-batched for now
template <class To>
struct rocblas_fetch_nrm2_batched
{
    template <class Ti>
    __forceinline__ __device__ To operator()(Ti x, ptrdiff_t tid)
    {
        return {fetch_abs2(x)};
    }
};

struct rocblas_finalize_nrm2_batched
{
    template <class To>
    __forceinline__ __host__ __device__ To operator()(To x)
    {
        return sqrt(x);
    }
};

template <rocblas_int NB, typename Ti, typename To>
rocblas_status rocblas_nrm2_batched_template(rocblas_handle  handle,
                                             rocblas_int     n,
                                             const Ti* const x[],
                                             rocblas_int     shiftx,
                                             rocblas_int     incx,
                                             rocblas_int     batch_count,
                                             To*             workspace,
                                             To*             results)
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

    return rocblas_reduction_strided_batched_kernel<NB, Ti,
                                            rocblas_fetch_nrm2_batched<To>,
                                            rocblas_reduce_sum,
                                            rocblas_finalize_nrm2_batched>(
        handle, n, x, shiftx, incx, 0, batch_count, workspace, results);
}
