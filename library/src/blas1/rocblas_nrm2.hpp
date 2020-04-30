/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas_reduction_template.hpp"

template <class To>
struct rocblas_fetch_nrm2
{
    template <class Ti>
    __forceinline__ __device__ To operator()(Ti x, ptrdiff_t tid)
    {
        return {fetch_abs2(x)};
    }
};

struct rocblas_finalize_nrm2
{
    template <class To>
    __forceinline__ __host__ __device__ To operator()(To x)
    {
        return sqrt(x);
    }
};

template <rocblas_int NB, bool ISBATCHED, typename Ti, typename To>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_nrm2_template(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const Ti*      x,
                                                             rocblas_int    shiftx,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             rocblas_int    batch_count,
                                                             To*            results,
                                                             To*            workspace)
{
    return rocblas_reduction_template<NB,
                                      ISBATCHED,
                                      rocblas_fetch_nrm2<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_nrm2>(
        handle, n, x, shiftx, incx, stridex, batch_count, results, workspace);
}
