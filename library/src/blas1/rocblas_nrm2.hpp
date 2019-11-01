/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
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

// allocate workspace inside this API
template <rocblas_int NB, typename Ti, typename To>
rocblas_status rocblas_nrm2_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     const Ti*      x,
                                     rocblas_int    shiftx,
                                     rocblas_int    incx,
                                     To*            workspace,
                                     To*            results)
{
    static constexpr bool           isbatched     = false;
    static constexpr rocblas_stride stridex_0     = 0;
    static constexpr rocblas_int    batch_count_1 = 1;

    return rocblas_reduction_template<NB,
                                      isbatched,
                                      rocblas_fetch_nrm2<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_nrm2>(
        handle, n, x, shiftx, incx, stridex_0, batch_count_1, results, workspace);
}
