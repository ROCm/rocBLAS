/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas_reduction_template.hpp"

template <typename T, std::enable_if_t<!std::is_same<T, rocblas_half>{}, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return std::norm(A);
}

template <typename T, std::enable_if_t<std::is_same<T, rocblas_half>{}, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return A * A;
}
template <class To>
struct rocblas_fetch_nrm2
{
    template <class Ti>
    __forceinline__ __device__ To operator()(Ti x) const
    {
        return {fetch_abs2(x)};
    }
};

struct rocblas_finalize_nrm2
{
    template <class To>
    __forceinline__ __host__ __device__ To operator()(To x) const
    {
        return sqrt(x);
    }
};

template <rocblas_int NB, bool ISBATCHED, typename Ti, typename To, typename Tex = To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_nrm2_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ti*      x,
                                   rocblas_int    shiftx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   rocblas_int    batch_count,
                                   To*            results,
                                   Tex*           workspace)
{
    return rocblas_reduction_template<NB,
                                      ISBATCHED,
                                      rocblas_fetch_nrm2<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_nrm2>(
        handle, n, x, shiftx, incx, stridex, batch_count, results, workspace);
}
