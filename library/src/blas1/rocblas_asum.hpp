/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas_reduction_template.hpp"

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
    static constexpr bool           isbatched     = false;
    static constexpr rocblas_stride stridex_0     = 0;
    static constexpr rocblas_int    batch_count_1 = 1;

    return rocblas_reduction_template<NB,
                                      isbatched,
                                      rocblas_fetch_asum<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_identity>(
        handle, n, x, shiftx, incx, stridex_0, batch_count_1, result, workspace);
}
