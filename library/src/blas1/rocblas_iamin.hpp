/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas_amax_amin.h"

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ void
        operator()(rocblas_index_value_t<To>& __restrict__ x,
                   const rocblas_index_value_t<To>& __restrict__ y)
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value < x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

template <rocblas_int NB, typename T, typename S>
rocblas_status rocblas_iamin_template(rocblas_handle            handle,
                                      rocblas_int               n,
                                      const T*                  x,
                                      rocblas_int               shiftx,
                                      rocblas_int               incx,
                                      rocblas_int*              result,
                                      rocblas_index_value_t<S>* workspace)
{
    static constexpr bool           isbatched     = false;
    static constexpr rocblas_stride stridex_0     = 0;
    static constexpr rocblas_int    batch_count_1 = 1;
    return rocblas_reduction_template<NB,
                                      isbatched,
                                      rocblas_fetch_amax_amin<S>,
                                      rocblas_reduce_amin,
                                      rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex_0, batch_count_1, result, workspace);
}
