/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas_iamax.hpp"

template <rocblas_int NB, typename T, typename S>
rocblas_status rocblas_iamax_batched_template(rocblas_handle            handle,
                                              rocblas_int               n,
                                              const T* const*           x,
                                              rocblas_int               shiftx,
                                              rocblas_int               incx,
                                              rocblas_int               batch_count,
                                              rocblas_int*              result,
                                              rocblas_index_value_t<S>* workspace)
{
    static constexpr bool           isbatched = true;
    static constexpr rocblas_stride stridex_0 = 0;
    return rocblas_reduction_template<NB,
                                      isbatched,
                                      rocblas_fetch_amax_amin<S>,
                                      rocblas_reduce_amax,
                                      rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex_0, batch_count, result, workspace);
}
