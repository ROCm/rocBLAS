/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "reduction_iaminmax_strided_batched.hpp"
#include "rocblas_amax_amin.hpp"

template <rocblas_int NB, bool ISBATCHED, typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T                   x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_iaminmax_reduction_strided_batched<NB,
                                                      rocblas_fetch_amax_amin<S>,
                                                      rocblas_reduce_amin,
                                                      rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}
