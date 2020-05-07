/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "fetch_template.h"
#include "handle.h"
#include "reduction_strided_batched.hpp"

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
    return rocblas_reduction_strided_batched_kernel<NB, FETCH, REDUCE, FINALIZE>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, results);
}
