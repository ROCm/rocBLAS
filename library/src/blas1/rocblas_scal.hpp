/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"
#include "rocblas/rocblas.h"

template <rocblas_int NB, typename Tex, typename Ta, typename Tx>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   Tx             x,
                                   rocblas_int    offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   rocblas_int    batch_count);
