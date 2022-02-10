/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"
#include "rocblas.h"

template <typename T, typename U>
rocblas_status rocblas_axpy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           ptrdiff_t      offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           ptrdiff_t      offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
//!
//! @brief General template to compute y = a * x + y.
//!
template <int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   Tx             x,
                                   ptrdiff_t      offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   Ty             y,
                                   ptrdiff_t      offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   rocblas_int    batch_count);
