/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <typename T>
bool quick_return_param(rocblas_handle handle, const T* param, rocblas_stride stride_param);

template <typename T>
bool quick_return_param(rocblas_handle handle, const T* const param[], rocblas_stride stride_param);

template <typename T>
rocblas_status rocblas_rotm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           T              y,
                                           rocblas_stride offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);

template <rocblas_int NB, bool BATCHED_OR_STRIDED, typename T, typename U>
rocblas_status rocblas_rotm_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     T              x,
                                     rocblas_stride offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     T              y,
                                     rocblas_stride offset_y,
                                     rocblas_int    incy,
                                     rocblas_stride stride_y,
                                     U              param,
                                     rocblas_stride offset_param,
                                     rocblas_stride stride_param,
                                     rocblas_int    batch_count);
