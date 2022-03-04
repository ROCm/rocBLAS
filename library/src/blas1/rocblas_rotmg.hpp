/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <typename T, typename U>
rocblas_status rocblas_rotmg_template(rocblas_handle handle,
                                      T              d1_in,
                                      rocblas_int    offset_d1,
                                      rocblas_stride stride_d1,
                                      T              d2_in,
                                      rocblas_int    offset_d2,
                                      rocblas_stride stride_d2,
                                      T              x1_in,
                                      rocblas_int    offset_x1,
                                      rocblas_stride stride_x1,
                                      U              y1_in,
                                      rocblas_int    offset_y1,
                                      rocblas_stride stride_y1,
                                      T              param,
                                      rocblas_int    offset_param,
                                      rocblas_stride stride_param,
                                      rocblas_int    batch_count);

template <typename T, typename U>
rocblas_status rocblas_rotmg_check_numerics_template(const char*    function_name,
                                                     rocblas_handle handle,
                                                     rocblas_int    n,
                                                     T              d1_in,
                                                     rocblas_int    offset_d1,
                                                     rocblas_stride stride_d1,
                                                     T              d2_in,
                                                     rocblas_int    offset_d2,
                                                     rocblas_stride stride_d2,
                                                     T              x1_in,
                                                     rocblas_int    offset_x1,
                                                     rocblas_stride stride_x1,
                                                     U              y1_in,
                                                     rocblas_int    offset_y1,
                                                     rocblas_stride stride_y1,
                                                     rocblas_int    batch_count,
                                                     const int      check_numerics,
                                                     bool           is_input);
