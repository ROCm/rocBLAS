/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <typename T, typename U>
ROCBLAS_KERNEL_NO_BOUNDS
    rocblas_rotg_check_numerics_vector_kernel(T                         a_in,
                                              rocblas_stride            offset_a,
                                              rocblas_stride            stride_a,
                                              T                         b_in,
                                              rocblas_stride            offset_b,
                                              rocblas_stride            stride_b,
                                              U                         c_in,
                                              rocblas_stride            offset_c,
                                              rocblas_stride            stride_c,
                                              T                         s_in,
                                              rocblas_stride            offset_s,
                                              rocblas_stride            stride_s,
                                              rocblas_check_numerics_t* abnormal);

template <typename T, typename U>
rocblas_status rocblas_rotg_check_numerics_template(const char*    function_name,
                                                    rocblas_handle handle,
                                                    rocblas_int    n,
                                                    T              a_in,
                                                    rocblas_stride offset_a,
                                                    rocblas_stride stride_a,
                                                    T              b_in,
                                                    rocblas_stride offset_b,
                                                    rocblas_stride stride_b,
                                                    U              c_in,
                                                    rocblas_stride offset_c,
                                                    rocblas_stride stride_c,
                                                    T              s_in,
                                                    rocblas_stride offset_s,
                                                    rocblas_stride stride_s,
                                                    rocblas_int    batch_count,
                                                    const int      check_numerics,
                                                    bool           is_input);

template <typename T, typename U>
rocblas_status rocblas_rotg_template(rocblas_handle handle,
                                     T              a_in,
                                     rocblas_stride offset_a,
                                     rocblas_stride stride_a,
                                     T              b_in,
                                     rocblas_stride offset_b,
                                     rocblas_stride stride_b,
                                     U              c_in,
                                     rocblas_stride offset_c,
                                     rocblas_stride stride_c,
                                     T              s_in,
                                     rocblas_stride offset_s,
                                     rocblas_stride stride_s,
                                     rocblas_int    batch_count);
