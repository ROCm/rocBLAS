/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <bool CONJ, rocblas_int NB, typename T, typename U>
rocblas_status rocblas_copy_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     T              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     U              y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count);

template <typename T, typename U>
rocblas_status rocblas_copy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_int    offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
