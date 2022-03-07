/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <typename T, typename U>
rocblas_status rocblas_tpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    m,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);

template <rocblas_int NB, typename A, typename X, typename W>
rocblas_status rocblas_tpmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transa,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     A                 a,
                                     rocblas_stride    offseta,
                                     rocblas_stride    stridea,
                                     X                 x,
                                     rocblas_stride    offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     W                 workspace,
                                     rocblas_stride    stridew,
                                     rocblas_int       batch_count);
