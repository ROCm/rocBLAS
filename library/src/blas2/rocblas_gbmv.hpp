/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"

/**
  *  Here, U is either a `const T* const*` or a `const T*`
  *  V is either a `T*` or a `T* const*`
  */
template <typename T, typename U, typename V>
rocblas_status rocblas_gbmv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       kl,
                                     rocblas_int       ku,
                                     const T*          alpha,
                                     U                 A,
                                     rocblas_stride    offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     U                 x,
                                     rocblas_stride    offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     const T*          beta,
                                     V                 y,
                                     rocblas_stride    offsety,
                                     rocblas_int       incy,
                                     rocblas_stride    stridey,
                                     rocblas_int       batch_count);

//TODO :-Add rocblas_check_numerics_gb_matrix_template for checking Matrix `A` which is a General Band matrix
template <typename T, typename U>
rocblas_status rocblas_gbmv_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           T                 A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           T                 x,
                                           rocblas_stride    offset_x,
                                           rocblas_int       inc_x,
                                           rocblas_stride    stride_x,
                                           U                 y,
                                           rocblas_stride    offset_y,
                                           rocblas_int       inc_y,
                                           rocblas_stride    stride_y,
                                           rocblas_int       batch_count,
                                           const int         check_numerics,
                                           bool              is_input);
