/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "gemv_device.hpp"
#include "handle.hpp"
#include "rocblas_gemv_threshold.hpp"

template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t rocblas_internal_gemv_kernel_workspace_size(
    rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int batch_count = 1);

template <typename T, typename U, typename V, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemv_template(rocblas_handle    handle,
                                   rocblas_operation transA,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   const U*          alpha,
                                   rocblas_stride    stride_alpha,
                                   const V*          A,
                                   rocblas_int       offseta,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   const V*          x,
                                   rocblas_int       offsetx,
                                   rocblas_int       incx,
                                   rocblas_stride    stridex,
                                   const U*          beta,
                                   rocblas_stride    stride_beta,
                                   W*                y,
                                   rocblas_int       offsety,
                                   rocblas_int       incy,
                                   rocblas_stride    stridey,
                                   rocblas_int       batch_count,
                                   T*                workspace = nullptr);

template <typename T, typename U>
rocblas_status rocblas_gemv_check_numerics(const char*       function_name,
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
