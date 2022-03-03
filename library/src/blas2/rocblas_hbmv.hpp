/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

/**
  *  U is always: const T* (either host or device)
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <typename U, typename V, typename W>
rocblas_status rocblas_hbmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     rocblas_int    k,
                                     U              alpha,
                                     V              A,
                                     rocblas_int    offseta,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     V              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     U              beta,
                                     W              y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count);

//TODO :-Add rocblas_check_numerics_hb_matrix_template for checking Matrix `A` which is a Hermitian Band matrix
template <typename T, typename U>
rocblas_status rocblas_hbmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           rocblas_int    k,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
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
