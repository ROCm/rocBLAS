/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hpmv_template(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     TScal          alpha,
                                     TConstPtr      AP,
                                     rocblas_int    offseta,
                                     rocblas_stride strideA,
                                     TConstPtr      x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     TScal          beta,
                                     TPtr           y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count);

//TODO :-Add rocblas_check_numerics_hp_matrix_template for checking Matrix `AP` which is a Hermitian Packed matrix
template <typename T, typename U>
rocblas_status rocblas_hpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              AP,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
