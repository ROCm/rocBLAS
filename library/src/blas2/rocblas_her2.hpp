/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

/**
 * TScal     is always: const T* (either2 host or device)
 * TConstPtr is either2: const T* OR const T* const*
 * TPtr      is either2:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   TScal          alpha,
                                   TConstPtr      x,
                                   rocblas_int    offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   TConstPtr      y,
                                   rocblas_int    offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   TPtr           A,
                                   rocblas_int    lda,
                                   rocblas_int    offset_A,
                                   rocblas_stride stride_A,
                                   rocblas_int    batch_count);

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_her2_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           U              x,
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
