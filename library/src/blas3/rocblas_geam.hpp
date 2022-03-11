/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_geam_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal             alpha,
                                     TConstPtr         A,
                                     rocblas_stride    offset_a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TScal             beta,
                                     TConstPtr         B,
                                     rocblas_stride    offset_b,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_b,
                                     TPtr              C,
                                     rocblas_stride    offset_c,
                                     rocblas_int       ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count);

template <typename T, typename U>
rocblas_status rocblas_geam_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           T                 A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           T                 B,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           U                 C,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count,
                                           const int         check_numerics,
                                           bool              is_input);
