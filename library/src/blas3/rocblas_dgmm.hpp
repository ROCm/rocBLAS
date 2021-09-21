/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#include "handle.hpp"

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_template(rocblas_handle handle,
                                     rocblas_side   side,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     TConstPtr      A,
                                     rocblas_int    offset_a,
                                     rocblas_int    lda,
                                     rocblas_stride stride_a,
                                     TConstPtr      X,
                                     rocblas_int    offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     TPtr           C,
                                     rocblas_int    offset_c,
                                     rocblas_int    ldc,
                                     rocblas_stride stride_c,
                                     rocblas_int    batch_count);
