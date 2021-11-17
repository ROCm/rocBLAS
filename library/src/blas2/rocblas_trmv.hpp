/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "../blas1/rocblas_dot.hpp"
#include "rocblas/rocblas.h"
#include <cstddef>

template <typename A, typename X, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmv_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_diagonal  diag,
                                   rocblas_int       m,
                                   A                 a,
                                   ptrdiff_t         offseta,
                                   rocblas_int       lda,
                                   rocblas_stride    stridea,
                                   X                 x,
                                   ptrdiff_t         offsetx,
                                   rocblas_int       incx,
                                   rocblas_stride    stridex,
                                   W                 workspace,
                                   rocblas_stride    stridew,
                                   rocblas_int       batch_count);

//TODO :-Add rocblas_check_numerics_tr_matrix_template for checking Matrix `A` which is a Triangular Matrix
template <typename T, typename U>
rocblas_status rocblas_trmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    m,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
