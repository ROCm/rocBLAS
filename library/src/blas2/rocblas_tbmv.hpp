/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <typename T, typename U, typename X>
inline rocblas_status rocblas_tbmv_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       k,
                                             U                 A,
                                             rocblas_int       lda,
                                             X                 x,
                                             rocblas_int       incx,
                                             rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)
        return rocblas_status_invalid_value;

    if(m < 0 || k < 0 || lda < k + 1 || !incx || batch_count < 0)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!m || !batch_count)
    {
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        return rocblas_status_success;
    }

    return rocblas_status_continue;
}

template <typename U, typename V>
rocblas_status rocblas_tbmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       k,
                                     U                 A,
                                     rocblas_int       offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     V                 x,
                                     rocblas_int       offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     rocblas_int       batch_count,
                                     V                 w_x_copy);

template <typename T, typename U>
rocblas_status rocblas_tbmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    m,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
