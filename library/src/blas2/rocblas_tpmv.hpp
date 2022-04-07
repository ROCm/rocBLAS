/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <typename T, typename U, typename X>
inline rocblas_status rocblas_tpmv_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             U                 A,
                                             X                 x,
                                             rocblas_int       incx,
                                             rocblas_int       batch_count,
                                             size_t&           dev_bytes)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)
        return rocblas_status_invalid_value;

    if(m < 0 || !incx || batch_count < 0)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!m || !batch_count)
    {
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        return rocblas_status_success;
    }

    dev_bytes = sizeof(T) * batch_count * m;
    if(handle->is_device_memory_size_query())
        return handle->set_optimal_device_memory_size(dev_bytes);

    if(!A || !x)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

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
