/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <bool CONJ, typename T, typename U, typename V, typename W>
inline rocblas_status rocblas_ger_arg_check(rocblas_int    m,
                                            rocblas_int    n,
                                            const V*       alpha,
                                            rocblas_stride stride_alpha,
                                            const U*       x,
                                            rocblas_int    offsetx,
                                            rocblas_int    incx,
                                            rocblas_int    stridex,
                                            const U*       y,
                                            rocblas_int    offsety,
                                            rocblas_int    incy,
                                            rocblas_int    stridey,
                                            W*             A,
                                            rocblas_int    offsetA,
                                            rocblas_int    lda,
                                            rocblas_int    strideA,
                                            rocblas_int    batch_count)
{
    if(m < 0 || n < 0 || !incx || !incy || lda < m || lda < 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!alpha || !x || !y || !A)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool CONJ, typename T, typename U, typename V, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_template(rocblas_handle handle,
                                  rocblas_int    m,
                                  rocblas_int    n,
                                  const V*       alpha,
                                  rocblas_stride stride_alpha,
                                  const U*       x,
                                  rocblas_int    offsetx,
                                  rocblas_int    incx,
                                  rocblas_int    stridex,
                                  const U*       y,
                                  rocblas_int    offsety,
                                  rocblas_int    incy,
                                  rocblas_int    stridey,
                                  W*             A,
                                  rocblas_int    offsetA,
                                  rocblas_int    lda,
                                  rocblas_int    strideA,
                                  rocblas_int    batch_count);

template <typename T, typename U>
rocblas_status rocblas_ger_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          rocblas_int    m,
                                          rocblas_int    n,
                                          U              A,
                                          rocblas_int    offset_a,
                                          rocblas_int    lda,
                                          rocblas_stride stride_a,
                                          T              x,
                                          rocblas_int    offset_x,
                                          rocblas_int    inc_x,
                                          rocblas_stride stride_x,
                                          T              y,
                                          rocblas_int    offset_y,
                                          rocblas_int    inc_y,
                                          rocblas_stride stride_y,
                                          rocblas_int    batch_count,
                                          const int      check_numerics,
                                          bool           is_input);
