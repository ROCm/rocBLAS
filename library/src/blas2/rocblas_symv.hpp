/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_hemv.hpp"

template <typename T, typename U, typename V, typename TPtr>
inline rocblas_status rocblas_symv_arg_check(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const V*       alpha,
                                             rocblas_stride stride_alpha,
                                             const U*       A,
                                             rocblas_int    offseta,
                                             rocblas_int    lda,
                                             rocblas_stride strideA,
                                             const U*       x,
                                             rocblas_int    offsetx,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             const V*       beta,
                                             rocblas_stride stride_beta,
                                             TPtr*          y,
                                             rocblas_int    offsety,
                                             rocblas_int    incy,
                                             rocblas_stride stridey,
                                             rocblas_int    batch_count)
{
    // only supports stride_alpha and stride_beta for device memory alpha/beta
    if((handle->pointer_mode == rocblas_pointer_mode_host) && (stride_alpha || stride_beta))
        return rocblas_status_not_implemented;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(n < 0 || lda < n || lda < 1 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if(!A || !x || !y || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  Note stride_alpha and stride_beta are only used AND only tested by rocSOLVER
  *  These strided scalar fetches are only supported for device_ptr mode
  */
template <typename T, typename U, typename V, typename TPtr, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const V*       alpha,
                                   rocblas_stride stride_alpha,
                                   const U*       A,
                                   rocblas_int    offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const U*       x,
                                   rocblas_int    offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const V*       beta,
                                   rocblas_stride stride_beta,
                                   TPtr*          y,
                                   rocblas_int    offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   W              workspace)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    // flag to check whether the kernel function being called is for hemv or symv
    // For hemv, IS_HEMV = true and for SYMV, IS_HEMV = false
    static constexpr bool IS_HEMV = false;

    /*Calling level 2 BLAS HEMV kernel functions in 'rocblas_hemv.hpp'. As SYMV and HEMV are nearly identical BLAS functions with the following changes
        1. In HEMV, the imaginary part of the main diagonal in the matrix `A` of is assumed to be zero. But, for SYMV both real and imaginary part is considered
        2. If matrix 'A' is a Hermitian matrix then A = A^H, where A^H is the conjugate transpose of matrix 'A', therefore the `conj()` helper function is used
        3. If matrix 'A' is a Symmetric matrix then A = A^T, Where A^T is the transpose of matrix 'A', therefore the `conj()` helper function is not used*/

    rocblas_status status = rocblas_internal_hemv_symv_template<IS_HEMV>(handle,
                                                                         uplo,
                                                                         n,
                                                                         alpha,
                                                                         stride_alpha,
                                                                         A,
                                                                         offseta,
                                                                         lda,
                                                                         strideA,
                                                                         x,
                                                                         offsetx,
                                                                         incx,
                                                                         stridex,
                                                                         beta,
                                                                         stride_beta,
                                                                         y,
                                                                         offsety,
                                                                         incy,
                                                                         stridey,
                                                                         batch_count,
                                                                         workspace);
    return status;
}

//TODO :-Add rocblas_check_numerics_sy_matrix_template for checking Matrix `A` which is a Symmetric Matrix
template <typename T, typename U>
rocblas_status rocblas_symv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
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
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}
