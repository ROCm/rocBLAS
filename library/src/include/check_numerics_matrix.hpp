/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas.h"

/**
  *
  * rocblas_check_numerics_ge_matrix_kernel(m, n, Aa, offset_a, lda, stride_a, abnormal)
  *
  * Info about rocblas_check_numerics_ge_matrix_kernel function:
  *
  *    It is the kernel function which checks a matrix for numerical abnormalities such as NaN/zero/Inf and updates the structure.
  *    ge in rocblas_check_numerics_ge_matrix_kernel refers to general.
  *
  * Parameters   : m            : number of rows of matrix 'A'
  *                n            : number of columns of matrix 'A'
  *                Aa           : Pointer to the matrix which is under consideration for numerical abnormalities
  *                offset_a     : Offset of matrix 'Aa'
  *                lda          : specifies the leading dimension of matrix 'Aa'
  *                stride_a     : Specifies the pointer increment between one matrix 'A_i' and the next one (Aa_i+1) (where (Aa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <typename T>
ROCBLAS_KERNEL_NO_BOUNDS rocblas_check_numerics_ge_matrix_kernel(rocblas_int               m,
                                                                 rocblas_int               n,
                                                                 T                         Aa,
                                                                 ptrdiff_t                 offset_a,
                                                                 rocblas_int               lda,
                                                                 rocblas_stride            stride_a,
                                                                 rocblas_check_numerics_t* abnormal)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);

        ptrdiff_t tid   = tx + ptrdiff_t(lda) * ty;
        auto      value = A[tid];
        if(!abnormal->has_zero && rocblas_iszero(value))
            abnormal->has_zero = true;
        if(!abnormal->has_NaN && rocblas_isnan(value))
            abnormal->has_NaN = true;
        if(!abnormal->has_Inf && rocblas_isinf(value))
            abnormal->has_Inf = true;
    }
}
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_check_numerics_ge_matrix_template(const char*       function_name,
                                                       rocblas_handle    handle,
                                                       rocblas_operation trans_a,
                                                       rocblas_int       m,
                                                       rocblas_int       n,
                                                       T                 A,
                                                       rocblas_int       offset_a,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       rocblas_int       batch_count,
                                                       const int         check_numerics,
                                                       bool              is_input);
