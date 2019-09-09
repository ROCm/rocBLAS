/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __TRSM_DEVICE_HPP__
#define __TRSM_DEVICE_HPP__

#include "utility.h"

template <typename T>
__device__ void copy_matrix_trsm(rocblas_int rows,
                                 rocblas_int cols,
                                 rocblas_int elem_size,
                                 const T*    a,
                                 rocblas_int lda,
                                 T*          b,
                                 rocblas_int ldb)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < rows && ty < cols)
        b[tx + ldb * ty] = a[tx + lda * ty];
}

template <typename T>
__global__ void copy_matrix_strided_batched_trsm(rocblas_int rows,
                                                 rocblas_int cols,
                                                 rocblas_int elem_size,
                                                 const T*    a,
                                                 rocblas_int lda,
                                                 rocblas_int stride_a,
                                                 T*          b,
                                                 rocblas_int ldb,
                                                 rocblas_int stride_b)
{
    const T* xa = a + hipBlockIdx_z * stride_a;
    T*       xb = b + hipBlockIdx_z * stride_b;
    copy_matrix_trsm(rows, cols, elem_size, xa, lda, xb, ldb);
}

template <typename T>
__global__ void copy_matrix_batched_trsm(rocblas_int rows,
                                         rocblas_int cols,
                                         rocblas_int elem_size,
                                         const T*    a[],
                                         rocblas_int lda,
                                         T*          b[],
                                         rocblas_int ldb,
                                         rocblas_int offset_a,
                                         rocblas_int offset_b)
{
    const T* xa = a[hipBlockIdx_z] + offset_a;
    T*       xb = b[hipBlockIdx_z] + offset_b;
    copy_matrix_trsm(rows, cols, elem_size, xa, lda, xb, ldb);
}

#endif // \IncludeGuard
