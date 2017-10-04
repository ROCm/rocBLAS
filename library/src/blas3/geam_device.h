/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// general case for any alpha, beta, lda, ldb, ldc
template<typename T>
static __device__ void
geam_device(
    rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n,
    T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ B, rocblas_int ldb,
    T beta,
          T *              C, rocblas_int ldc)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (tx < m && ty < n)
    {
        if (transA == rocblas_operation_none && transB == rocblas_operation_none)
        {
            C[tx + ldc*ty] = (alpha) * A[tx + lda*ty] + beta * B[tx + ldb*ty];
        }
        else if (transA == rocblas_operation_none && transB == rocblas_operation_transpose)
        {
            C[tx + ldc*ty] = (alpha) * A[tx + lda*ty] + beta * B[ldb*tx + ty];
        }
        else if (transA == rocblas_operation_transpose && transB == rocblas_operation_none)
        {
            C[tx + ldc*ty] = (alpha) * A[lda*tx + ty] + beta * B[tx + ldb*ty];
        }
        else if (transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
        {
            C[tx + ldc*ty] = (alpha) * A[lda*tx + ty] + beta * B[ldb*tx + ty];
        }
    }
}

//  special case:
//  only one matrix contributes because   0 == alpha || 0 == beta
template<typename T>
static __device__ void
geam_2matrix_device(
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    T alpha,
    const T * __restrict__ A, rocblas_int lda,
          T *              C, rocblas_int ldc)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (tx < m && ty < n)
    {
        if (alpha == 0)
        {
            C[tx + ldc*ty] = 0;
        }
        else if (transA == rocblas_operation_none)
        {
            C[tx + ldc*ty] = (alpha) * A[tx + lda*ty];
        }
        else
        {
            C[tx + ldc*ty] = (alpha) * A[lda*tx + ty];
        }
    }
}
                   
//  special case:
//  treat matrices as contiguous vectors because
//  m == lda && m == ldb && m == ldc && none == transA && none == transB
template<typename T>
static __device__ void
geam_1D_device(
    rocblas_int size,
    T alpha,
    const T * __restrict__ A,
    const T * __restrict__ B,
    T beta,
          T *              C)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tx < size)
    {
        if (alpha == 0 && beta == 0)
        {
            C[tx] = 0;
        }
        else
        {
            C[tx] = alpha * A[tx] + beta * B[tx];
        }
    }
}
                   
//  special case:
//  treat matrices as contiguous vectors, and only one matrix contributes, because
//  m == lda && m == ldb && m == ldc && none == transA && none == transB && (0 == alpha || 0 == beta)
template<typename T>
static __device__ void
geam_1D_2matrix_device(
    rocblas_int size,
    T alpha,
    const T * __restrict__ A,
          T *              C)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tx < size)
    {
        if (alpha == 0)
        {
            C[tx] = 0;
        }
        else
        {
            C[tx] = alpha * A[tx];
        }
    }
}
