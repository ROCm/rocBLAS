/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// general case for any alpha, beta, lda, ldb, ldc
template <typename T>
static __device__ void geam_device(rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   T alpha,
                                   const T* __restrict__ A,
                                   rocblas_int lda,
                                   T beta,
                                   const T* __restrict__ B,
                                   rocblas_int ldb,
                                   T* C,
                                   rocblas_int ldc)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        int a_index;
        int b_index;
        int c_index = tx + ldc * ty;

        if(transA == rocblas_operation_none)
        {
            a_index = tx + ty * lda;
        }
        else
        {
            a_index = tx * lda + ty;
        }

        if(transB == rocblas_operation_none)
        {
            b_index = tx + ty * ldb;
        }
        else
        {
            b_index = tx * ldb + ty;
        }

        C[c_index] = fma(beta, B[b_index], alpha * A[a_index]);
    }
}

//  special case:
//  only one matrix contributes because   0 == alpha || 0 == beta
template <typename T>
static __device__ void geam_2matrix_device(rocblas_operation transA,
                                           rocblas_int m,
                                           rocblas_int n,
                                           T alpha,
                                           const T* __restrict__ A,
                                           rocblas_int lda,
                                           T* C,
                                           rocblas_int ldc)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        int c_index = tx + ldc * ty;
        if(alpha == 0)
        {
            C[c_index] = 0;
        }
        else
        {
            int a_index;

            if(transA == rocblas_operation_none)
            {
                a_index = tx + ty * lda;
            }
            else
            {
                a_index = tx * lda + ty;
            }
            C[c_index] = alpha * A[a_index];
        }
    }
}

//  special case:
//  treat matrices as contiguous vectors because
//  m == lda && m == ldb && m == ldc && none == transA && none == transB
template <typename T>
static __device__ void geam_1D_device(
    rocblas_int size, T alpha, const T* __restrict__ A, T beta, const T* __restrict__ B, T* C)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tx < size)
    {
        if(alpha == 0 && beta == 0)
        {
            C[tx] = 0;
        }
        else
        {
            C[tx] = fma(beta, B[tx], alpha * A[tx]);
        }
    }
}

//  special case:
//  treat matrices as contiguous vectors, and only one matrix contributes, because
//  m == lda && m == ldb && m == ldc && none == transA && none == transB && (0 == alpha || 0 ==
//  beta)
template <typename T>
static __device__ void
geam_1D_2matrix_device(rocblas_int size, T alpha, const T* __restrict__ A, T* C)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tx < size)
    {
        if(alpha == 0)
        {
            C[tx] = 0;
        }
        else
        {
            C[tx] = alpha * A[tx];
        }
    }
}
// special case
// inplace
template <typename T>
static __device__ void geam_inplace_device(rocblas_operation transB,
                                           rocblas_int m,
                                           rocblas_int n,
                                           T alpha,
                                           T beta,
                                           const T* __restrict__ B,
                                           rocblas_int ldb,
                                           T* C,
                                           rocblas_int ldc)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        int b_index;
        int c_index = tx + ldc * ty;

        if(beta == 0)
        {
            C[c_index] = alpha * C[c_index];
        }
        else
        {
            if(transB == rocblas_operation_none)
            {
                b_index = tx + ty * ldb;
            }
            else
            {
                b_index = tx * ldb + ty;
            }
            if(alpha == 0)
            {
                C[c_index] = beta * B[b_index];
            }
            else
            {
                C[c_index] = fma(beta, B[b_index], alpha * C[c_index]);
            }
        }
    }
}
