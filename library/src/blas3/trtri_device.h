/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _TRTRI_DEVICE_H_
#define _TRTRI_DEVICE_H_

/*
 * ===========================================================================
 *    This file provide common device function for trtri routines
 * ===========================================================================
 */

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix  A

        inv(A);

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    diag      rocblas_diagonal.
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         rocblas_int.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[output]
    invA      pointer storing the inverse matrix A on the GPU.

    ********************************************************************/

template <typename T, rocblas_int NB>
__device__ void trtri_device(rocblas_fill uplo,
                             rocblas_diagonal diag,
                             rocblas_int n,
                             const T* A,
                             rocblas_int lda,
                             T* invA,
                             rocblas_int ldinvA)
{

    // quick return
    if(n <= 0)
        return;

    int tx = hipThreadIdx_x;

    __shared__ T sA[NB * NB];

    // read matrix A into shared memory, only need to read lower part
    // its inverse will overwrite the shared memory

    if(tx < n)
    {
        if(uplo == rocblas_fill_lower)
        {
            // compute only diagonal element
            for(int i = 0; i <= tx; i++)
            {
                sA[tx + i * n] = A[tx + i * lda];
            }
        }
        else
        { // transpose A in sA if upper
            for(int i = n - 1; i >= tx; i--)
            {
                sA[(n - 1 - tx) + (n - 1 - i) * n] = A[tx + i * lda];
            }
        }
    }
    __syncthreads(); // if NB < 64, this synch can be avoided

    // invert the diagonal element
    if(tx < n)
    {
        // compute only diagonal element
        if(diag == rocblas_diagonal_unit)
        {
            sA[tx + tx * n] = 1.0;
        }
        else
        { // inverse the diagonal
            if(sA[tx + tx * n] == 0.0)
            {                          // notice this does not apply for complex
                sA[tx + tx * n] = 1.0; // means the matrix is singular
            }
            else
            {
                sA[tx + tx * n] = 1.0 / sA[tx + tx * n];
            }
        }
    }
    __syncthreads(); // if NB < 64, this synch can be avoided on AMD Fiji

    // solve the inverse of A column by column, each inverse(A)' column will overwrite sA'column
    // which store A
    // this operation is safe
    for(int col = 0; col < n; col++)
    {

        T reg = 0;
        // use the diagonal one to update current column
        if(tx > col)
            reg += sA[tx + col * n] * sA[col + col * n];

        __syncthreads(); // if NB < 64, this synch can be avoided on AMD Fiji

        // in each column, it solves step, each step solve an inverse(A)[step][col]
        for(int step = col + 1; step < n; step++)
        {

            // only tx == step solve off-diagonal
            if(tx == step)
            {
                // solve the step row, off-diagonal elements, notice sA[tx][tx] is already inversed,
                // so multiply
                sA[tx + col * n] = (0 - reg) * sA[tx + tx * n];
            }

            __syncthreads(); // if NB < 64, this synch can be avoided on AMD Fiji

            // tx > step  update with (tx = step)'s result
            if(tx > step)
            {
                reg += sA[tx + step * n] * sA[step + col * n];
            }
            __syncthreads(); // if NB < 64, this synch can be avoided on AMD Fiji
        }
        __syncthreads();
    }

    if(tx < n)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i <= tx; i++)
            {
                invA[tx + i * lda] = sA[tx + i * n];
            }
        }
        else
        { // transpose back to A from sA if upper
            for(int i = n - 1; i >= tx; i--)
            {
                invA[tx + i * lda] = sA[(n - 1 - tx) + (n - 1 - i) * n];
            }
        }
    }
}

#define STRSM_BLOCK 192
#define DTRSM_BLOCK 128

#endif // _TRTRI_DEVICE_H_
