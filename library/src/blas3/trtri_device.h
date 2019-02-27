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
                invA[tx + i * ldinvA] = sA[tx + i * n];
            }
        }
        else
        { // transpose back to A from sA if upper
            for(int i = n - 1; i >= tx; i--)
            {
                invA[tx + i * ldinvA] = sA[(n - 1 - tx) + (n - 1 - i) * n];
            }
        }
    }
}

template <typename T, rocblas_int IB>
__device__ void custom_trtri_device(rocblas_fill uplo,
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

    __shared__ T diag1[IB * IB];
    __shared__ T diag2[IB * IB];
    __shared__ T sA[IB * IB];
    __shared__ T temp[IB * IB];

    T* diagP       = tx < n ? diag1 : (tx < 2 * n ? diag2 : sA);
    int Aoffset    = tx < n ? 0 : n * lda + n;
    int AInvoffset = tx < n ? 0 : n * ldinvA + n;
    int index      = tx < n ? tx : (tx < 2 * n ? tx - n : tx - 2 * n);
    int r          = tx % n;
    int c          = tx / n;

    // read matrix A into shared memory, only need to read lower part
    // its inverse will overwrite the shared memory

    if(tx < 2 * n)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i < n; i++)
            {
                diagP[index + i * n] = i <= index ? A[Aoffset + index + i * lda] : 0.0f;
            }
        }
        else
        { // transpose A in sA if upper
            for(int i = n - 1; i >= 0; i--)
            {
                diagP[(n - 1 - index) + (n - 1 - i) * n] =
                    i >= index ? A[Aoffset + index + i * lda] : 0.0f;
            }
        }
    }
    else if(tx < n * 3)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i < n; i++)
            {
                diagP[index + i * n] = A[n + index + i * lda];
            }
        }
        else
        { // transpose A in diag1 if upper
            for(int i = n - 1; i >= 0; i--)
            {
                diagP[index + i * n] = A[n * lda + index + i * lda];
            }
        }
    }

    __syncthreads(); // if IB < 64, this synch can be avoided - IB = 16 here

    // invert the diagonal element
    if(tx < 2 * n)
    {
        // compute only diagonal element
        if(diag == rocblas_diagonal_unit)
        {
            diagP[index + index * n] = 1.0;
        }
        else
        { // inverse the diagonal
            if(diagP[index + index * n] == 0.0)
            {                                   // notice this does not apply for complex
                diagP[index + index * n] = 1.0; // means the matrix is singular
            }
            else
            {
                diagP[index + index * n] = 1.0 / diagP[index + index * n];
            }
        }
    }

    __syncthreads(); // if IB < 64, this synch can be avoided on AMD Fiji

    // solve the inverse of A column by column, each inverse(A)' column will overwrite diag1'column
    // which store A
    // this operation is safe
    if(tx < 2 * n)
    {
        for(int col = 0; col < n; col++)
        {

            T reg = 0;
            // use the diagonal one to update current column
            if(index > col)
            {
                reg += diagP[index + col * n] * diagP[col + col * n];
            }

            // __syncthreads(); // if IB < 64, this synch can be avoided on AMD Fiji

            // in each column, it solves step, each step solve an inverse(A)[step][col]
            for(int step = col + 1; step < n; step++)
            {

                // only tx == step solve off-diagonal
                if(index == step)
                {
                    // solve the step row, off-diagonal elements, notice diag1[tx][tx] is already
                    // inversed,
                    // so multiply
                    diagP[index + col * n] = (0 - reg) * diagP[index + index * n];
                }

                // __syncthreads(); // if IB < 64, this synch can be avoided on AMD Fiji

                // tx > step  update with (tx = step)'s result
                if(index > step)
                {
                    reg += diagP[index + step * n] * diagP[step + col * n];
                }
                // __syncthreads(); // if IB < 64, this synch can be avoided on AMD Fiji
            }
            // __syncthreads();
        }
    }

    __syncthreads();

    if(uplo == rocblas_fill_lower)
    {
        if(tx < n * n)
        {
            T sum = 0.0f;
            for(int k = c; k < IB; k++)
            {
                sum += sA[r + k * IB] * diag1[k + c * IB];
            }
            temp[r + c * IB] = sum;
        }
    }
    else
    {
        if(tx < n * n)
        {
            T sum = 0.0f;
            for(int k = 0; k < c + 1; k++)
            {
                sum += sA[r + k * IB] * diag2[(IB - 1 - k) + (IB - 1 - c) * IB];
            }
            temp[r + c * IB] = sum;
        }
    }

    __syncthreads();

    if(uplo == rocblas_fill_lower)
    {
        if(tx < n * n)
        {
            T sum = 0.0f;
            for(int k = 0; k < r + 1; k++)
            {
                sum += -1.0f * diag2[r + k * n] * temp[k + c * n];
            }
            invA[n + r + c * ldinvA] = sum;
        }
    }
    else
    {
        if(tx < n * n)
        {
            T sum = 0.0f;
            for(int k = r; k < IB; k++)
            {
                sum += -1.0f * diag1[(n - 1 - r) + (n - 1 - k) * n] * temp[k + c * n];
            }
            invA[n * ldinvA + r + c * ldinvA] = sum;
        }
    }

    if(tx < 2 * n)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i <= index; i++)
            {
                invA[AInvoffset + index + i * ldinvA] = diagP[index + i * n];
            }
        }
        else
        { // transpose back to A from sA if upper
            for(int i = n - 1; i >= index; i--)
            {
                invA[AInvoffset + index + i * ldinvA] = diagP[(n - 1 - index) + (n - 1 - i) * n];
            }
        }
    }
}

#endif // _TRTRI_DEVICE_H_
