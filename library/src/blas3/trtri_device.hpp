/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __TRTRI_DEVICE_HPP__
#define __TRTRI_DEVICE_HPP__

#include "utility.h"

template <rocblas_int IB, typename T>
__device__ void custom_trtri_device(rocblas_fill     uplo,
                                    rocblas_diagonal diag,
                                    rocblas_int      n,
                                    const T*         A,
                                    rocblas_int      lda,
                                    T*               invA,
                                    rocblas_int      ldinvA)
{
    // quick return
    if(n <= 0)
        return;

    int tx = hipThreadIdx_x;

    __shared__ T diag1[IB * IB];
    __shared__ T diag2[IB * IB];
    __shared__ T sA[IB * IB];
    __shared__ T temp[IB * IB];

    T*  diagP      = tx < n ? diag1 : (tx < 2 * n ? diag2 : sA);
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
                diagP[index + i * n] = i <= index ? A[Aoffset + index + i * lda] : 0.0f;
        }
        else
        { // transpose A in sA if upper
            for(int i = n - 1; i >= 0; i--)
            {
                diagP[(n - 1 - index) + (n - 1 - i) * n]
                    = i >= index ? A[Aoffset + index + i * lda] : 0.0f;
            }
        }
    }
    else if(tx < n * 3)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i < n; i++)
                diagP[index + i * n] = A[n + index + i * lda];
        }
        else
        { // transpose A in diag1 if upper
            for(int i = n - 1; i >= 0; i--)
                diagP[index + i * n] = A[n * lda + index + i * lda];
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
            { // notice this does not apply for complex
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
                reg += diagP[index + col * n] * diagP[col + col * n];

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
            T sum(0);
            for(int k = c; k < IB; k++)
                sum += sA[r + k * IB] * diag1[k + c * IB];
            temp[r + c * IB] = sum;
        }
    }
    else
    {
        if(tx < n * n)
        {
            T sum(0);
            for(int k = 0; k < c + 1; k++)
                sum += sA[r + k * IB] * diag2[(IB - 1 - k) + (IB - 1 - c) * IB];
            temp[r + c * IB] = sum;
        }
    }

    __syncthreads();

    if(uplo == rocblas_fill_lower)
    {
        if(tx < n * n)
        {
            T sum(0);
            for(int k = 0; k < r + 1; k++)
                sum += -1.0f * diag2[r + k * n] * temp[k + c * n];
            invA[n + r + c * ldinvA] = sum;
        }
    }
    else
    {
        if(tx < n * n)
        {
            T sum(0);
            for(int k = r; k < IB; k++)
                sum += -1.0f * diag1[(n - 1 - r) + (n - 1 - k) * n] * temp[k + c * n];
            invA[n * ldinvA + r + c * ldinvA] = sum;
        }
    }

    if(tx < 2 * n)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i <= index; i++)
                invA[AInvoffset + index + i * ldinvA] = diagP[index + i * n];
        }
        else
        { // transpose back to A from sA if upper
            for(int i = n - 1; i >= index; i--)
                invA[AInvoffset + index + i * ldinvA] = diagP[(n - 1 - index) + (n - 1 - i) * n];
        }
    }
}

template <rocblas_int NB, typename T>
__device__ void trtri_device(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      n,
                             const T*         A,
                             rocblas_int      lda,
                             T*               invA,
                             rocblas_int      ldinvA)
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
                sA[tx + i * n] = A[tx + i * lda];
        }
        else
        { // transpose A in sA if upper
            for(int i = n - 1; i >= tx; i--)
                sA[(n - 1 - tx) + (n - 1 - i) * n] = A[tx + i * lda];
        }
    }
    __syncthreads(); // if NB < 64, this synch can be avoided

    // invert the diagonal element
    if(tx < n)
    {
        // compute only diagonal element
        if(diag == rocblas_diagonal_unit)
            sA[tx + tx * n] = 1.0;
        else
        { // inverse the diagonal
            if(sA[tx + tx * n] == 0.0) // notice this does not apply for complex
                sA[tx + tx * n] = 1.0; // means the matrix is singular
            else
                sA[tx + tx * n] = 1.0 / sA[tx + tx * n];
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
                reg += sA[tx + step * n] * sA[step + col * n];
            __syncthreads(); // if NB < 64, this synch can be avoided on AMD Fiji
        }
        __syncthreads();
    }

    if(tx < n)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i <= tx; i++)
                invA[tx + i * ldinvA] = sA[tx + i * n];
        }
        else
        { // transpose back to A from sA if upper
            for(int i = n - 1; i >= tx; i--)
                invA[tx + i * ldinvA] = sA[(n - 1 - tx) + (n - 1 - i) * n];
        }
    }
}

// flag indicate whether write into A or invA
template <rocblas_int NB, typename T>
__global__ void trtri_small_kernel_strided_batched(rocblas_fill     uplo,
                                                   rocblas_diagonal diag,
                                                   rocblas_int      n,
                                                   const T*         A,
                                                   rocblas_int      lda,
                                                   rocblas_int      stride_A,
                                                   rocblas_int      sub_stride_A,
                                                   T*               invA,
                                                   rocblas_int      ldinvA,
                                                   rocblas_int      stride_invA,
                                                   rocblas_int      sub_stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A    = A + hipBlockIdx_x * sub_stride_A + hipBlockIdx_y * stride_A;
    T*       individual_invA = invA + hipBlockIdx_x * sub_stride_invA + hipBlockIdx_y * stride_invA;

    trtri_device<NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T>
__global__ void trtri_remainder_kernel_strided_batched(rocblas_fill     uplo,
                                                       rocblas_diagonal diag,
                                                       rocblas_int      n,
                                                       const T*         A,
                                                       rocblas_int      lda,
                                                       rocblas_int      stride_A,
                                                       rocblas_int      sub_stride_A,
                                                       T*               invA,
                                                       rocblas_int      ldinvA,
                                                       rocblas_int      stride_invA,
                                                       rocblas_int      sub_stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A    = A + hipBlockIdx_x * sub_stride_A + hipBlockIdx_y * stride_A;
    T*       individual_invA = invA + hipBlockIdx_x * sub_stride_invA + hipBlockIdx_y * stride_invA;

    trtri_device<2 * NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int IB, typename T>
__global__ void trtri_diagonal_kernel_strided_batched(rocblas_fill     uplo,
                                                      rocblas_diagonal diag,
                                                      rocblas_int      n,
                                                      const T*         A,
                                                      rocblas_int      lda,
                                                      rocblas_int      stride_A,
                                                      rocblas_int      sub_stride_A,
                                                      T*               invA,
                                                      rocblas_int      ldinvA,
                                                      rocblas_int      stride_invA,
                                                      rocblas_int      sub_stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A

    rocblas_int tiles        = n / IB / 2;
    const T*    individual_A = A + (IB * 2 * lda + IB * 2) * (hipBlockIdx_x % tiles)
                            + sub_stride_A * (hipBlockIdx_x / tiles) + hipBlockIdx_y * stride_A;
    T* individual_invA = invA + (IB * 2 * ldinvA + IB * 2) * (hipBlockIdx_x % tiles)
                         + sub_stride_invA * (hipBlockIdx_x / tiles) + hipBlockIdx_y * stride_invA;

    custom_trtri_device<IB>(uplo,
                            diag,
                            min(IB, n - (hipBlockIdx_x % tiles) * IB),
                            individual_A,
                            lda,
                            individual_invA,
                            ldinvA);
}

#endif // \include guard