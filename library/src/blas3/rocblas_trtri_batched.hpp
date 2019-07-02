/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_TRTRI_BATCHED_HPP_
#define _ROCBLAS_TRTRI_BATCHED_HPP_

#include "gemm.hpp"
#include "handle.h"

// return the number of elements in a NxN matrix that do not belong to the triangular region
constexpr size_t num_non_tri_elements(size_t n)
{
    return n * (n - 1) / 2;
}

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

template <typename T>
__device__ void rocblas_tritri_batched_fill_upper(
    size_t offset, size_t idx, rocblas_int n, rocblas_int lda, rocblas_int bsa, T value, T* A)
{
    rocblas_int row = n - 2 - floor(sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    rocblas_int col = idx + row + 1 - n * (n - 1) / 2 + (n - row) * (n - row - 1) / 2;

    size_t final_offset = offset * bsa + (row * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__device__ void rocblas_tritri_batched_fill_lower(
    size_t offset, size_t idx, rocblas_int lda, rocblas_int bsa, T value, T* A)
{
    rocblas_int row = (rocblas_int)((-1 + sqrt(8 * idx + 1)) / 2);
    rocblas_int col = idx - row * (row + 1) / 2;

    size_t final_offset = offset * bsa + ((row + 1) * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__global__ void rocblas_trtri_batched_fill(rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           rocblas_long   num_zero_elem,
                                           rocblas_int    lda,
                                           rocblas_int    bsa,
                                           T*             A,
                                           rocblas_int    batch_count)
{
    // number of elements in a given matrix that will be zeroed
    size_t num_elements_total_to_zero = num_zero_elem * batch_count;
    size_t tx                         = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    while(tx < num_elements_total_to_zero)
    {
        // determine which matrix in batch we're working on
        size_t offset = tx / num_zero_elem;
        // determine local matrix index
        size_t idx = tx % num_zero_elem;

        if(uplo == rocblas_fill_upper)
            rocblas_tritri_batched_fill_lower(offset, idx, lda, bsa, T(0), A);
        else if(uplo == rocblas_fill_lower)
            rocblas_tritri_batched_fill_upper(offset, idx, n, lda, bsa, T(0), A);

        tx += hipBlockDim_x * hipGridDim_x;
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
__global__ void trtri_small_kernel_batched(rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           const T*         A,
                                           rocblas_int      lda,
                                           rocblas_int      bsa,
                                           T*               invA,
                                           rocblas_int      ldinvA,
                                           rocblas_int      bsinvA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A    = A + hipBlockIdx_x * bsa;
    T*       individual_invA = invA + hipBlockIdx_x * bsinvA;

    trtri_device<NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T>
__global__ void trtri_remainder_kernel_batched(rocblas_fill     uplo,
                                               rocblas_diagonal diag,
                                               rocblas_int      n,
                                               const T*         A,
                                               rocblas_int      lda,
                                               rocblas_int      bsa,
                                               T*               invA,
                                               rocblas_int      ldinvA,
                                               rocblas_int      bsinvA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A    = A + hipBlockIdx_x * bsa;
    T*       individual_invA = invA + hipBlockIdx_x * bsinvA;

    trtri_device<2 * NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_small_batched(rocblas_handle   handle,
                                           rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           const T*         A,
                                           rocblas_int      lda,
                                           rocblas_int      bsa,
                                           T*               invA,
                                           rocblas_int      ldinvA,
                                           rocblas_int      bsinvA,
                                           rocblas_int      batch_count)
{
    if(n > NB)
    {
        printf("n is %d must be less than %d, will exit\n", n, NB);
        return rocblas_status_not_implemented;
    }

    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(rocblas_trtri_batched_fill,
                       dim3(numBlocks, 1, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       batch_count);

    dim3 grid(batch_count);
    dim3 threads(NB);

    hipLaunchKernelGGL(trtri_small_kernel_batched<NB>,
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       bsa,
                       invA,
                       ldinvA,
                       bsinvA);

    return rocblas_status_success;
}

template <rocblas_int IB, typename T>
__global__ void trtri_diagonal_kernel_batched(rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T*         A,
                                              rocblas_int      lda,
                                              rocblas_int      bsa,
                                              T*               invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      bsinvA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A

    rocblas_int tiles = n / IB / 2;
    const T*    individual_A
        = A + (IB * 2 * lda + IB * 2) * (hipBlockIdx_x % tiles) + bsa * (hipBlockIdx_x / tiles);
    T* individual_invA = invA + (IB * 2 * ldinvA + IB * 2) * (hipBlockIdx_x % tiles)
                         + bsinvA * (hipBlockIdx_x / tiles);

    custom_trtri_device<IB>(uplo,
                            diag,
                            min(IB, n - (hipBlockIdx_x % tiles) * IB),
                            individual_A,
                            lda,
                            individual_invA,
                            ldinvA);
}

// compute square block of invA
template <typename T>
rocblas_status trtri_strided_gemm_block(rocblas_handle handle,
                                        rocblas_int    M,
                                        rocblas_int    N,
                                        const T*       A,
                                        rocblas_int    ld_A,
                                        rocblas_int    stride_A,
                                        const T*       invAg1,
                                        const T*       invAg2a,
                                        T*             invAg2c,
                                        rocblas_int    ld_invA,
                                        rocblas_int    stride_invA,
                                        T*             C,
                                        rocblas_int    ld_C,
                                        rocblas_int    stride_C,
                                        rocblas_int    batch)
{
    rocblas_status     status;
    static constexpr T one          = 1;
    static constexpr T zero         = 0;
    static constexpr T negative_one = -1;

#ifndef NDEBUG
    printf("first batched gemm\n");
#endif
    // first batched gemm compute C = A21*invA11 (lower) or C = A12*invA22 (upper)
    // distance between each invA11 or invA22 is stride_invA,  stride_A for each A21 or A12, C
    // of size IB * IB
    status = rocblas_gemm_strided_batched_template(handle,
                                                   rocblas_operation_none,
                                                   rocblas_operation_none,
                                                   M,
                                                   N,
                                                   N,
                                                   &one,
                                                   (const T*)A,
                                                   ld_A,
                                                   stride_A,
                                                   (const T*)invAg1,
                                                   ld_invA,
                                                   stride_invA,
                                                   &zero,
                                                   (T*)C,
                                                   ld_C,
                                                   stride_C,
                                                   batch);

#ifndef NDEBUG
    printf("second batched gemm\n");
#endif
    // second batched gemm compute  invA21 = -invA22 * C (lower) or invA12 = -invA11*C (upper)
    // distance between each invA21 or invA12 is stride_invA,
    status = rocblas_gemm_strided_batched_template(handle,
                                                   rocblas_operation_none,
                                                   rocblas_operation_none,
                                                   M,
                                                   N,
                                                   M,
                                                   &negative_one,
                                                   (const T*)invAg2a,
                                                   ld_invA,
                                                   stride_invA,
                                                   (const T*)C,
                                                   ld_C,
                                                   stride_C,
                                                   &zero,
                                                   (T*)invAg2c,
                                                   ld_invA,
                                                   stride_invA,
                                                   batch);

    return status;
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_large_batched(rocblas_handle   handle,
                                           rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           const T*         A,
                                           rocblas_int      lda,
                                           rocblas_int      bsa,
                                           T*               invA,
                                           rocblas_int      ldinvA,
                                           rocblas_int      bsinvA,
                                           rocblas_int      batch_count,
                                           T*               C_tmp)
{
    dim3 grid_trtri(n / NB / 2 * batch_count);
    dim3 threads(NB * NB);

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA - Only deals with maximum even and complete NBxNB diagonals
    hipLaunchKernelGGL(trtri_diagonal_kernel_batched<NB>,
                       grid_trtri,
                       threads,
                       0,
                       handle->rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       bsa,
                       invA,
                       ldinvA,
                       bsinvA);

    rocblas_int remainder = n - (n / NB / 2) * 2 * NB;
    if(remainder > 0)
    {
        dim3 grid_remainder(batch_count);
        dim3 threads_remainder(remainder);

        hipLaunchKernelGGL(trtri_remainder_kernel_batched<NB>,
                           grid_remainder,
                           threads_remainder,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           diag,
                           remainder,
                           (const T*)A + (n - remainder) + (n - remainder) * lda,
                           lda,
                           bsa,
                           (T*)invA + (n - remainder) + (n - remainder) * ldinvA,
                           ldinvA,
                           bsinvA);
    }

    if(n <= 2 * NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(rocblas_trtri_batched_fill,
                       dim3(numBlocks, 1, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       batch_count);

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    static constexpr auto IB = NB * 2;
    rocblas_int           current_n;

    for(current_n = IB; current_n * 2 <= n; current_n *= 2)
    {
        rocblas_int tiles_per_batch = n / current_n / 2;

        if(tiles_per_batch > batch_count)
        {
            for(int i = 0; i < batch_count; i++)
            {
                trtri_strided_gemm_block(
                    handle,
                    current_n,
                    current_n,
                    (const T*)(A
                               + (uplo == rocblas_fill_lower ? current_n + i * bsa
                                                             : current_n * lda + i * bsa)),
                    lda,
                    2 * current_n * lda + 2 * current_n,
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? 0 + i * bsinvA
                                      : current_n * ldinvA + current_n + i * bsinvA)),
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? current_n * ldinvA + current_n + i * bsinvA
                                      : 0 + i * bsinvA)),
                    (T*)(invA
                         + (uplo == rocblas_fill_lower ? current_n + i * bsinvA
                                                       : current_n * ldinvA + i * bsinvA)),
                    ldinvA,
                    2 * current_n * ldinvA + 2 * current_n,
                    (T*)(invA
                         + (uplo == rocblas_fill_lower
                                ? (n - current_n) * ldinvA + i * bsinvA
                                : (n - current_n * tiles_per_batch) + i * bsinvA)),
                    ldinvA,
                    current_n,
                    tiles_per_batch);
            }
        }
        else
        {
            for(int i = 0; i < tiles_per_batch; i++)
            {
                rocblas_int stride_A, stride_invA;
                stride_A    = (2 * current_n * lda + 2 * current_n);
                stride_invA = (2 * current_n * ldinvA + 2 * current_n);
                trtri_strided_gemm_block(
                    handle,
                    current_n,
                    current_n,
                    (const T*)(A
                               + (uplo == rocblas_fill_lower ? current_n + i * stride_A
                                                             : current_n * lda + i * stride_A)),
                    lda,
                    bsa,
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? 0 + i * stride_invA
                                      : current_n * ldinvA + current_n + i * stride_invA)),
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? current_n * ldinvA + current_n + i * stride_invA
                                      : 0 + i * stride_invA)),
                    (T*)(invA
                         + (uplo == rocblas_fill_lower ? current_n + i * stride_invA
                                                       : current_n * ldinvA + i * stride_invA)),
                    ldinvA,
                    bsinvA,
                    (T*)(invA
                         + (uplo == rocblas_fill_lower
                                ? (n - current_n) * ldinvA + i * current_n
                                : (n - current_n * tiles_per_batch) + i * current_n)),
                    ldinvA,
                    bsinvA,
                    batch_count);
            }
        }
    }

    hipLaunchKernelGGL(rocblas_trtri_batched_fill,
                       dim3(numBlocks, 1, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       batch_count);

    remainder                = (n / NB) * NB - current_n - ((n / NB) % 2 == 0 ? 0 : NB);
    rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16

    if(remainder || oddRemainder)
    {
        if(remainder > 0)
        {
            trtri_strided_gemm_block(
                handle,
                uplo == rocblas_fill_lower ? remainder : current_n,
                uplo == rocblas_fill_lower ? current_n : remainder,
                (const T*)(A + (uplo == rocblas_fill_lower ? current_n : current_n * lda)),
                lda,
                bsa,
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n)),
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0)),
                (T*)(invA + (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA)),
                ldinvA,
                bsinvA,
                C_tmp,
                uplo == rocblas_fill_lower ? remainder : current_n,
                remainder * current_n,
                batch_count);
        }

        if(oddRemainder > 0) // solve small oddRemainder
        {
            current_n = n - oddRemainder;

            trtri_strided_gemm_block(
                handle,
                uplo == rocblas_fill_lower ? oddRemainder : current_n,
                uplo == rocblas_fill_lower ? current_n : oddRemainder,
                (const T*)(A + (uplo == rocblas_fill_lower ? current_n : current_n * lda)),
                lda,
                bsa,
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n)),
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0)),
                (T*)(invA + (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA)),
                ldinvA,
                bsinvA,
                C_tmp,
                uplo == rocblas_fill_lower ? oddRemainder : current_n,
                oddRemainder * current_n,
                batch_count);
        }
    }

    return rocblas_status_success;
}

template <rocblas_int NB>
constexpr size_t rocblas_trtri_batched_temp_size(rocblas_int n, rocblas_int batch_count)
{
    size_t size = 0;
    if(n > NB * 2 && batch_count > 0)
    {
        rocblas_int current_n = NB * 2;
        while(current_n * 2 <= n)
            current_n *= 2;
        rocblas_int remainder    = (n / NB) * NB - current_n - ((n / NB) % 2 == 0 ? 0 : NB);
        rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16
        if(remainder || oddRemainder)
            size = size_t(remainder ? remainder * current_n : oddRemainder * (n - remainder))
                   * batch_count;
    }
    return size;
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_batched_template(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T*         A,
                                              rocblas_int      lda,
                                              rocblas_int      bsa,
                                              T*               invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      bsinvA,
                                              rocblas_int      batch_count,
                                              T*               C_tmp)
{
    if(!n || !batch_count)
        return rocblas_status_success;
    if(n <= NB)
        return rocblas_trtri_small_batched<NB>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    else
        return rocblas_trtri_large_batched<NB>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count, C_tmp);
}

#endif // _ROCBLAS_TRTRI_BATCHED_HPP_
