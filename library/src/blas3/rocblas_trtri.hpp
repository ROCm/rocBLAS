/* ************************************************************************
* Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
* ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "gemm.hpp"

template <typename U, typename V>
inline rocblas_status rocblas_trtri_arg_check(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              U                A,
                                              rocblas_int      lda,
                                              V                invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      batch_count)
{

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || n < 0 || lda < n || ldinvA < n)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!n || !batch_count)
        return rocblas_status_success;

    if(!A || !invA)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <rocblas_int IB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_custom_trtri_device(rocblas_fill     uplo,
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

    int tx = threadIdx.x;

    __shared__ T diag1[IB * IB];
    __shared__ T diag2[IB * IB];
    __shared__ T sA[IB * IB];
    __shared__ T temp[IB * IB];

    T*  diagP = tx < n ? diag1 : (tx < 2 * n ? diag2 : sA);
    int index = tx < n ? tx : (tx < 2 * n ? tx - n : tx - 2 * n);
    int r     = tx % n;
    int c     = tx / n;

    // read matrix A into shared memory, only need to read lower part
    // its inverse will overwrite the shared memory

    if(tx < 2 * n)
    {
        int Aoffset = tx < n ? 0 : n * lda + n;

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
        int AInvoffset = tx < n ? 0 : n * ldinvA + n;

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
ROCBLAS_KERNEL_ILF void rocblas_trtri_device(rocblas_fill     uplo,
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

    int tx = threadIdx.x;

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

// return the number of elements in a NxN matrix that do not belong to the triangular region
constexpr size_t rocblas_num_non_tri_elements(size_t n)
{
    return n * (n - 1) / 2;
}

template <typename T>
ROCBLAS_KERNEL_ILF void rocblas_tritri_fill_upper(rocblas_stride offset,
                                                  size_t         idx,
                                                  rocblas_int    n,
                                                  rocblas_int    lda,
                                                  rocblas_stride sub_stride_A,
                                                  T              value,
                                                  T*             A)
{
    rocblas_int row = n - 2 - floor(sqrt(4 * n * (n - 1) - 7 - 8 * idx) / 2.0 - 0.5);
    rocblas_int col = idx + row + 1 - n * (n - 1) / 2 + (n - row) * (n - row - 1) / 2;

    size_t final_offset = offset * sub_stride_A + (row * lda) + col;

    A[final_offset] = value;
}

template <typename T>
ROCBLAS_KERNEL_ILF void rocblas_tritri_fill_lower(
    rocblas_stride offset, size_t idx, rocblas_int lda, rocblas_stride sub_stride_A, T value, T* A)
{
    rocblas_int row = (rocblas_int)((-1 + sqrt(8 * idx + 1)) / 2);
    rocblas_int col = idx - row * (row + 1) / 2;

    size_t final_offset = offset * sub_stride_A + ((row + 1) * lda) + col;

    A[final_offset] = value;
}

template <rocblas_int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_trtri_fill(rocblas_handle handle,
                   rocblas_fill   uplo,
                   rocblas_int    n,
                   rocblas_stride num_zero_elem,
                   rocblas_int    lda,
                   rocblas_stride sub_stride_A,
                   U              A,
                   rocblas_stride offset_A,
                   rocblas_stride stride_A,
                   rocblas_int    sub_batch_count)
{
    // number of elements in a given matrix that will be zeroed
    size_t num_elements_total_to_zero = num_zero_elem * sub_batch_count;
    size_t tx                         = size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    while(tx < num_elements_total_to_zero)
    {
        // determine which matrix in batch we're working on
        size_t offset = tx / num_zero_elem;
        // determine local matrix index
        size_t idx = tx % num_zero_elem;

        T* aptr = load_ptr_batch(A, blockIdx.y, offset_A, stride_A);

        if(uplo == rocblas_fill_upper)
            rocblas_tritri_fill_lower(offset, idx, lda, sub_stride_A, T(0), aptr);
        else if(uplo == rocblas_fill_lower)
            rocblas_tritri_fill_upper(offset, idx, n, lda, sub_stride_A, T(0), aptr);

        tx += size_t(blockDim.x) * gridDim.x;
    }
}

// flag indicate whether write into A or invA
template <rocblas_int NB, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_trtri_small_kernel(rocblas_fill     uplo,
                           rocblas_diagonal diag,
                           rocblas_int      n,
                           U                A,
                           rocblas_stride   offset_A,
                           rocblas_int      lda,
                           rocblas_stride   stride_A,
                           rocblas_stride   sub_stride_A,
                           V                invA,
                           rocblas_stride   offset_invA,
                           rocblas_int      ldinvA,
                           rocblas_stride   stride_invA,
                           rocblas_stride   sub_stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A
        = load_ptr_batch(A, blockIdx.y, offset_A, stride_A) + blockIdx.x * sub_stride_A;
    T* individual_invA
        = load_ptr_batch(invA, blockIdx.y, offset_invA, stride_invA) + blockIdx.x * sub_stride_invA;

    rocblas_trtri_device<NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T, typename U, typename V>
ROCBLAS_KERNEL_NO_BOUNDS rocblas_trtri_remainder_kernel(rocblas_fill     uplo,
                                                        rocblas_diagonal diag,
                                                        rocblas_int      n,
                                                        U                A,
                                                        rocblas_stride   offset_A,
                                                        rocblas_int      lda,
                                                        rocblas_stride   stride_A,
                                                        rocblas_stride   sub_stride_A,
                                                        V                invA,
                                                        rocblas_stride   offset_invA,
                                                        rocblas_int      ldinvA,
                                                        rocblas_stride   stride_invA,
                                                        rocblas_stride   sub_stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A
        = load_ptr_batch(A, blockIdx.y, offset_A, stride_A) + blockIdx.x * sub_stride_A;
    T* individual_invA
        = load_ptr_batch(invA, blockIdx.y, offset_invA, stride_invA) + blockIdx.x * sub_stride_invA;

    rocblas_trtri_device<2 * NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T, typename U, typename V>
rocblas_status rocblas_trtri_small(rocblas_handle   handle,
                                   rocblas_fill     uplo,
                                   rocblas_diagonal diag,
                                   rocblas_int      n,
                                   U                A,
                                   rocblas_stride   offset_A,
                                   rocblas_int      lda,
                                   rocblas_stride   stride_A,
                                   rocblas_stride   sub_stride_A,
                                   V                invA,
                                   rocblas_stride   offset_invA,
                                   rocblas_int      ldinvA,
                                   rocblas_stride   stride_invA,
                                   rocblas_stride   sub_stride_invA,
                                   rocblas_int      batch_count,
                                   rocblas_int      sub_batch_count)
{
    if(n > NB)
        return rocblas_status_not_implemented;

    static constexpr size_t blockSize = 128;
    size_t tri_elements_to_zero       = rocblas_num_non_tri_elements(n) * sub_batch_count;
    size_t numBlocks                  = (tri_elements_to_zero + blockSize - 1) / blockSize;

    hipLaunchKernelGGL((rocblas_trtri_fill<blockSize, T>),
                       dim3(numBlocks, batch_count, 1),
                       dim3(blockSize),
                       0,
                       handle->get_stream(),
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       rocblas_num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       offset_invA,
                       0,
                       sub_batch_count);

    dim3 grid(sub_batch_count, batch_count);
    dim3 threads(NB);

    hipLaunchKernelGGL((rocblas_trtri_small_kernel<NB, T>),
                       grid,
                       threads,
                       0,
                       handle->get_stream(),
                       uplo,
                       diag,
                       n,
                       A,
                       offset_A,
                       lda,
                       stride_A,
                       sub_stride_A,
                       invA,
                       offset_invA,
                       ldinvA,
                       stride_invA,
                       sub_stride_invA);

    return rocblas_status_success;
}

template <rocblas_int IB, typename T, typename U, typename V>
ROCBLAS_KERNEL(IB* IB)
rocblas_trtri_diagonal_kernel(rocblas_fill     uplo,
                              rocblas_diagonal diag,
                              rocblas_int      n,
                              U                A,
                              rocblas_stride   offset_A,
                              rocblas_int      lda,
                              rocblas_stride   stride_A,
                              rocblas_stride   sub_stride_A,
                              V                invA,
                              rocblas_stride   offset_invA,
                              rocblas_int      ldinvA,
                              rocblas_stride   stride_invA,
                              rocblas_stride   sub_stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A

    rocblas_int tiles        = n / IB / 2;
    const T*    individual_A = load_ptr_batch(A, blockIdx.y, offset_A, stride_A)
                            + (IB * 2 * lda + IB * 2) * (blockIdx.x % tiles)
                            + sub_stride_A * (blockIdx.x / tiles);
    T* individual_invA = load_ptr_batch(invA, blockIdx.y, offset_invA, stride_invA)
                         + (IB * 2 * ldinvA + IB * 2) * (blockIdx.x % tiles)
                         + sub_stride_invA * (blockIdx.x / tiles);

    auto rem = n - (blockIdx.x % tiles) * IB;
    rocblas_custom_trtri_device<IB>(
        uplo, diag, rem < IB ? rem : IB, individual_A, lda, individual_invA, ldinvA);
}

// compute square block of invA
template <bool BATCHED, bool STRIDED, typename T, typename U, typename V>
rocblas_status rocblas_trtri_gemm_block(rocblas_handle handle,
                                        rocblas_int    M,
                                        rocblas_int    N,
                                        U              A,
                                        rocblas_int    ld_A,
                                        rocblas_stride stride_A,
                                        rocblas_stride sub_stride_A,
                                        U              invAg1,
                                        U              invAg2a,
                                        V              invAg2c,
                                        rocblas_int    ld_invA,
                                        rocblas_stride stride_invA,
                                        rocblas_stride sub_stride_invA,
                                        V              C,
                                        rocblas_int    ld_C,
                                        rocblas_stride stride_C,
                                        rocblas_stride sub_stride_C,
                                        rocblas_int    batch_count,
                                        rocblas_int    sub_blocks,
                                        rocblas_stride offset_A       = 0,
                                        rocblas_int    offset_invAg1  = 0,
                                        rocblas_int    offset_invAg2a = 0,
                                        rocblas_int    offset_invAg2c = 0,
                                        rocblas_stride offset_C       = 0)
{
    std::unique_ptr<T*[]> host_A;
    std::unique_ptr<T*[]> host_invAg1;
    std::unique_ptr<T*[]> host_invAg2a;
    std::unique_ptr<T*[]> host_invAg2c;
    std::unique_ptr<T*[]> host_C;

    if(BATCHED)
    {
        host_A       = std::make_unique<T*[]>(batch_count);
        host_invAg1  = std::make_unique<T*[]>(batch_count);
        host_invAg2a = std::make_unique<T*[]>(batch_count);
        host_invAg2c = std::make_unique<T*[]>(batch_count);
        host_C       = std::make_unique<T*[]>(batch_count);

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&host_A[0], A, batch_count * sizeof(T*), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&host_invAg1[0], invAg1, batch_count * sizeof(T*), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&host_invAg2a[0], invAg2a, batch_count * sizeof(T*), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&host_invAg2c[0], invAg2c, batch_count * sizeof(T*), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&host_C[0], C, batch_count * sizeof(T*), hipMemcpyDeviceToHost));
    }

    rocblas_status status       = rocblas_status_success;
    static const T one          = T(1);
    static const T zero         = T(0);
    static const T negative_one = T(-1);

    // first batched gemm compute C = A21*invA11 (lower) or C = A12*invA22 (upper)
    // distance between each invA11 or invA22 is sub_stride_invA, sub_stride_A for each A21 or A12, C
    // of size IB * IB
    for(int b = 0; b < batch_count; b++)
    {
        const T *aptr, *invAg1ptr, *invAg2ptr;
        T *      cptr, *invAg2cptr;

        if(BATCHED)
        {
            aptr       = load_ptr_batch(&host_A[0], b, offset_A, stride_A);
            invAg1ptr  = load_ptr_batch(&host_invAg1[0], b, offset_invAg1, stride_invA);
            invAg2ptr  = load_ptr_batch(&host_invAg2a[0], b, offset_invAg2a, stride_invA);
            cptr       = load_ptr_batch(&host_C[0], b, offset_C, stride_C);
            invAg2cptr = load_ptr_batch(&host_invAg2c[0], b, offset_invAg2c, stride_invA);
        }
        else
        {
            aptr       = load_ptr_batch(A, b, offset_A, stride_A);
            invAg1ptr  = load_ptr_batch(invAg1, b, offset_invAg1, stride_invA);
            invAg2ptr  = load_ptr_batch(invAg2a, b, offset_invAg2a, stride_invA);
            cptr       = load_ptr_batch(C, b, offset_C, stride_C);
            invAg2cptr = load_ptr_batch(invAg2c, b, offset_invAg2c, stride_invA);
        }

        // We are naively iterating through the batches, and uses sub-batches in a strided_batched style.
        status = rocblas_internal_gemm_template<false>(handle,
                                                       rocblas_operation_none,
                                                       rocblas_operation_none,
                                                       M,
                                                       N,
                                                       N,
                                                       &one,
                                                       aptr,
                                                       0,
                                                       ld_A,
                                                       sub_stride_A,
                                                       invAg1ptr,
                                                       0,
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       &zero,
                                                       cptr,
                                                       0,
                                                       ld_C,
                                                       sub_stride_C,
                                                       sub_blocks);

        if(status != rocblas_status_success)
            break;

        // second batched gemm compute  invA21 = -invA22 * C (lower) or invA12 = -invA11*C (upper)
        // distance between each invA21 or invA12 is stride_invA,
        status = rocblas_internal_gemm_template<false>(handle,
                                                       rocblas_operation_none,
                                                       rocblas_operation_none,
                                                       M,
                                                       N,
                                                       M,
                                                       &negative_one,
                                                       invAg2ptr,
                                                       0,
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       cptr,
                                                       0,
                                                       ld_C,
                                                       sub_stride_C,
                                                       &zero,
                                                       invAg2cptr,
                                                       0,
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       sub_blocks);
        if(status != rocblas_status_success)
            break;
    }

    return status;
}

template <rocblas_int NB, bool BATCHED, bool STRIDED, typename T, typename U, typename V>
rocblas_status rocblas_trtri_large(rocblas_handle   handle,
                                   rocblas_fill     uplo,
                                   rocblas_diagonal diag,
                                   rocblas_int      n,
                                   U                A,
                                   rocblas_stride   offset_Ain,
                                   rocblas_int      lda,
                                   rocblas_stride   stride_A,
                                   rocblas_stride   sub_stride_Ain,
                                   V                invA,
                                   rocblas_stride   offset_invAin,
                                   rocblas_int      ldinvA,
                                   rocblas_stride   stride_invA,
                                   rocblas_stride   sub_stride_invAin,
                                   rocblas_int      batch_count,
                                   rocblas_int      sub_batch_count,
                                   V                w_C_tmp)
{
    dim3 grid_trtri(n / NB / 2 * sub_batch_count, batch_count);
    dim3 threads(NB * NB);

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA - Only deals with maximum even and complete NBxNB diagonals

    hipLaunchKernelGGL((rocblas_trtri_diagonal_kernel<NB, T>),
                       grid_trtri,
                       threads,
                       0,
                       handle->get_stream(),
                       uplo,
                       diag,
                       n,
                       A,
                       offset_Ain,
                       lda,
                       stride_A,
                       sub_stride_Ain,
                       invA,
                       offset_invAin,
                       ldinvA,
                       stride_invA,
                       sub_stride_invAin);

    rocblas_int remainder = n - (n / NB / 2) * 2 * NB;
    if(remainder > 0)
    {
        dim3 grid_remainder(sub_batch_count, batch_count);
        dim3 threads_remainder(remainder);

        rocblas_int offset_A2    = (n - remainder) + (n - remainder) * lda + offset_Ain;
        rocblas_int offset_invA2 = (n - remainder) + (n - remainder) * ldinvA + offset_invAin;
        hipLaunchKernelGGL((rocblas_trtri_remainder_kernel<NB, T>),
                           grid_remainder,
                           threads_remainder,
                           0,
                           handle->get_stream(),
                           uplo,
                           diag,
                           remainder,
                           A,
                           offset_A2,
                           lda,
                           stride_A,
                           sub_stride_Ain,
                           invA,
                           offset_invA2,
                           ldinvA,
                           stride_invA,
                           sub_stride_invAin);
    }

    if(n <= 2 * NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    static constexpr size_t sub_block_size = 128;
    size_t tri_elements_to_zero            = rocblas_num_non_tri_elements(n) * sub_batch_count;
    size_t num_sub_blocks = (tri_elements_to_zero + sub_block_size - 1) / sub_block_size;

    hipLaunchKernelGGL((rocblas_trtri_fill<sub_block_size, T>),
                       dim3(num_sub_blocks, batch_count, 1),
                       dim3(sub_block_size),
                       0,
                       handle->get_stream(),
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       rocblas_num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       offset_invAin,
                       stride_invA,
                       sub_batch_count);

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    static constexpr auto IB = NB * 2;
    rocblas_int           current_n;

    for(current_n = IB; current_n * 2 <= n; current_n *= 2)
    {
        rocblas_int tiles_per_batch = n / current_n / 2;
        if(tiles_per_batch > sub_batch_count)
        {
            for(int i = 0; i < sub_batch_count; i++)
            {
                rocblas_int offset_A
                    = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_Ain
                                                  : current_n * lda + i * sub_stride_Ain);
                rocblas_int offset_invA1
                    = (uplo == rocblas_fill_lower
                           ? 0 + i * sub_stride_invAin
                           : current_n * ldinvA + current_n + i * sub_stride_invAin);
                rocblas_int offset_invA2
                    = (uplo == rocblas_fill_lower
                           ? current_n * ldinvA + current_n + i * sub_stride_invAin
                           : 0 + i * sub_stride_invAin);
                rocblas_int offset_invA3
                    = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_invAin
                                                  : current_n * ldinvA + i * sub_stride_invAin);
                rocblas_int offset_C
                    = (uplo == rocblas_fill_lower
                           ? (n - current_n) * ldinvA + i * sub_stride_invAin
                           : (n - current_n * tiles_per_batch) + i * sub_stride_invAin);

                offset_A += offset_Ain;
                offset_invA1 += offset_invAin;
                offset_invA2 += offset_invAin;
                offset_invA3 += offset_invAin;
                offset_C += offset_invAin;

                rocblas_trtri_gemm_block<BATCHED, STRIDED, T>(handle,
                                                              current_n,
                                                              current_n,
                                                              (U)A,
                                                              lda,
                                                              stride_A,
                                                              2 * current_n * lda + 2 * current_n,
                                                              (U)invA,
                                                              (U)invA,
                                                              (V)invA,
                                                              ldinvA,
                                                              stride_invA,
                                                              2 * current_n * ldinvA
                                                                  + 2 * current_n,
                                                              (V)invA,
                                                              ldinvA,
                                                              stride_invA,
                                                              current_n,
                                                              batch_count,
                                                              tiles_per_batch,
                                                              offset_A,
                                                              offset_invA1,
                                                              offset_invA2,
                                                              offset_invA3,
                                                              offset_C);
            }
        }
        else
        {
            for(int i = 0; i < tiles_per_batch; i++)
            {
                rocblas_int sub_stride_A2    = (2 * current_n * lda + 2 * current_n);
                rocblas_int sub_stride_invA2 = (2 * current_n * ldinvA + 2 * current_n);

                rocblas_int offset_A
                    = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_A2
                                                  : current_n * lda + i * sub_stride_A2);

                rocblas_int offset_invA1
                    = (uplo == rocblas_fill_lower
                           ? 0 + i * sub_stride_invA2
                           : current_n * ldinvA + current_n + i * sub_stride_invA2);

                rocblas_int offset_invA2
                    = (uplo == rocblas_fill_lower
                           ? current_n * ldinvA + current_n + i * sub_stride_invA2
                           : 0 + i * sub_stride_invA2);

                rocblas_int offset_invA3
                    = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_invA2
                                                  : current_n * ldinvA + i * sub_stride_invA2);

                rocblas_int offset_C = (uplo == rocblas_fill_lower
                                            ? (n - current_n) * ldinvA + i * current_n
                                            : (n - current_n * tiles_per_batch) + i * current_n);

                offset_A += offset_Ain;
                offset_invA1 += offset_invAin;
                offset_invA2 += offset_invAin;
                offset_invA3 += offset_invAin;
                offset_C += offset_invAin;

                rocblas_trtri_gemm_block<BATCHED, STRIDED, T>(handle,
                                                              current_n,
                                                              current_n,
                                                              (U)A,
                                                              lda,
                                                              stride_A,
                                                              sub_stride_Ain,
                                                              (U)invA,
                                                              (U)invA,
                                                              (V)invA,
                                                              ldinvA,
                                                              stride_invA,
                                                              sub_stride_invAin,
                                                              (V)invA,
                                                              ldinvA,
                                                              stride_invA,
                                                              sub_stride_invAin,
                                                              batch_count,
                                                              sub_batch_count,
                                                              offset_A,
                                                              offset_invA1,
                                                              offset_invA2,
                                                              offset_invA3,
                                                              offset_C);
            }
        }
    }

    hipLaunchKernelGGL((rocblas_trtri_fill<sub_block_size, T>),
                       dim3(num_sub_blocks, batch_count, 1),
                       dim3(sub_block_size),
                       0,
                       handle->get_stream(),
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       rocblas_num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       offset_invAin,
                       stride_invA,
                       sub_batch_count);

    // Set remainder to the closest power of 2 <= to the leftover block size
    // Odd remainder will handle the rest, including any parts missed
    // at the end of the previous for loop
    remainder = (n / IB) * IB - current_n;
    if(!rocblas_is_po2(remainder))
        remainder = rocblas_previous_po2(remainder);
    rocblas_int oddRemainder = n - current_n - remainder;

    // For some large sizes (eg. n = 736), this needs to be iterated over
    // more than once. This is because the for loop above leaves remainder sections,
    // and in some cases this happens with multiple sizes.
    if(remainder > 0)
    {
        rocblas_int offset_A = (uplo == rocblas_fill_lower ? current_n : current_n * lda);
        rocblas_int offset_invA1
            = (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n);
        rocblas_int offset_invA2
            = (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0);
        rocblas_int offset_invA3 = (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA);

        offset_A += offset_Ain;
        offset_invA1 += offset_invAin;
        offset_invA2 += offset_invAin;
        offset_invA3 += offset_invAin;

        rocblas_trtri_gemm_block<BATCHED, STRIDED, T>(
            handle,
            uplo == rocblas_fill_lower ? remainder : current_n,
            uplo == rocblas_fill_lower ? current_n : remainder,
            (U)A,
            lda,
            stride_A,
            0,
            (U)invA,
            (U)invA,
            (V)invA,
            ldinvA,
            stride_invA,
            0,
            (V)w_C_tmp,
            uplo == rocblas_fill_lower ? remainder : current_n,
            0,
            0,
            batch_count,
            1,
            offset_A,
            offset_invA1,
            offset_invA2,
            offset_invA3,
            0);
    }

    while(oddRemainder)
    {
        current_n            = n - oddRemainder;
        rocblas_int offset_A = (uplo == rocblas_fill_lower ? current_n : current_n * lda);
        rocblas_int offset_invA1
            = (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n);
        rocblas_int offset_invA2
            = (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0);
        rocblas_int offset_invA3 = (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA);

        offset_A += offset_Ain;
        offset_invA1 += offset_invAin;
        offset_invA2 += offset_invAin;
        offset_invA3 += offset_invAin;

        rocblas_trtri_gemm_block<BATCHED, STRIDED, T>(
            handle,
            uplo == rocblas_fill_lower ? oddRemainder : current_n,
            uplo == rocblas_fill_lower ? current_n : oddRemainder,
            (U)A,
            lda,
            stride_A,
            0,
            (U)invA,
            (U)invA,
            (V)invA,
            ldinvA,
            stride_invA,
            0,
            (V)w_C_tmp,
            uplo == rocblas_fill_lower ? oddRemainder : current_n,
            0,
            0,
            batch_count,
            1,
            offset_A,
            offset_invA1,
            offset_invA2,
            offset_invA3,
            0);

        // If oddRemainder > IB and is not a power of 2, then there's
        // still some leftover, so calculate new remainders.
        if(!rocblas_is_po2(oddRemainder) && oddRemainder > IB)
        {
            oddRemainder = rocblas_previous_po2(oddRemainder);
            oddRemainder = n - current_n - oddRemainder;
        }
        else
        {
            // We're done everything.
            oddRemainder = 0;
        }
    }

    return rocblas_status_success;
}

template <rocblas_int NB>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t rocblas_internal_trtri_temp_size(rocblas_int n,
                                                                         rocblas_int batch_count)
{
    rocblas_int IB   = NB * 2;
    size_t      size = 0;
    if(n > IB && batch_count > 0)
    {
        rocblas_int current_n = IB;
        while(current_n * 2 <= n)
            current_n *= 2;
        rocblas_int remainder = (n / IB) * IB - current_n;
        if(!rocblas_is_po2(remainder))
            remainder = rocblas_previous_po2(remainder);
        rocblas_int oddRemainder = n - current_n - remainder;

        size_t sizeRemainder = remainder ? remainder * current_n : 0;
        size_t sizeOdd       = 0;

        while(oddRemainder)
        {
            current_n         = n - oddRemainder;
            size_t curSizeOdd = oddRemainder * (n - oddRemainder);
            sizeOdd           = sizeOdd > curSizeOdd ? sizeOdd : curSizeOdd;

            if(!rocblas_is_po2(oddRemainder) && oddRemainder > IB)
            {
                oddRemainder = rocblas_previous_po2(oddRemainder);
                oddRemainder = n - current_n - oddRemainder;
            }
            else
            {
                oddRemainder = 0;
            }
        }

        if(sizeRemainder || sizeOdd)
            size = (sizeRemainder > sizeOdd ? sizeRemainder : sizeOdd) * batch_count;
    }
    return size;
}

template <rocblas_int NB, bool BATCHED, bool STRIDED, typename T, typename U, typename V>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trtri_template(rocblas_handle   handle,
                                    rocblas_fill     uplo,
                                    rocblas_diagonal diag,
                                    rocblas_int      n,
                                    U                A,
                                    rocblas_stride   offset_A,
                                    rocblas_int      lda,
                                    rocblas_stride   stride_A,
                                    rocblas_stride   sub_stride_A,
                                    V                invA,
                                    rocblas_stride   offset_invA,
                                    rocblas_int      ldinvA,
                                    rocblas_stride   stride_invA,
                                    rocblas_stride   sub_stride_invA,
                                    rocblas_int      batch_count,
                                    rocblas_int      sub_batch_count,
                                    V                w_C_tmp)
{
    if(!n || !sub_batch_count)
        return rocblas_status_success;

    if(n <= NB)
    {
        return rocblas_trtri_small<NB, T>(handle,
                                          uplo,
                                          diag,
                                          n,
                                          A,
                                          offset_A,
                                          lda,
                                          stride_A,
                                          sub_stride_A,
                                          invA,
                                          offset_invA,
                                          ldinvA,
                                          stride_invA,
                                          sub_stride_invA,
                                          batch_count,
                                          sub_batch_count);
    }
    else
    {
        return rocblas_trtri_large<NB, BATCHED, STRIDED, T>(handle,
                                                            uplo,
                                                            diag,
                                                            n,
                                                            A,
                                                            offset_A,
                                                            lda,
                                                            stride_A,
                                                            sub_stride_A,
                                                            invA,
                                                            offset_invA,
                                                            ldinvA,
                                                            stride_invA,
                                                            sub_stride_invA,
                                                            batch_count,
                                                            sub_batch_count,
                                                            w_C_tmp);
    }
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_trtri_check_numerics(const char*    function_name,
                                            rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            TConstPtr*     A,
                                            rocblas_int    lda,
                                            rocblas_stride stride_a,
                                            TPtr*          invA,
                                            rocblas_int    ldinvA,
                                            rocblas_stride stride_invA,
                                            rocblas_int    batch_count,
                                            const int      check_numerics,
                                            bool           is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              uplo,
                                                              rocblas_client_triangular_matrix,
                                                              n,
                                                              n,
                                                              A,
                                                              0,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          n,
                                                          n,
                                                          invA,
                                                          0,
                                                          ldinvA,
                                                          stride_invA,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    return check_numerics_status;
}
