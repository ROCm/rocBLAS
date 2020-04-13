/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "../blas1/rocblas_copy.hpp"
#include "handle.h"

/**
  *  Helper for the non-transpose case. Iterates through each diagonal
  *  and creates partial sums for each ty.
  */
template <rocblas_int DIM_Y, typename T>
__device__ T tbmvn_kernel_helper(rocblas_int ty,
                                 rocblas_int ind,
                                 bool        upper,
                                 bool        diag,
                                 rocblas_int m,
                                 rocblas_int k,
                                 const T*    A,
                                 rocblas_int lda,
                                 const T*    x_copy)
{
    T           res_A = 0.0;
    rocblas_int col   = ty; // ty defines the column of banded & regular matrix

    // Since the column is consistent, we can iterate up the diagonal
    for(col = ty; col < m; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = upper ? ind + (k - col) : ind - col;

        if(ind < m)
        {
            // Regular case, simply multiply
            if(row < k && row > 0)
            {
                res_A += (A[row + col * lda] * x_copy[col]);
            }
            else if(row == 0)
            {
                // If main diagonal && diag, don't reference matrix, assume 1.
                if(diag && (!upper || k == 0 && upper))
                    res_A += x_copy[col];
                else
                    res_A += (A[row + col * lda] * x_copy[col]);
            }
            else if(row == k)
            {
                // If diag, don't reference matrix, assume 1.
                if(diag && upper)
                    res_A += x_copy[col];
                else
                    res_A += (A[row + col * lda] * x_copy[col]);
            }
        }
    }
    return res_A;
}

/**
  *  Helper for the (conjugate-)transpose case. Iterates through each diagonal
  *  and creates partial sums for each ty.
  *  The conjugate basically switches A from upper -> lower or lower -> upper
  *  triangular matrix. Since A is compressed, the indexing changes, and we
  *  basically just iterate down columns.
  */
template <rocblas_int DIM_Y, typename T>
__device__ T tbmvt_kernel_helper(bool        CONJ,
                                 rocblas_int ty,
                                 rocblas_int ind,
                                 bool        upper,
                                 bool        diag,
                                 rocblas_int m,
                                 rocblas_int k,
                                 const T*    A,
                                 rocblas_int lda,
                                 const T*    x_copy)
{
    T           res_A = 0.0;
    rocblas_int row   = ty; // for transpose case, ty defines the row

    for(row = ty; row < lda; row += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int col = ind;

        if(col < m)
        {
            if(upper)
            {
                // Regular case
                rocblas_int min_row = k - col;
                if(row < k && row >= k - col && row != k)
                {
                    res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                              * x_copy[row - min_row]);
                }
                else if(row == k)
                {
                    // if main diagonal && diag then don't reference A, assume 1.
                    if(diag)
                        res_A += x_copy[row - min_row];
                    else
                        res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                                  * x_copy[row - min_row]);
                }
                else if(row > k)
                    break;
            }
            else
            {
                if(row <= k && row <= m - 1 - col && row > 0)
                {
                    res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                              * x_copy[row + col]);
                }
                else if(row == 0)
                {
                    if(diag)
                        res_A += x_copy[row + col];
                    else
                        res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                                  * x_copy[row + col]);
                }
                else if(row > k)
                    break;
            }
        }
    }
    return res_A;
}

/**
  *  A combined kernel to handle all tbmv cases (transpose, conjugate, normal).
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void tbmvx_kernel_calc(rocblas_operation transA,
                                  bool              upper,
                                  bool              diag,
                                  rocblas_int       m,
                                  rocblas_int       k,
                                  const T*          A,
                                  rocblas_int       lda,
                                  const T*          x_copy,
                                  T*                x,
                                  rocblas_int       incx)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    // Create "tilted" blocks. With the compaction, each diagonal,
    // (from top right to bottom left) is like a row in a normal
    // matrix, so the blocks are "tilted" to the right.
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;
    // Indexing is different for transpose/non-transpose case. To keep it clean
    // it's separated in two helper functions. They could potentially be combined
    // if more elegant logic is used.
    if(transA == rocblas_operation_none)
    {
        res_A = tbmvn_kernel_helper<DIM_Y>(ty, ind, upper, diag, m, k, A, lda, x_copy);
    }
    else
    {
        bool CONJ = transA == rocblas_operation_conjugate_transpose;
        res_A     = tbmvt_kernel_helper<DIM_Y>(CONJ, ty, ind, upper, diag, m, k, A, lda, x_copy);
    }
    // Store partial sums for the diagonal
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    ind       = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X && ind < m)
    {
        // Add the partial sums of each diagonal and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        // Update x.
        x[ind * incx] = (sdata[thread_id]);
    }
}

/**
  *  Loads pointers (in case of future batched versions) and launches
  *  the actual calculation kernel.
  *
  *  Summary of banded matrices:
  *  Two types of banded matrices exist, upper and lower. These matrices consist of
  *  the centre diagonal, along with 'k' sub-diagonals (if lower) or super-diagonals (if upper).
  *
  *  These matrices are then compacted into a banded storage format. For upper-triangular,
  *  the k'th super-diagonal resides on the right-hand side of the first row, k-1th on the second,
  *  etc, with the main diagonal on the k'th row.
  *
  *  Ex: (upper; m = 5; k = 2)
  *
  *  1 6 9 0 0              0 0 9 8 7
  *  0 2 7 8 0              0 6 7 8 9
  *  0 0 3 8 7     ---->    1 2 3 4 5
  *  0 0 0 4 9              0 0 0 0 0
  *  0 0 0 0 5              0 0 0 0 0
  *
  *  For lower-triangular, the main diagonal resides on the 0'th row, working up to the k'th
  *  sub-diagonal residing on the left-hand side of the k'th row.
  *
  *  Ex: (lower; m = 5; k = 2)
  *
  *  1 0 0 0 0              1 2 3 4 5
  *  6 2 0 0 0              6 7 8 9 0
  *  9 7 3 0 0     ---->    9 8 7 0 0
  *  0 8 8 4 0              0 0 0 0 0
  *  0 0 7 9 5              0 0 0 0 0
  *
  *  The empty parts of these sparse matrices are not to be touched. As can be seen, the column
  *  of each element is preserved in the compaction, and the diagonals are "pushed" upwards and
  *  reside on the same row as the other elements of the same diagonal.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V>
__global__ void tbmvx_kernel(rocblas_operation transA,
                             bool              upper,
                             bool              diag,
                             rocblas_int       m,
                             rocblas_int       k,
                             U                 Aa,
                             ptrdiff_t         shifta,
                             rocblas_int       lda,
                             rocblas_stride    strideA,
                             U                 xa_copy,
                             V                 xa,
                             ptrdiff_t         shiftx,
                             rocblas_int       incx,
                             rocblas_stride    stridex)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const auto* A      = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const auto* x_copy = load_ptr_batch(xa_copy, hipBlockIdx_y, 0, m);
    auto*       x      = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    tbmvx_kernel_calc<DIM_X, DIM_Y>(transA, upper, diag, m, k, A, lda, x_copy, x, incx);
}

/**
  *  First, makes a copy of 'x', then uses a modified gemv algorithm
  *  to perform x := transA(A) * x_copy
  *  x_copy should be of size sizeof(T) * m bytes * batch_count.
  *
  *  Here, U is either a `const T* const*` or a `const T*`
  *  V is either a `T*` or a `T* const*`
  */
template <typename U, typename V>
rocblas_status rocblas_tbmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       k,
                                     U                 A,
                                     rocblas_int       offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     V                 x,
                                     rocblas_int       offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     rocblas_int       batch_count,
                                     V                 x_copy)
{
    // quick return
    if(!m || !batch_count)
        return rocblas_status_success;

    // First we make a copy of x so we can avoid RAW race conditions in the kernel
    int  copy_blocks = (m - 1) / 256 + 1;
    dim3 copy_grid(copy_blocks, batch_count);
    dim3 copy_threads(256);

    rocblas_status status = rocblas_copy_template<false, 256>(
        handle, m, x, offsetx, incx, stridex, x_copy, 0, 1, m, batch_count);

    if(status != rocblas_status_success)
        return status;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;

    // (gemv) TBMVX_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    static constexpr int TBMVX_DIM_X = 64;
    static constexpr int TBMVX_DIM_Y = 16;
    rocblas_int          blocks      = (m - 1) / (TBMVX_DIM_X) + 1;
    dim3                 tbmvx_grid(blocks, batch_count);
    dim3                 tbmvx_threads(TBMVX_DIM_X, TBMVX_DIM_Y);
    const bool           trans = transA == rocblas_operation_none;
    const bool           conj  = transA == rocblas_operation_conjugate_transpose;

    // Launch a modified gemv kernel. The logic is similar to gemv just with modified
    // indices for the banded matrices.
    hipLaunchKernelGGL((tbmvx_kernel<TBMVX_DIM_X, TBMVX_DIM_Y>),
                       tbmvx_grid,
                       tbmvx_threads,
                       0,
                       handle->rocblas_stream,
                       transA,
                       uplo == rocblas_fill_upper,
                       diag == rocblas_diagonal_unit,
                       m,
                       k,
                       A,
                       offseta,
                       lda,
                       strideA,
                       (U)x_copy,
                       x,
                       shiftx,
                       incx,
                       stridex);

    return rocblas_status_success;
}

#endif
