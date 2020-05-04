/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GBMV_HPP__
#define __ROCBLAS_GBMV_HPP__
#include "../blas1/rocblas_copy.hpp"

/**
  *  Helper for the non-transpose case. Iterates through each diagonal
  *  and creates partial sums for each ty.
  */
template <rocblas_int DIM_Y, typename T>
__device__ T gbmvn_kernel_helper(rocblas_int ty,
                                 rocblas_int ind,
                                 rocblas_int m,
                                 rocblas_int n,
                                 rocblas_int kl,
                                 rocblas_int ku,
                                 const T*    A,
                                 rocblas_int lda,
                                 const T*    x,
                                 rocblas_int incx)
{
    T           res_A = 0.0;
    rocblas_int col   = ty; // ty defines the column of banded & regular matrix

    // Since the column is consistent, we can iterate up the diagonal
    for(col = ty; col < n; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = ind + (ku - col);

        if(ind < m)
        {
            if(row <= kl + ku && row >= 0)
            {
                if(row <= ku && col >= (ku - row) && col < (ku - row + m))
                {
                    res_A += (A[row + col * lda] * x[col * incx]);
                }
                if(row > ku && col < m - (row - ku))
                {
                    res_A += (A[row + col * lda] * x[col * incx]);
                }
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
__device__ T gbmvt_kernel_helper(bool        CONJ,
                                 rocblas_int ty,
                                 rocblas_int ind,
                                 rocblas_int m,
                                 rocblas_int n,
                                 rocblas_int kl,
                                 rocblas_int ku,
                                 const T*    A,
                                 rocblas_int lda,
                                 const T*    x,
                                 rocblas_int incx)
{
    T           res_A = 0.0;
    rocblas_int row   = ty; // for transpose case, ty defines the row

    for(row = ty; row < lda; row += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int col = ind;

        if(col < n)
        {
            if(row <= kl + ku && row >= 0)
            {
                if(row <= ku && col >= (ku - row) && col < (ku - row + m))
                {
                    res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                              * x[(row - ku + col) * incx]);
                }
                else if((row > ku && row <= kl + ku) && col < m - (row - ku))
                {
                    res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                              * x[(row - ku + col) * incx]);
                }
            }
        }
    }
    return res_A;
}

/**
  *  A combined kernel to handle all gbmv cases (transpose, conjugate, normal).
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void gbmvx_kernel_calc(rocblas_operation transA,
                                  rocblas_int       m,
                                  rocblas_int       n,
                                  rocblas_int       kl,
                                  rocblas_int       ku,
                                  T                 alpha,
                                  const T*          A,
                                  rocblas_int       lda,
                                  const T*          x,
                                  rocblas_int       incx,
                                  T                 beta,
                                  T*                y,
                                  rocblas_int       incy)
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
        res_A = gbmvn_kernel_helper<DIM_Y>(ty, ind, m, n, kl, ku, A, lda, x, incx);
    }
    else
    {
        bool CONJ = transA == rocblas_operation_conjugate_transpose;
        res_A     = gbmvt_kernel_helper<DIM_Y>(CONJ, ty, ind, m, n, kl, ku, A, lda, x, incx);
    }
    // Store partial sums for the diagonal
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    thread_id           = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    ind                 = hipBlockIdx_x * DIM_X + thread_id;
    rocblas_int max_ind = transA == rocblas_operation_none ? m : n;
    if(thread_id < DIM_X && ind < max_ind)
    {
        // Add the partial sums of each diagonal and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }
        if(ind < max_ind)
        {
            // Update y.
            if(beta != 0)
            {
                y[ind * incy] = (alpha * sdata[thread_id]) + (beta * y[ind * incy]);
            }
            else
            {
                y[ind * incy] = alpha * sdata[thread_id];
            }
        }
    }
}

/**
  *  Loads pointers (in case of future batched versions) and launches
  *  the actual calculation kernel.
  *
  *  Summary of banded matrices:
  *  Banded matrices consist of the centre diagonal, along with 'kl' sub-diagonals and 'ku' super-diagonals.
  *
  *  These matrices are then compacted into a banded storage format. The main diagonal resides on the (ku+1)'th row,
  *  the the first super-diagonal on the RHS of the ku'th row, the first sub-diagonal on the LHS of the (ku+2)'th row, etc.
  *
  *  Ex: (m = 5, n = 5; ku = 1, kl = 2)
  *
  *  1 2 0 0 0              0 2 2 2 2
  *  3 1 2 0 0              1 1 1 1 1    <- main diag on (ku+1)'th row = (1+1)'th row = 2nd row
  *  4 3 1 2 0     ---->    3 3 3 3 0
  *  0 4 3 1 9              4 4 4 0 0
  *  0 0 4 3 1              0 0 0 0 0
  *
  *  Note: This definition uses 1-indexing as seen above.
  *
  *  The empty parts of these sparse matrices are not to be touched. As can be seen, the column
  *  of each element is preserved in the compaction, and the diagonals are "pushed" upwards and
  *  reside on the same row as the other elements of the same diagonal.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V, typename W>
__global__ void gbmvx_kernel(rocblas_operation transA,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       kl,
                             rocblas_int       ku,
                             U                 alphaa,
                             V                 Aa,
                             ptrdiff_t         shifta,
                             rocblas_int       lda,
                             rocblas_stride    strideA,
                             V                 xa,
                             ptrdiff_t         shiftx,
                             rocblas_int       incx,
                             rocblas_stride    stridex,
                             U                 betaa,
                             W                 ya,
                             ptrdiff_t         shifty,
                             rocblas_int       incy,
                             rocblas_stride    stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const auto* A     = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const auto* x     = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    auto*       y     = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);
    auto        alpha = load_scalar(alphaa, hipBlockIdx_y, 0);
    auto        beta  = load_scalar(betaa, hipBlockIdx_y, 0);

    gbmvx_kernel_calc<DIM_X, DIM_Y>(transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

/**
  *  Here, U is either a `const T* const*` or a `const T*`
  *  V is either a `T*` or a `T* const*`
  */
template <typename T, typename U, typename V>
rocblas_status rocblas_gbmv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       kl,
                                     rocblas_int       ku,
                                     const T*          alpha,
                                     U                 A,
                                     rocblas_int       offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     U                 x,
                                     rocblas_int       offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     const T*          beta,
                                     V                 y,
                                     rocblas_int       offsety,
                                     rocblas_int       incy,
                                     rocblas_stride    stridey,
                                     rocblas_int       batch_count)
{
    // quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx
        = incx < 0 ? offsetx - ptrdiff_t(incx) * (transA == rocblas_operation_none ? n - 1 : m - 1)
                   : offsetx;
    auto shifty
        = incy < 0 ? offsety - ptrdiff_t(incy) * (transA == rocblas_operation_none ? m - 1 : n - 1)
                   : offsety;

    // (gemv) GBMVX_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    rocblas_int          block_dim   = transA == rocblas_operation_none ? m : n;
    static constexpr int GBMVX_DIM_X = 64;
    static constexpr int GBMVX_DIM_Y = 16;
    rocblas_int          blocks      = (block_dim - 1) / (GBMVX_DIM_X) + 1;
    dim3                 gbmvx_grid(blocks, batch_count);
    dim3                 gbmvx_threads(GBMVX_DIM_X, GBMVX_DIM_Y);
    const bool           trans = transA == rocblas_operation_none;
    const bool           conj  = transA == rocblas_operation_conjugate_transpose;

    // Launch a modified gemv kernel. The logic is similar to gemv just with modified
    // indices for the banded matrices.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((gbmvx_kernel<GBMVX_DIM_X, GBMVX_DIM_Y>),
                           gbmvx_grid,
                           gbmvx_threads,
                           0,
                           handle->rocblas_stream,
                           transA,
                           m,
                           n,
                           kl,
                           ku,
                           alpha,
                           A,
                           offseta,
                           lda,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }
    else
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL((gbmvx_kernel<GBMVX_DIM_X, GBMVX_DIM_Y>),
                           gbmvx_grid,
                           gbmvx_threads,
                           0,
                           handle->rocblas_stream,
                           transA,
                           m,
                           n,
                           kl,
                           ku,
                           *alpha,
                           A,
                           offseta,
                           lda,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           *beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }

    return rocblas_status_success;
}

#endif
