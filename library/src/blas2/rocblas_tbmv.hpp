/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

// copy_helper
template <typename T, typename U, typename V>
__global__ void tbmv_copy_helper(rocblas_int    n,
                                 const U        xa,
                                 ptrdiff_t      shiftx,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 V              ya,
                                 ptrdiff_t      shifty,
                                 rocblas_int    incy,
                                 rocblas_stride stridey)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(tid < n)
    {
        const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
        T*       y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

        y[tid * incy] = x[tid * incx];
    }
}

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
                                 const T*    x_copy,
                                 rocblas_int incx)
{
    T           res_A = 0.0;
    rocblas_int row   = ty; // ty defines the column of banded & regular matrix

    for(row = ty; row < m; row += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int col     = ind; // upper ? ind + (k - col) : ind - col;
        rocblas_int min_row = upper ? k - col : 0;
        rocblas_int adder   = upper ? 0 : col;
        rocblas_int max_row = upper ? k : m - 1 - col;

        if(col < m)
        {
            if(upper)
            {
                if(row < k && row >= k - col && row != k)
                {
                    res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                              * x_copy[row - (min_row) + adder]);
                }
                else if(row == k)
                {
                    if(diag)
                        res_A += x_copy[row - (min_row) + adder];
                    else
                        res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                                  * x_copy[row - (min_row) + adder]);
                }
            }
            else
            {
                if(row <= k && row <= m - 1 - col && row > 0)
                {
                    res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                              * x_copy[row - (min_row) + adder]);
                }
                else if(row == 0)
                {
                    if(diag)
                        res_A += x_copy[row - (min_row) + adder];
                    else
                        res_A += ((CONJ ? conj(A[row + col * lda]) : A[row + col * lda])
                                  * x_copy[row - (min_row) + adder]);
                }
            }
        }
    }
    return res_A;
}

template <rocblas_int DIM_Y, typename T>
__device__ T tbmvn_kernel_helper(rocblas_int ty,
                                 rocblas_int ind,
                                 bool        upper,
                                 bool        diag,
                                 rocblas_int m,
                                 rocblas_int k,
                                 const T*    A,
                                 rocblas_int lda,
                                 const T*    x_copy,
                                 rocblas_int incx)
{
    T           res_A = 0.0;
    rocblas_int col   = ty; // ty defines the column of banded & regular matrix

    for(col = ty; col < m; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = upper ? ind + (k - col) : ind - col;

        if(ind < m)
        {
            if(row < k && row > 0)
            {
                res_A += (A[row + col * lda] * x_copy[col]);
            }
            else if(row == 0)
            {
                if(diag && !upper)
                    res_A += x_copy[col];
                else if(k == 0 && diag && upper)
                    res_A += x_copy[col];
                else
                    res_A += (A[row + col * lda] * x_copy[col]);
            }
            else if(row == k)
            {
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
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;
    if(transA == rocblas_operation_none)
    {
        res_A = tbmvn_kernel_helper<DIM_Y, T>(ty, ind, upper, diag, m, k, A, lda, x_copy, incx);
    }
    else
    {
        bool CONJ = transA == rocblas_operation_conjugate_transpose;
        res_A
            = tbmvt_kernel_helper<DIM_Y, T>(CONJ, ty, ind, upper, diag, m, k, A, lda, x_copy, incx);
    }
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    ind       = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        if(ind < m)
        {
            x[ind * incx] = (sdata[thread_id]);
        }
    }
}

/**
  *  Loads pointers (in case of future batched versions) and launches
  *  the actual calculation kernel.
  *
  *
  *  Two types of banded matrices exist, upper and lower. These matrices consist of
  *  the centre diagonal, along with 'k' sub-diagonals (if lower) or super-diagonals (if upper).
  *
  *  These matrices are then compacted into a banded storage format. For upper-triangular,
  *  the k'th super-diagonal resides on the right-hand side of the first row, k-1th on the second,
  *  with the main diagonal on the k'th row.
  *
  *  Ex: (upper; m = 5; k = 2)
  *
  *  1 6 9 0 0              0 0 9 8 7
  *  0 2 7 8 0              0 6 7 8 9
  *  0 0 3 8 7     ---->    1 2 3 4 5
  *  0 0 0 4 2              0 0 0 0 0
  *  0 0 0 0 5              0 0 0 0 0
  *
  *  For lower-triangular, the main diagonal resides on the 0'th row, working up to the k'th
  *  sub-diagonal residing on the right-hand side of the k'th row.
  *
  *  Ex: (lower; m = 5; k = 2)
  *
  *  1 0 0 0 0              1 2 3 4 5
  *  6 2 0 0 0              0 6 7 8 9
  *  9 7 3 0 0     ---->    0 0 9 8 7
  *  0 8 8 4 0              0 0 0 0 0
  *  0 0 7 9 5              0 0 0 0 0
  *
  *  The empty parts of these sparse matrices are not to be touched.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__global__ void tbmvx_kernel(rocblas_operation transA,
                             bool              upper,
                             bool              diag,
                             rocblas_int       m,
                             rocblas_int       k,
                             const T*          Aa,
                             ptrdiff_t         shifta,
                             rocblas_int       lda,
                             rocblas_stride    strideA,
                             const T*          xa_copy,
                             T*                xa,
                             ptrdiff_t         shiftx,
                             rocblas_int       incx,
                             rocblas_stride    stridex)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const T* A      = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const T* x_copy = load_ptr_batch(xa_copy, hipBlockIdx_y, 0, m);
    T*       x      = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    tbmvx_kernel_calc<DIM_X, DIM_Y, T>(transA, upper, diag, m, k, A, lda, x_copy, x, incx);
}

/**
  *  First, makes a copy of 'x', then uses a modified gemv algorithm
  *  to perform x := transA(A) * x_copy
  */
template <typename T>
rocblas_status rocblas_tbmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       k,
                                     const T*          A,
                                     rocblas_int       offseta,
                                     rocblas_int       lda,
                                     rocblas_stride    strideA,
                                     T*                x,
                                     rocblas_int       offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     rocblas_int       batch_count,
                                     const T*          x_copy)
{
    //quick return
    if(!m || !batch_count)
        return rocblas_status_success;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    offsetx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;

    // First we make a copy of x so we can avoid RAW race conditions in the kernel
    int  copy_blocks = (m - 1) / 256 + 1;
    dim3 copy_grid(copy_blocks, batch_count);
    dim3 copy_threads(256);

    hipLaunchKernelGGL((tbmv_copy_helper<T>),
                       copy_grid,
                       copy_threads,
                       0,
                       handle->rocblas_stream,
                       m,
                       x,
                       offsetx,
                       incx,
                       stridex,
                       (T*)x_copy,
                       0,
                       1,
                       m);

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
    hipLaunchKernelGGL((tbmvx_kernel<TBMVX_DIM_X, TBMVX_DIM_Y, T>),
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
                       x_copy,
                       x,
                       offsetx,
                       incx,
                       stridex);

    return rocblas_status_success;
}

#endif
