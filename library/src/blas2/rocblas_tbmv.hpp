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

// for upper matrices
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void tbmvn_kernel_calc(bool        upper,
                                  bool        diag,
                                  rocblas_int m,
                                  rocblas_int k,
                                  const T*    A,
                                  rocblas_int lda,
                                  const T*    x_copy,
                                  T*          x,
                                  rocblas_int incx)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;
    T res_x = 0.0;

    rocblas_int col = ty; // ty defines the column of banded & regular matrix

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

template <rocblas_int DIM_X, rocblas_int DIM_Y, bool CONJ, typename T>
__device__ void tbmvt_kernel_calc(bool        upper,
                                  bool        diag,
                                  rocblas_int m,
                                  rocblas_int k,
                                  const T*    A,
                                  rocblas_int lda,
                                  const T*    x_copy,
                                  T*          x,
                                  rocblas_int incx)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;
    T res_x = 0.0;

    rocblas_int row = ty; // ty defines the column of banded & regular matrix

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

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__global__ void tbmvn_kernel(bool           upper,
                             bool           diag,
                             rocblas_int    m,
                             rocblas_int    k,
                             const T*       Aa,
                             ptrdiff_t      shifta,
                             rocblas_int    lda,
                             rocblas_stride strideA,
                             const T*       xa_copy,
                             T*             xa,
                             ptrdiff_t      shiftx,
                             rocblas_int    incx,
                             rocblas_stride stridex)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const T* A      = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const T* x_copy = load_ptr_batch(xa_copy, hipBlockIdx_y, 0, m);
    T*       x      = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    tbmvn_kernel_calc<DIM_X, DIM_Y, T>(upper, diag, m, k, A, lda, x_copy, x, incx);
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, bool CONJ, typename T>
__global__ void tbmvt_kernel(bool           upper,
                             bool           diag,
                             rocblas_int    m,
                             rocblas_int    k,
                             const T*       Aa,
                             ptrdiff_t      shifta,
                             rocblas_int    lda,
                             rocblas_stride strideA,
                             const T*       xa_copy,
                             T*             xa,
                             ptrdiff_t      shiftx,
                             rocblas_int    incx,
                             rocblas_stride stridex)
{
    const T* A      = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const T* x_copy = load_ptr_batch(xa_copy, hipBlockIdx_y, 0, m);
    T*       x      = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    tbmvt_kernel_calc<DIM_X, DIM_Y, CONJ, T>(upper, diag, m, k, A, lda, x_copy, x, incx);
}

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

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    offsetx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;

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

    if(transA == rocblas_operation_none)
    {
        // TBMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int TBMVN_DIM_X = 64;
        static constexpr int TBMVN_DIM_Y = 16;
        rocblas_int          blocks      = (m - 1) / (TBMVN_DIM_X) + 1;
        dim3                 tbmvn_grid(blocks, batch_count);
        dim3                 tbmvn_threads(TBMVN_DIM_X, TBMVN_DIM_Y);

        hipLaunchKernelGGL((tbmvn_kernel<TBMVN_DIM_X, TBMVN_DIM_Y, T>),
                           tbmvn_grid,
                           tbmvn_threads,
                           0,
                           rocblas_stream,
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
    }
    else if(transA == rocblas_operation_transpose)
    {
        // transpose
        // TBMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int TBMVT_DIM_X = 64;
        static constexpr int TBMVT_DIM_Y = 16;
        rocblas_int          blocks      = (m - 1) / (TBMVT_DIM_X) + 1;
        dim3                 tbmvt_grid(blocks, batch_count);
        dim3                 tbmvt_threads(TBMVT_DIM_X, TBMVT_DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((tbmvt_kernel<TBMVT_DIM_X, TBMVT_DIM_Y, false, T>),
                               tbmvt_grid,
                               tbmvt_threads,
                               0,
                               rocblas_stream,
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
        }
    }
    else // conjugate transpose
    {
        // TBMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int TBMVT_DIM_X = 64;
        static constexpr int TBMVT_DIM_Y = 16;
        rocblas_int          blocks      = (m - 1) / (TBMVT_DIM_X) + 1;
        dim3                 tbmvt_grid(blocks, batch_count);
        dim3                 tbmvt_threads(TBMVT_DIM_X, TBMVT_DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((tbmvt_kernel<TBMVT_DIM_X, TBMVT_DIM_Y, true, T>),
                               tbmvt_grid,
                               tbmvt_threads,
                               0,
                               rocblas_stream,
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
        }
    }
    return rocblas_status_success;
}

#endif
