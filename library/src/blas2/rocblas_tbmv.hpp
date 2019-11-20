/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "handle.h"
#include "rocblas.h"

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void tbmvn_kernel_calc(
    rocblas_int m, rocblas_int k, const T* A, rocblas_int lda, T* x, rocblas_int incx)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A;
    T res_x;

    res_A = res_x = T(0);

    rocblas_int m_tail = (k + 1) % (DIM_Y); // height % block height
    rocblas_int col = ty; // not really the col, more like col of the diagonal, row changes as well
        // Note: ty is weird vertical diagonal index of diagonal blocks.
        //       In this case, ind is the horizontal index onf block.

    for(col = ty; col < (k - m_tail); col += DIM_Y)
    {
        if(ind < k + 1) // else it's zero or outside of matrix
        {
            res_A += A[ind + col * lda] * x[col * incx]; // TODO: some fancy indexing.
        }
    }
    // TODO: Start here
    // if m is not multiple of (DIM_Y)
    if(m_tail > 0)
    {
        if(col + 0 < m)
            res_x = x[(col)*incx];
        else
            res_x = T(0);

        if(ind < m)
        {
            res_A += A[ind + (col)*lda * (col < m)] * res_x;
        }
    }

    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        if(ind < m)
        {
            x[ind * incx] = (sdata[thread_id]);
        }
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__global__ void tbmvn_kernel(rocblas_int    m,
                             rocblas_int    k,
                             const T*       Aa,
                             ptrdiff_t      shifta,
                             rocblas_int    lda,
                             rocblas_stride strideA,
                             T*             xa,
                             ptrdiff_t      shiftx,
                             rocblas_int    incx,
                             rocblas_stride stridex)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const T* A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    T*       x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    tbmvn_kernel_calc<DIM_X, DIM_Y, T>(m, k, A, lda, x, incx);
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
                                     rocblas_int       batch_count)
{
    //quick return
    if(!m || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    offsetx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;

    if(transA == rocblas_operation_none)
    {
        // TBMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int TBMVN_DIM_X = 64;
        static constexpr int TBMVN_DIM_Y = 16;
        rocblas_int          blocks      = (m - 1) / (TBMVN_DIM_X) + 1;
        dim3                 tbmvn_grid(blocks, batch_count);
        dim3                 tbmvn_threads(TBMVN_DIM_X, TBMVN_DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((tbmvn_kernel<TBMVN_DIM_X, TBMVN_DIM_Y, T>),
                               tbmvn_grid,
                               tbmvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               k,
                               A,
                               offseta,
                               lda,
                               strideA,
                               x,
                               offsetx,
                               incx,
                               stridex);
        }
    }
    else if(transA == rocblas_operation_transpose)
    {
        // // transpose
        // // number of columns on the y-dim of the grid
        // static constexpr int NB = 256;
        // dim3                 tbmvt_grid(n, batch_count);
        // dim3                 tbmvt_threads(NB);

        // if(handle->pointer_mode == rocblas_pointer_mode_device)
        // {
        //     hipLaunchKernelGGL((tbmvt_kernel<NB, T>),
        //                        tbmvt_grid,
        //                        tbmvt_threads,
        //                        0,
        //                        rocblas_stream,
        //                        m,
        //                        k,
        //                        A,
        //                        offseta,
        //                        lda,
        //                        strideA,
        //                        x,
        //                        offsetx,
        //                        incx,
        //                        stridex);
        // }
    }
    else // conjugate transpose
    {
        // // conjugate transpose
        // // number of columns on the y-dim of the grid
        // static constexpr int NB = 256;
        // dim3                 tbmvc_grid(n, batch_count);
        // dim3                 tbmvc_threads(NB);

        // if(handle->pointer_mode == rocblas_pointer_mode_device)
        // {
        //     hipLaunchKernelGGL((tbmvc_kernel<NB, T>),
        //                        tbmvc_grid,
        //                        tbmvc_threads,
        //                        0,
        //                        rocblas_stream,
        //                        m,
        //                        n,
        //                        A,
        //                        offseta,
        //                        lda,
        //                        strideA,
        //                        x,
        //                        offsetx,
        //                        incx,
        //                        stridex);
        // }
    }
    return rocblas_status_success;
}

#endif
