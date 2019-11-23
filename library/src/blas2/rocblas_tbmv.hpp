/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TBMV_HPP__
#define __ROCBLAS_TBMV_HPP__
#include "handle.h"
#include "rocblas.h"
#include "utility.h"

// for upper matrices
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void tbmvn_kernel_calc(
    rocblas_int m, rocblas_int k, const T* A, rocblas_int lda, T* x, rocblas_int incx)
{
    // TODO: 99.9% sure there's a race condition in here somewhere which makes it a little
    //       sketchy, seems to work for now though?
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0;
    T res_x = 0;

    rocblas_int m_tail = (m) % DIM_Y; // TODO: can't this be k+1 mod DIM_Y?
    rocblas_int col
        = ty; // ty defines the column of banded matrix/regular matrix since they're the same

    for(col = ty; col < m; col += DIM_Y) //(m - m_tail); col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = ind + (k - col);
        // if(row < 0) break;
        if(ind < m)
        {
            // T tmpA = 0;
            // if(row <= k && row >= 0)
            //     tmpA = A[row + col * lda];

            T tmpA = A[row + col * lda];
            if(row > k || row < 0)
                tmpA = 0;

            res_A += (tmpA * x[col * incx]);
        }
    }

    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    if(thread_id < DIM_X)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        if(ind < m)
        {
            x[ind] = (sdata[thread_id]);
        }
    }
}

template <rocblas_int NB_X, typename T>
__device__ void tbmvt_kernel_calc(
    rocblas_int m, rocblas_int k, const T* A, rocblas_int lda, const T* x, rocblas_int incx)
{
    rocblas_int tx = hipThreadIdx_x;

    if(tx < m)
        A += tx;

    rocblas_int col = hipBlockIdx_x;
    A += col * lda;

    T res;
    res = 0.0;

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    for(rocblas_int i = 0; i < m_full; i += NB_X)
        res += (A[i]) * x[(tx + i) * incx];

    if(tx + m_full < m)
        res += (A[m_full]) * x[(tx + m_full) * incx];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if(NB_X > 16)
    {
        // rocblas_sum_reduce<NB_X>(tx, sdata);
    }
    else
    {
        __syncthreads();

        if(tx == 0)
        {
            for(rocblas_int i = 1; i < m && i < NB_X; i++)
                sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if(tx == 0)
    {
        x[col * incx] = sdata[0];
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

    tbmvn_kernel_calc<DIM_X, DIM_Y, T>(m, k, Aa, lda, xa, incx);
}

template <rocblas_int NB_X, typename T>
__global__ void tbmvt_kernel(rocblas_int    m,
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
    const T* A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    T*       x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    tbmvt_kernel_calc<NB_X, T>(m, k, A, lda, x, incx);
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
        static constexpr int TBMVN_DIM_X = 64; //8;//64;
        static constexpr int TBMVN_DIM_Y = 16; //8;//16;
        rocblas_int          blocks      = (m - 1) / (TBMVN_DIM_X) + 1;
        dim3                 tbmvn_grid(blocks); //, batch_count);
        dim3                 tbmvn_threads(TBMVN_DIM_X, TBMVN_DIM_Y);
        std::cout << "blocks: " << blocks << ", dim_X: " << TBMVN_DIM_X
                  << ", dim_y: " << TBMVN_DIM_Y << "\n";
        std::cout << "m: " << m << ", k: " << k << "\n";

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
                               0, //offseta,
                               lda,
                               0, //strideA,
                               x,
                               0, //offsetx,
                               incx,
                               0); //stridex);
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
