/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

// uses dot shuffle reductions
#include "../blas1/rocblas_dot.hpp"

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          typename T_lda,
          typename T,
          typename U,
          std::enable_if_t<!std::is_same<T, rocblas_double_complex>{}, int> = 0>
ROCBLAS_KERNEL_ILF void gemvn_kernel_calc(rocblas_int m,
                                          rocblas_int n,
                                          U           alpha,
                                          const T*    A,
                                          T_lda       lda,
                                          const T*    x,
                                          rocblas_int incx,
                                          U           beta,
                                          T*          y,
                                          rocblas_int incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    if(!alpha)
    {
        if(thread_id < DIM_X * 4)
        {
            rocblas_int ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
            if(ind < m)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = hipThreadIdx_x;
    rocblas_int ty = hipThreadIdx_y;

    rocblas_int ind;

    __shared__ T sdata[DIM_X * 4 * DIM_Y];

    T res_A[4];
    T res_x[4];

    res_A[0] = res_A[1] = res_A[2] = res_A[3] = T{0};

    ind = hipBlockIdx_x * DIM_X * 4 + tx;

    rocblas_int n_tail = n % (4 * DIM_Y);
    rocblas_int col    = ty * 4;

    for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
    {
        res_x[0] = x[(col + 0) * incx];
        res_x[1] = x[(col + 1) * incx];
        res_x[2] = x[(col + 2) * incx];
        res_x[3] = x[(col + 3) * incx];

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * lda] * res_x[0];
            res_A[0] += A[ind + (col + 1) * lda] * res_x[1];
            res_A[0] += A[ind + (col + 2) * lda] * res_x[2];
            res_A[0] += A[ind + (col + 3) * lda] * res_x[3];

            if(ind + DIM_X < m)
            {
                res_A[1] += A[ind + DIM_X + (col + 0) * lda] * res_x[0];
                res_A[1] += A[ind + DIM_X + (col + 1) * lda] * res_x[1];
                res_A[1] += A[ind + DIM_X + (col + 2) * lda] * res_x[2];
                res_A[1] += A[ind + DIM_X + (col + 3) * lda] * res_x[3];

                if(ind + 2 * DIM_X < m)
                {
                    res_A[2] += A[ind + 2 * DIM_X + (col + 0) * lda] * res_x[0];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 1) * lda] * res_x[1];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 2) * lda] * res_x[2];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 3) * lda] * res_x[3];

                    if(ind + 3 * DIM_X < m)
                    {
                        res_A[3] += A[ind + 3 * DIM_X + (col + 0) * lda] * res_x[0];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 1) * lda] * res_x[1];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 2) * lda] * res_x[2];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 3) * lda] * res_x[3];
                    }
                }
            }
        }
    }

    // if n is not multiple of (DIM_Y * 4)
    if(n_tail > 0)
    {
        res_x[0] = res_x[1] = res_x[2] = res_x[3] = T{0};

        if(col + 0 < n)
        {
            res_x[0] = x[(col + 0) * incx];

            if(col + 1 < n)
            {
                res_x[1] = x[(col + 1) * incx];

                if(col + 2 < n)
                {
                    res_x[2] = x[(col + 2) * incx];

                    if(col + 3 < n)
                        res_x[3] = x[(col + 3) * incx];
                }
            }
        }

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * lda * (col + 0 < n)] * res_x[0];
            res_A[0] += A[ind + (col + 1) * lda * (col + 1 < n)] * res_x[1];
            res_A[0] += A[ind + (col + 2) * lda * (col + 2 < n)] * res_x[2];
            res_A[0] += A[ind + (col + 3) * lda * (col + 3 < n)] * res_x[3];

            if(ind + DIM_X < m)
            {
                res_A[1] += A[ind + DIM_X + (col + 0) * lda * (col + 0 < n)] * res_x[0];
                res_A[1] += A[ind + DIM_X + (col + 1) * lda * (col + 1 < n)] * res_x[1];
                res_A[1] += A[ind + DIM_X + (col + 2) * lda * (col + 2 < n)] * res_x[2];
                res_A[1] += A[ind + DIM_X + (col + 3) * lda * (col + 3 < n)] * res_x[3];

                if(ind + 2 * DIM_X < m)
                {
                    res_A[2] += A[ind + 2 * DIM_X + (col + 0) * lda * (col + 0 < n)] * res_x[0];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 1) * lda * (col + 1 < n)] * res_x[1];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 2) * lda * (col + 2 < n)] * res_x[2];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 3) * lda * (col + 3 < n)] * res_x[3];

                    if(ind + 3 * DIM_X < m)
                    {
                        res_A[3] += A[ind + 3 * DIM_X + (col + 0) * lda * (col + 0 < n)] * res_x[0];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 1) * lda * (col + 1 < n)] * res_x[1];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 2) * lda * (col + 2 < n)] * res_x[2];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 3) * lda * (col + 3 < n)] * res_x[3];
                    }
                }
            }
        }
    }

    sdata[tx + ty * DIM_X * 4]             = res_A[0];
    sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
    sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
    sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

    __syncthreads();

    if(thread_id < DIM_X * 4)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];

        ind = hipBlockIdx_x * DIM_X * 4 + thread_id;

        if(ind < m)
            y[ind * incy]
                = beta ? alpha * sdata[thread_id] + beta * y[ind * incy] : alpha * sdata[thread_id];
    }
}

// Overload for double precision complex numbers. We run out of registers
// if we use the above algorithm.
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T_lda, typename U>
ROCBLAS_KERNEL_ILF void gemvn_kernel_calc(rocblas_int                   m,
                                          rocblas_int                   n,
                                          U                             alpha,
                                          const rocblas_double_complex* A,
                                          T_lda                         lda,
                                          const rocblas_double_complex* x,
                                          rocblas_int                   incx,
                                          U                             beta,
                                          rocblas_double_complex*       y,
                                          rocblas_int                   incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    if(!alpha)
    {
        if(thread_id < DIM_X)
        {
            rocblas_int ind = hipBlockIdx_x * DIM_X + thread_id;
            if(ind < m)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = hipBlockIdx_x * DIM_X + tx;

    __shared__ rocblas_double_complex sdata[DIM_X * DIM_Y];

    rocblas_double_complex res_A;
    rocblas_double_complex res_x;

    res_A = res_x = rocblas_double_complex{0, 0};

    rocblas_int n_tail = n % (DIM_Y);
    rocblas_int col    = ty;

    for(col = ty; col < (n - n_tail); col += DIM_Y)
    {
        res_x = x[(col)*incx];

        if(ind < m)
        {
            res_A += A[ind + col * lda] * x[col * incx];
        }
    }

    // if n  is not multiple of (DIM_Y)
    if(n_tail > 0)
    {
        if(col + 0 < n)
            res_x = x[(col)*incx];
        else
            res_x = rocblas_double_complex{0, 0};

        if(ind < m)
        {
            res_A += A[ind + (col)*lda * (col < n)] * res_x;
        }
    }

    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    if(thread_id < DIM_X)
    {
        // always alpha non zero as !alpha quick return
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        ind = hipBlockIdx_x * DIM_X + thread_id;

        if(ind < m)
        {
            y[ind * incy]
                = beta ? alpha * sdata[thread_id] + beta * y[ind * incy] : alpha * sdata[thread_id];
        }
    }
}

template <bool CONJ, rocblas_int NB_X, typename T_lda, typename T, typename U>
ROCBLAS_KERNEL_ILF void gemvt_kernel_calc(rocblas_int m,
                                          rocblas_int n,
                                          U           alpha,
                                          const T*    A,
                                          T_lda       lda,
                                          const T*    x,
                                          rocblas_int incx,
                                          U           beta,
                                          T*          y,
                                          rocblas_int incy)
{
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int col = hipBlockIdx_x;

    if(!alpha)
    {
        if(tx == 0)
            y[col * incy] = beta ? beta * y[col * incy] : 0;
        return;
    }

    if(tx < m)
        A += tx;

    A += col * size_t(lda);

    T res = 0;

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    for(rocblas_int i = 0; i < m_full; i += NB_X)
        res += (CONJ ? conj(A[i]) : A[i]) * x[(tx + i) * incx];

    if(tx + m_full < m)
        res += (CONJ ? conj(A[m_full]) : A[m_full]) * x[(tx + m_full) * incx];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if(NB_X > 16)
    {
        rocblas_sum_reduce<NB_X>(tx, sdata);
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
        // !alpha handled earlier by early return
        y[col * incy] = beta ? alpha * sdata[0] + beta * y[col * incy] : alpha * sdata[0];
    }
}

//Optimized kernel for GEMV transpose case when m or n is less than 6000
template <bool CONJ, rocblas_int NB_X, typename T_lda, typename T, typename U>
ROCBLAS_KERNEL_ILF void gemvt_warp_reduce_kernel_calc(rocblas_int m,
                                                      rocblas_int n,
                                                      U           alpha,
                                                      const T* __restrict__ A,
                                                      T_lda lda,
                                                      const T* __restrict__ x,
                                                      rocblas_int incx,
                                                      U           beta,
                                                      T* __restrict__ y,
                                                      rocblas_int incy)
{
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int col = hipBlockIdx_x;

    if(!alpha)
    {
        if(tx == 0)
            y[col * incy] = beta ? beta * y[col * incy] : 0;
        return;
    }

    if(tx < m)
        A += tx;

    //Each BlockIdx.x takes care of each column of matrix A
    A += col * size_t(lda);

    T res = 0;

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    //Each column of Matrix A is multiplied with vector x and the resultant value is stored in res.
    //If m > NB_X, then the threads are reused and the multiplied values will be accumalated.
    for(rocblas_int i = 0; tx + i < m_full; i += NB_X)
        res += (CONJ ? conj(A[i]) : A[i]) * x[(tx + i) * incx];

    if(tx + m_full < m)
        res += (CONJ ? conj(A[m_full]) : A[m_full]) * x[(tx + m_full) * incx];

    if(NB_X <= warpSize)
    {
        //shuffle warp reduction if NB_X is less than or equal to 64 (WarpSize)
        res = wavefront_reduce<NB_X>(res);
    }
    else
    {
        //block shuffle warp reduction if NB_X is greater than 64 (WarpSize)
        res = rocblas_dot_block_reduce<NB_X>(res);
    }

    if(tx == 0)
    {
        // !alpha handled earlier by early return
        y[col * incy] = beta ? alpha * res + beta * y[col * incy] : alpha * res;
    }
}

template <bool CONJ, rocblas_int NB_X, rocblas_int WIN, typename T_lda, typename T, typename U>
ROCBLAS_KERNEL_ILF void gemvt_sn_kernel_calc(rocblas_int m,
                                             rocblas_int n,
                                             U           alpha,
                                             const T*    A,
                                             T_lda       lda,
                                             const T*    x,
                                             rocblas_int incx,
                                             T*          workspace)
{
    // skinny n kernel

    rocblas_int tx = hipThreadIdx_x;

    // offset blocks * cols * batch
    workspace += size_t(hipGridDim_x) * n * hipBlockIdx_y;

    // We need to short-circuit if alpha==0 and not propagate NaNs
    if(!alpha)
    {
        if(tx == 0)
            for(int i = 0; i < n; i++)
                workspace[hipBlockIdx_x + (i)*hipGridDim_x] = 0;
        return;
    }

    int row = tx * WIN + hipBlockIdx_x * NB_X * WIN;
    A += row;

    constexpr int NC = 4;

    rocblas_int n_tail = n % NC;

    rocblas_int m_tail = m % WIN;

    T sum[NC];
    T xvec[WIN];

    int i = 0; // col
    for(i = 0; i < n - n_tail; i += NC)
    {
        sum[0] = sum[1] = sum[2] = sum[3] = T{0};

        if(row + WIN <= m)
        {
            for(int j = 0; j < WIN; j++)
            {
                xvec[j] = x[(row + j) * incx];
            }
            for(int j = 0; j < WIN; j++)
            {
                for(int k = 0; k < NC; k++)
                    sum[k] += (CONJ ? conj(A[(i + k) * lda + j]) : A[(i + k) * lda + j]) * xvec[j];
            }
        }
        else if(row + m_tail <= m)
        {
            for(int j = 0; j < m_tail; j++)
            {
                xvec[j] = x[(row + j) * incx];
            }
            for(int j = 0; j < m_tail; j++)
            {
                for(int k = 0; k < NC; k++)
                    sum[k] += (CONJ ? conj(A[(i + k) * lda + j]) : A[(i + k) * lda + j]) * xvec[j];
            }
        }

        for(int k = 0; k < NC; k++)
            sum[k] = rocblas_dot_block_reduce<NB_X>(sum[k]);

        if(tx == 0)
        {
            for(int k = 0; k < NC; k++)
                workspace[hipBlockIdx_x + (k + i) * hipGridDim_x] = alpha * sum[k];
        }
    }
    for(; i < n; i++)
    {
        sum[0] = T{0};

        if(row + WIN <= m)
        {
            for(int j = 0; j < WIN; j++)
            {
                xvec[j] = x[(row + j) * incx];
            }
            for(int j = 0; j < WIN; j++)
            {
                sum[0] += (CONJ ? conj(A[(i + 0) * lda + j]) : A[(i + 0) * lda + j]) * xvec[j];
            }
        }
        else if(row + m_tail <= m)
        {
            for(int j = 0; j < m_tail; j++)
            {
                xvec[j] = x[(row + j) * incx];
            }
            for(int j = 0; j < m_tail; j++)
            {
                sum[0] += (CONJ ? conj(A[(i + 0) * lda + j]) : A[(i + 0) * lda + j]) * xvec[j];
            }
        }
        sum[0] = rocblas_dot_block_reduce<NB_X>(sum[0]);
        if(tx == 0)
            workspace[hipBlockIdx_x + (i)*hipGridDim_x] = alpha * sum[0];
    }
}

template <rocblas_int NB, rocblas_int WIN, typename T, typename U, typename W>
ROCBLAS_KERNEL __launch_bounds__(NB) void rocblas_gemvt_sn_reduce(rocblas_int    n_sums,
                                                                  U              beta_device_host,
                                                                  rocblas_stride stride_beta,
                                                                  W* __restrict__ ya,
                                                                  ptrdiff_t      shifty,
                                                                  rocblas_int    incy,
                                                                  rocblas_stride stridey,
                                                                  T* __restrict__ workspace)
{
    T*   y    = load_ptr_batch(ya, hipBlockIdx_z, shifty, stridey);
    auto beta = load_scalar(beta_device_host, hipBlockIdx_z, stride_beta);

    T sum{0};

    int offset = size_t(n_sums) * hipGridDim_y * hipBlockIdx_z + hipBlockIdx_y * n_sums;
    workspace += offset;

    int inc = hipBlockDim_x * WIN;

    int i         = hipThreadIdx_x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += workspace[i + j];
    }
    if(hipThreadIdx_x < remainder)
    {
        sum += workspace[n_sums - 1 - hipThreadIdx_x];
    }
    sum = rocblas_dot_block_reduce<NB>(sum);

    if(hipThreadIdx_x == 0)
    {
        y[hipBlockIdx_y * incy] = beta ? (y[hipBlockIdx_y * incy] * beta) + sum : sum;
    }
}

template <bool CONJ, rocblas_int NB_X, typename T, typename U>
ROCBLAS_KERNEL_ILF void gemvtsm_kernel_calc(rocblas_int m,
                                            rocblas_int n,
                                            U           alpha,
                                            const T*    A,
                                            rocblas_int lda,
                                            const T*    x,
                                            rocblas_int incx,
                                            U           beta,
                                            T*          y,
                                            rocblas_int incy)
{
    // small m <= 64 kernel

    rocblas_int tx = hipThreadIdx_x;

    if(!alpha)
    {
        if(beta)
            for(rocblas_int i = 0; i < n; i += NB_X)
            {
                rocblas_int col = i + tx;
                if(col < n)
                    y[col * incy] *= beta;
            }
        else
            for(rocblas_int i = 0; i < n; i += NB_X)
            {
                rocblas_int col = i + tx;
                if(col < n)
                    y[col * incy] = 0;
            }
        return;
    }

    __shared__ T shared_x[64];

    if(tx < m)
        shared_x[tx] = alpha * x[tx * incx];
    __syncthreads();

    for(rocblas_int i = 0; i < n; i += NB_X)
    {
        rocblas_int col = i + tx;
        if(col < n)
        {
            rocblas_int idx  = col * incy;
            T           res  = beta ? beta * y[idx] : 0;
            const T*    Aptr = A + col * size_t(lda);
            for(rocblas_int l = 0; l < m; ++l)
                res += shared_x[l] * (CONJ ? conj(Aptr[l]) : Aptr[l]);
            y[idx] = res;
        }
    }
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          typename T_lda,
          typename T,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void gemvn_kernel(rocblas_int    m,
                                                                 rocblas_int    n,
                                                                 U              alpha_device_host,
                                                                 rocblas_stride stride_alpha,
                                                                 const V*       Aa,
                                                                 ptrdiff_t      shifta,
                                                                 T_lda          lda,
                                                                 rocblas_stride strideA,
                                                                 const V*       xa,
                                                                 ptrdiff_t      shiftx,
                                                                 rocblas_int    incx,
                                                                 rocblas_stride stridex,
                                                                 U              beta_device_host,
                                                                 rocblas_stride stride_beta,
                                                                 W*             ya,
                                                                 ptrdiff_t      shifty,
                                                                 rocblas_int    incy,
                                                                 rocblas_stride stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    gemvn_kernel_calc<DIM_X, DIM_Y, T_lda>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// lda always cast to size_t so single kernel
template <bool        CONJ,
          rocblas_int NB_X,
          typename T_lda,
          typename T,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL __launch_bounds__(NB_X) void gemvt_kernel(rocblas_int    m,
                                                         rocblas_int    n,
                                                         U              alpha_device_host,
                                                         rocblas_stride stride_alpha,
                                                         const V*       Aa,
                                                         ptrdiff_t      shifta,
                                                         T_lda          lda,
                                                         rocblas_stride strideA,
                                                         const V*       xa,
                                                         ptrdiff_t      shiftx,
                                                         rocblas_int    incx,
                                                         rocblas_stride stridex,
                                                         U              beta_device_host,
                                                         rocblas_stride stride_beta,
                                                         W*             ya,
                                                         ptrdiff_t      shifty,
                                                         rocblas_int    incy,
                                                         rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    gemvt_kernel_calc<CONJ, NB_X, T_lda>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

//Optimized kernel for GEMV transpose case when m or n is less than 6000
template <bool        CONJ,
          rocblas_int NB_X,
          typename T_lda,
          typename T,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL __launch_bounds__(NB_X) void gemvt_warp_reduce_kernel(rocblas_int m,
                                                                     rocblas_int n,
                                                                     U           alpha_device_host,
                                                                     rocblas_stride stride_alpha,
                                                                     const V*       Aa,
                                                                     ptrdiff_t      shifta,
                                                                     T_lda          lda,
                                                                     rocblas_stride strideA,
                                                                     const V*       xa,
                                                                     ptrdiff_t      shiftx,
                                                                     rocblas_int    incx,
                                                                     rocblas_stride stridex,
                                                                     U beta_device_host,
                                                                     rocblas_stride stride_beta,
                                                                     W*             ya,
                                                                     ptrdiff_t      shifty,
                                                                     rocblas_int    incy,
                                                                     rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    gemvt_warp_reduce_kernel_calc<CONJ, NB_X, T_lda>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <bool        CONJ,
          rocblas_int NB_X,
          rocblas_int WIN,
          typename T_lda,
          typename T,
          typename U,
          typename V>
ROCBLAS_KERNEL __launch_bounds__(NB_X) void gemvt_sn_kernel(rocblas_int    m,
                                                            rocblas_int    n,
                                                            U              alpha_device_host,
                                                            rocblas_stride stride_alpha,
                                                            const V*       Aa,
                                                            ptrdiff_t      shifta,
                                                            T_lda          lda,
                                                            rocblas_stride strideA,
                                                            const V*       xa,
                                                            ptrdiff_t      shiftx,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            T*             workspace)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);

    const T* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    gemvt_sn_kernel_calc<CONJ, NB_X, WIN, T_lda>(m, n, alpha, A, lda, x, incx, workspace);
}

template <bool CONJ, rocblas_int NB_X, typename T, typename U, typename V, typename W>
ROCBLAS_KERNEL __launch_bounds__(NB_X) void gemvtsm_kernel(rocblas_int    m,
                                                           rocblas_int    n,
                                                           U              alpha_device_host,
                                                           rocblas_stride stride_alpha,
                                                           const V*       Aa,
                                                           ptrdiff_t      shifta,
                                                           rocblas_int    lda,
                                                           rocblas_stride strideA,
                                                           const V*       xa,
                                                           ptrdiff_t      shiftx,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           U              beta_device_host,
                                                           rocblas_stride stride_beta,
                                                           W*             ya,
                                                           ptrdiff_t      shifty,
                                                           rocblas_int    incy,
                                                           rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_x, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_x, stride_beta);

    if(!alpha && beta == 1)
        return;

    // batch in hipBlockIdx_x not y
    const T* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_x, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_x, shiftx, stridex);

    T* y = load_ptr_batch(ya, hipBlockIdx_x, shifty, stridey);

    gemvtsm_kernel_calc<CONJ, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}
