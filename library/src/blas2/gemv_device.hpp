/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __GEMV_DEVICE_HPP__
#define __GEMV_DEVICE_HPP__

#include "../blas1/reduction.h"
#include "utility.h"

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          typename T,
          typename U,
          typename std::enable_if<!std::is_same<T, rocblas_double_complex>{}, int>::type = 0>
__device__ void gemvn_kernel_calc(rocblas_int m,
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
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind;

    __shared__ T sdata[DIM_X * 4 * DIM_Y];

    T res_A[4]; // micor tile is 4 * 4
    T res_x[4];

    res_A[0] = res_x[0] = 0.0;
    res_A[1] = res_x[0] = 0.0;
    res_A[2] = res_x[0] = 0.0;
    res_A[3] = res_x[0] = 0.0;

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
        }

        if(ind + DIM_X < m)
        {
            res_A[1] += A[ind + DIM_X + (col + 0) * lda] * res_x[0];
            res_A[1] += A[ind + DIM_X + (col + 1) * lda] * res_x[1];
            res_A[1] += A[ind + DIM_X + (col + 2) * lda] * res_x[2];
            res_A[1] += A[ind + DIM_X + (col + 3) * lda] * res_x[3];
        }

        if(ind + 2 * DIM_X < m)
        {
            res_A[2] += A[ind + 2 * DIM_X + (col + 0) * lda] * res_x[0];
            res_A[2] += A[ind + 2 * DIM_X + (col + 1) * lda] * res_x[1];
            res_A[2] += A[ind + 2 * DIM_X + (col + 2) * lda] * res_x[2];
            res_A[2] += A[ind + 2 * DIM_X + (col + 3) * lda] * res_x[3];
        }

        if(ind + 3 * DIM_X < m)
        {
            res_A[3] += A[ind + 3 * DIM_X + (col + 0) * lda] * res_x[0];
            res_A[3] += A[ind + 3 * DIM_X + (col + 1) * lda] * res_x[1];
            res_A[3] += A[ind + 3 * DIM_X + (col + 2) * lda] * res_x[2];
            res_A[3] += A[ind + 3 * DIM_X + (col + 3) * lda] * res_x[3];
        }
    }

    // if n  is not multiple of (DIM_Y * 4)
    if(n_tail > 0)
    {
        if(col + 0 < n)
            res_x[0] = x[(col + 0) * incx];
        else
            res_x[0] = 0.0;

        if(col + 1 < n)
            res_x[1] = x[(col + 1) * incx];
        else
            res_x[1] = 0.0;

        if(col + 2 < n)
            res_x[2] = x[(col + 2) * incx];
        else
            res_x[2] = 0.0;

        if(col + 3 < n)
            res_x[3] = x[(col + 3) * incx];
        else
            res_x[3] = 0.0;

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * lda * (col + 0 < n)] * res_x[0];
            res_A[0] += A[ind + (col + 1) * lda * (col + 1 < n)] * res_x[1];
            res_A[0] += A[ind + (col + 2) * lda * (col + 2 < n)] * res_x[2];
            res_A[0] += A[ind + (col + 3) * lda * (col + 3 < n)] * res_x[3];
        }

        if(ind + DIM_X < m)
        {
            res_A[1] += A[ind + DIM_X + (col + 0) * lda * (col + 0 < n)] * res_x[0];
            res_A[1] += A[ind + DIM_X + (col + 1) * lda * (col + 1 < n)] * res_x[1];
            res_A[1] += A[ind + DIM_X + (col + 2) * lda * (col + 2 < n)] * res_x[2];
            res_A[1] += A[ind + DIM_X + (col + 3) * lda * (col + 3 < n)] * res_x[3];
        }

        if(ind + 2 * DIM_X < m)
        {
            res_A[2] += A[ind + 2 * DIM_X + (col + 0) * lda * (col + 0 < n)] * res_x[0];
            res_A[2] += A[ind + 2 * DIM_X + (col + 1) * lda * (col + 1 < n)] * res_x[1];
            res_A[2] += A[ind + 2 * DIM_X + (col + 2) * lda * (col + 2 < n)] * res_x[2];
            res_A[2] += A[ind + 2 * DIM_X + (col + 3) * lda * (col + 3 < n)] * res_x[3];
        }

        if(ind + 3 * DIM_X < m)
        {
            res_A[3] += A[ind + 3 * DIM_X + (col + 0) * lda * (col + 0 < n)] * res_x[0];
            res_A[3] += A[ind + 3 * DIM_X + (col + 1) * lda * (col + 1 < n)] * res_x[1];
            res_A[3] += A[ind + 3 * DIM_X + (col + 2) * lda * (col + 2 < n)] * res_x[2];
            res_A[3] += A[ind + 3 * DIM_X + (col + 3) * lda * (col + 3 < n)] * res_x[3];
        }
    }

    sdata[tx + ty * DIM_X * 4]             = res_A[0];
    sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
    sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
    sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
    if(thread_id < DIM_X * 4)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];

        if(ind < m)
        {
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

// Overload for double precision complex numbers. We run out of registers
// if we use the above algorithm.
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U>
__device__ void gemvn_kernel_calc(rocblas_int                   m,
                                  rocblas_int                   n,
                                  U                             alpha,
                                  const rocblas_double_complex* A,
                                  rocblas_int                   lda,
                                  const rocblas_double_complex* x,
                                  rocblas_int                   incx,
                                  U                             beta,
                                  rocblas_double_complex*       y,
                                  rocblas_int                   incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

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

    ind = hipBlockIdx_x * DIM_X + thread_id;
    if(thread_id < DIM_X)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        if(ind < m)
        {
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

template <rocblas_int NB_X, typename T, typename U>
__device__ void gemvc_kernel_calc(rocblas_int m,
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
        res += conj(A[i]) * x[(tx + i) * incx];

    if(tx + m_full < m)
        res += conj(A[m_full]) * x[(tx + m_full) * incx];

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
        if(beta != 0)
        {
            y[col * incy] = alpha * sdata[0] + beta * y[col * incy];
        }
        else
        {
            y[col * incy] = alpha * sdata[0];
        }
    }
}

template <rocblas_int NB_X, typename T, typename U>
__device__ void gemvt_kernel_calc(rocblas_int m,
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
        if(beta != 0)
        {
            y[col * incy] = alpha * sdata[0] + beta * y[col * incy];
        }
        else
        {
            y[col * incy] = alpha * sdata[0];
        }
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U, typename V, typename W>
__global__ void gemvn_kernel(rocblas_int    m,
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
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    const T* A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    T*       y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    gemvn_kernel_calc<DIM_X, DIM_Y>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <rocblas_int NB_X, typename T, typename U, typename V, typename W>
__global__ void gemvc_kernel(rocblas_int    m,
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
    const T* A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    T*       y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    gemvc_kernel_calc<NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <rocblas_int NB_X, typename T, typename U, typename V, typename W>
__global__ void gemvt_kernel(rocblas_int    m,
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
    const T* A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    T*       y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    gemvt_kernel_calc<NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

#endif
