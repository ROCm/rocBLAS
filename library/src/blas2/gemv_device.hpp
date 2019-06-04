/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __GEMV_DEVICE_HPP__
#define __GEMV_DEVICE_HPP__

#include <hip/hip_runtime.h>
#include "../blas1/reduction.h"

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
__device__ void gemvn_kernel(rocblas_int m,
                             rocblas_int n,
                             U alpha,
                             const T* A,
                             rocblas_int lda,
                             const T* x,
                             rocblas_int incx,
                             U beta,
                             T* y,
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
        res_x[0] = (col + 0 < n) ? x[(col + 0) * incx] : 0;
        res_x[1] = (col + 1 < n) ? x[(col + 1) * incx] : 0;
        res_x[2] = (col + 2 < n) ? x[(col + 2) * incx] : 0;
        res_x[3] = (col + 3 < n) ? x[(col + 3) * incx] : 0;

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
                y[ind * incy] = alpha * sdata[thread_id] + beta * y[ind * incy];
            }
            else
            {
                y[ind * incy] = alpha * sdata[thread_id];
            }
        }
    }
}


template <rocblas_int NB_X, typename T, typename U>
__device__ void gemvc_kernel(rocblas_int m,
                             rocblas_int n,
                             U alpha,
                             const T* A,
                             rocblas_int lda,
                             const T* x,
                             rocblas_int incx,
                             U beta,
                             T* y,
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

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
__global__ void gemvn_kernel_strided(rocblas_int m,
                             rocblas_int n,
                             U alpha_device_host,
                             const T* Aa,
                             rocblas_int lda,
                             rocblas_int strideA,
                             const T* xa,
                             rocblas_int incx,
                             rocblas_int stridex,
                             U beta_device_host,
                             T* ya,
                             rocblas_int incy,
                             rocblas_int stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate
    
    const T* A;
    const T* x;
    T* y;
    A = Aa + hipBlockIdx_y*strideA;
    x = xa + hipBlockIdx_y*stridex;
    y = ya + hipBlockIdx_y*stridey;
    
    if(incx < 0)
        x -= ssize_t(incx) * (n - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (m - 1);

    auto alpha              = load_scalar(alpha_device_host);
    auto beta               = load_scalar(beta_device_host);
    
    gemvn_kernel<DIM_X,DIM_Y>(m,n,alpha,A,lda,x,incx,beta,y,incy);
}

template <rocblas_int NB_X, typename T, typename U>
__global__ void gemvc_kernel_strided(rocblas_int m,
                             rocblas_int n,
                             U alpha_device_host,
                             const T* Aa,
                             rocblas_int lda,
                             rocblas_int strideA,
                             const T* xa,
                             rocblas_int incx,
                             rocblas_int stridex,
                             U beta_device_host,
                             T* ya,
                             rocblas_int incy,
                             rocblas_int stridey)
{
    const T* A;
    const T* x;
    T* y;
    A = Aa + hipBlockIdx_y*strideA;
    x = xa + hipBlockIdx_y*stridex;
    y = ya + hipBlockIdx_y*stridey;

    if(incx < 0)
        x -= ssize_t(incx) * (m - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (n - 1);
    
    auto alpha     = load_scalar(alpha_device_host);
    auto beta      = load_scalar(beta_device_host);
    
    gemvc_kernel<NB_X>(m,n,alpha,A,lda,x,incx,beta,y,incy);
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
__global__ void gemvn_kernel_batched(rocblas_int m,
                            rocblas_int n,
                            U alpha_device_host,
                            const T *const Aa[],
                            rocblas_int lda,
                            const T *const xa[],
                            rocblas_int incx,
                            U beta_device_host,
                            T *const ya[],
                            rocblas_int incy)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate
    
    const T* A;
    const T* x;
    T* y;
    A = Aa[hipBlockIdx_y];
    x = xa[hipBlockIdx_y];
    y = ya[hipBlockIdx_y];
    
    if(incx < 0)
        x -= ssize_t(incx) * (n - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (m - 1);

    auto alpha              = load_scalar(alpha_device_host);
    auto beta               = load_scalar(beta_device_host);
    
    gemvn_kernel<DIM_X,DIM_Y>(m,n,alpha,A,lda,x,incx,beta,y,incy);

}

template <rocblas_int NB_X, typename T, typename U>
__global__ void gemvc_kernel_batched(rocblas_int m,
                            rocblas_int n,
                            U alpha_device_host,
                            const T *const Aa[],
                            rocblas_int lda,
                            const T *const xa[],
                            rocblas_int incx,
                            U beta_device_host,
                            T *const ya[],
                            rocblas_int incy)
{
    const T* A;
    const T* x;
    T* y;
    A = Aa[hipBlockIdx_y];
    x = xa[hipBlockIdx_y];
    y = ya[hipBlockIdx_y];

    if(incx < 0)
        x -= ssize_t(incx) * (m - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (n - 1);
    
    auto alpha     = load_scalar(alpha_device_host);
    auto beta      = load_scalar(beta_device_host);
    
    gemvc_kernel<NB_X>(m,n,alpha,A,lda,x,incx,beta,y,incy);

}

#endif
