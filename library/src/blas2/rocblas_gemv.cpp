/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"
#include "../blas1/reduction.h"

namespace {

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
__global__ void gemvn_kernel(rocblas_int m,
                             rocblas_int n,
                             U alpha_device_host,
                             const T* __restrict__ A,
                             rocblas_int lda,
                             const T* __restrict__ x,
                             rocblas_int incx,
                             U beta_device_host,
                             T* y,
                             rocblas_int incy)
{
    auto alpha              = load_scalar(alpha_device_host);
    auto beta               = load_scalar(beta_device_host);
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

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
            y[ind * incy] = alpha * sdata[thread_id] + beta * y[ind * incy];
    }
}

template <rocblas_int NB_X, typename T, typename U>
__global__ void gemvc_kernel(rocblas_int m,
                             rocblas_int n,
                             U alpha_device_host,
                             const T* __restrict__ A,
                             rocblas_int lda,
                             const T* __restrict__ x,
                             rocblas_int incx,
                             U beta_device_host,
                             T* y,
                             rocblas_int incy)
{
    auto alpha     = load_scalar(alpha_device_host);
    auto beta      = load_scalar(beta_device_host);
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
        y[col * incy] = alpha * sdata[0] + beta * y[col * incy];
}

template <typename>
constexpr char rocblas_gemv_name[] = "unknown";
template <>
constexpr char rocblas_gemv_name<float>[] = "rocblas_sgemv";
template <>
constexpr char rocblas_gemv_name<double>[] = "rocblas_dgemv";

/*! \brief BLAS Level 2 API

    \details
    xGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    trans     rocblas_operation
    @param[in]
    m         rocblas_int
    @param[in]
    n         rocblas_int
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[in]
    beta      specifies the scalar beta.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template <typename T>
rocblas_status rocblas_gemv(rocblas_handle handle,
                            rocblas_operation transA,
                            rocblas_int m,
                            rocblas_int n,
                            const T* alpha,
                            const T* A,
                            rocblas_int lda,
                            const T* x,
                            rocblas_int incx,
                            const T* beta,
                            T* y,
                            rocblas_int incy)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;
    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto transA_letter = rocblas_transpose_letter(transA);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemv_name<T>,
                          transA,
                          m,
                          n,
                          *alpha,
                          A,
                          lda,
                          x,
                          incx,
                          *beta,
                          y,
                          incy);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f gemv -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          transA_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--alpha",
                          *alpha,
                          "--lda",
                          lda,
                          "--incx",
                          incx,
                          "--beta",
                          *beta,
                          "--incy",
                          incy);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemv_name<T>,
                          transA,
                          m,
                          n,
                          alpha,
                          A,
                          lda,
                          x,
                          incx,
                          beta,
                          y,
                          incy);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_gemv_name<T>,
                        "transA",
                        transA_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "lda",
                        lda,
                        "incx",
                        incx,
                        "incy",
                        incy);
    }

    if(!A || !x || !y)
        return rocblas_status_invalid_pointer;

    if(m < 0 || n < 0 || lda < m || lda < 1 || !incx || !incy)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if(!m || !n)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(transA == rocblas_operation_none)
    {
        // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int GEMVN_DIM_X = 64;
        static constexpr int GEMVN_DIM_Y = 16;
        rocblas_int blocks               = (m - 1) / (GEMVN_DIM_X * 4) + 1;

        dim3 gemvn_grid(blocks);
        dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

        if(incx < 0)
            x -= ssize_t(incx) * (n - 1);
        if(incy < 0)
            y -= ssize_t(incy) * (m - 1);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               x,
                               incx,
                               beta,
                               y,
                               incy);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL((gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y>),
                               gemvn_grid,
                               gemvn_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               x,
                               incx,
                               *beta,
                               y,
                               incy);
        }
    }
    else
    {
        // transpose
        // number of columns on the y-dim of the grid, using gemvc because gemvt(transpose) is a
        // instance of gemvc (conjugate)
        static constexpr int NB = 256;
        dim3 gemvc_grid(n);
        dim3 gemvc_threads(NB);

        if(incx < 0)
            x -= ssize_t(incx) * (m - 1);
        if(incy < 0)
            y -= ssize_t(incy) * (n - 1);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL(gemvc_kernel<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               x,
                               incx,
                               beta,
                               y,
                               incy);
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;

            hipLaunchKernelGGL(gemvc_kernel<NB>,
                               gemvc_grid,
                               gemvc_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               x,
                               incx,
                               *beta,
                               y,
                               incy);
        }
    }
    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sgemv(rocblas_handle handle,
                             rocblas_operation transA,
                             rocblas_int m,
                             rocblas_int n,
                             const float* alpha,
                             const float* A,
                             rocblas_int lda,
                             const float* x,
                             rocblas_int incx,
                             const float* beta,
                             float* y,
                             rocblas_int incy)
{
    return rocblas_gemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

rocblas_status rocblas_dgemv(rocblas_handle handle,
                             rocblas_operation transA,
                             rocblas_int m,
                             rocblas_int n,
                             const double* alpha,
                             const double* A,
                             rocblas_int lda,
                             const double* x,
                             rocblas_int incx,
                             const double* beta,
                             double* y,
                             rocblas_int incy)
{
    return rocblas_gemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

} // extern "C"
