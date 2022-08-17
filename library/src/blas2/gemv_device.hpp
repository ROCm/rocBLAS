/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * Copyright (c) 2012-, King Abdullah University of Science and Technology
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ************************************************************************ */

#pragma once

// uses dot shuffle reductions
#include "../blas1/rocblas_reduction.hpp"
// uses recursive folding reduction
#include "../blas1/reduction.hpp"

template <rocblas_int NB, typename T, typename Ta, typename Tx>
ROCBLAS_KERNEL(NB)
gemv_scal_kernel(rocblas_int    n,
                 Ta             beta_device_host,
                 rocblas_stride stride_beta,
                 Tx             ya,
                 rocblas_stride offset_y,
                 rocblas_int    incy,
                 rocblas_stride stride_y)
{
    auto* __restrict__ y = load_ptr_batch(ya, blockIdx.y, offset_y, stride_y);
    auto beta            = load_scalar(beta_device_host, blockIdx.y, stride_beta);
    if(beta == 1)
        return;
    ptrdiff_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // bound
    if(tid < n)
    {
        if(beta == 0)
        {
            y[tid * incy] = T(0);
        }
        else
        {
            y[tid * incy] = y[tid * incy] * beta;
        }
    }
}

template <int DIM_X,
          int DIM_Y,
          int elements_per_thread,
          typename T,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCBLAS_KERNEL_ILF void gemvn_double_buffered_kernel_calc(rocblas_int rows,
                                                          rocblas_int cols,
                                                          T           alpha,
                                                          const T* __restrict__ A,
                                                          rocblas_int lda,
                                                          const T* __restrict__ x,
                                                          rocblas_int incx,
                                                          T* __restrict__ y,
                                                          rocblas_int incy)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);

    T res_1_ = T(0);
    T res_2_ = T(0);
    T areg_upper[elements_per_thread];
    T areg_lower[elements_per_thread];

    __shared__ T la[DIM_X * (2 * DIM_Y)];

    int count = (cols / DIM_X) / gridDim.y + (by < (cols / DIM_X) % gridDim.y);
    {
        int start = by * ((cols / DIM_X) / gridDim.y) + min(by, (cols / DIM_X) % gridDim.y);

        // Advance 'A'
        A += DIM_X * bx;
        A += start * DIM_X * size_t(lda);

        // Advance 'x'
        x += start * DIM_X * incx;

        // Advance 'y'
        y += (bx * DIM_X) * incy;
    }

    if(count == 0)
        return;

    const int j = ty_ * elements_per_thread * lda + tx_;

// read upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        areg_upper[k] = A[j + k * lda];

    int Vblocks = 0;
    //#pragma unroll
    for(Vblocks = 0; Vblocks < count; Vblocks++)
    {
// read lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            areg_lower[k] = A[(DIM_X / 2) + j + k * lda];

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            res_1_ += areg_upper[k] * x[(ty_ * elements_per_thread + k) * incx];

        A += DIM_X * lda;

        // read upper from next block
        if(Vblocks != count - 1)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                areg_upper[k] = A[j + k * lda];
        }

// compute lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            res_2_ += areg_lower[k] * x[(ty_ * elements_per_thread + k) * incx];

        x += DIM_X * incx;
    }

    la[ty_ * DIM_X + tx_]               = res_1_;
    la[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
    __syncthreads();

    if(ty == 0)
    {
        res_1_ = T(0);
#pragma unroll
        for(int k = 0; k < 2 * DIM_Y; k++)
            res_1_ += la[k * DIM_X + tx];

        atomicAdd(&y[tx * incy], (alpha * res_1_));
    }
}

template <int DIM_X,
          int DIM_Y,
          int elements_per_thread,
          typename T,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCBLAS_KERNEL_ILF void
    gemvn_double_buffered_generic_kernel_calc(rocblas_int rows,
                                              rocblas_int cols,
                                              T           alpha,
                                              const T*    A,
                                              rocblas_int lda,
                                              const T* __restrict__ x,
                                              rocblas_int incx,
                                              T* __restrict__ y,
                                              rocblas_int       incy,
                                              const rocblas_int irregular_cols,
                                              const rocblas_int rows_mod_gemv_bs,
                                              const rocblas_int cols_mod_gemv_bs)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);

    T res_1_                          = T(0);
    T res_2_                          = T(0);
    T areg_upper[elements_per_thread] = {T(0)};
    T areg_lower[elements_per_thread] = {T(0)};

    __shared__ T la[DIM_X * (2 * DIM_Y)];
    __shared__ T xbuff[DIM_X]; // used for the last irregular part of x

    int count = (cols / DIM_X) / gridDim.y + (by < (cols / DIM_X) % gridDim.y);

    {
        int start = by * ((cols / DIM_X) / gridDim.y) + min(by, (cols / DIM_X) % gridDim.y);

        // Advance 'A'
        A += DIM_X * bx;
        A += start * DIM_X * size_t(lda);

        // Advance 'x'
        x += start * DIM_X * incx;

        // Advance 'y'
        y += (bx * DIM_X) * incy;
    }

    const T* A_ = A;
    if(by != gridDim.y - 1)
    {
        if(count == 0)
            return;
    }

    // test special case, when rows mod block_size is zero
    if(rows_mod_gemv_bs == 0)
    {
        if(bx == gridDim.x - 1)
            return;
    }

    // load the last segment of x
    if(by == gridDim.y - 1)
    {
        if(cols_mod_gemv_bs != 0)
        {
            if(ty == 0)
            {
                if(tx < cols_mod_gemv_bs)
                    xbuff[tx] = x[(count * DIM_X + tx) * incx];
                else
                    xbuff[tx] = T(0);
            }
        }
    }

    const int j = ty_ * elements_per_thread * lda + tx_;

    __syncthreads();

    if(count > 0)
    {
        // read upper
        if(bx == gridDim.x - 1)
        {
            if(tx_ < rows_mod_gemv_bs)
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    areg_upper[k] = A[j + k * lda];
            }
        }
        else
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                areg_upper[k] = A[j + k * lda];
        }
    }

    // -- Main Loop
    int Vblocks = 0;
    //#pragma unroll
    for(Vblocks = 0; Vblocks < count; Vblocks++)
    {
        // read lower
        if(bx == gridDim.x - 1)
        {
            if(tx_ + (DIM_X / 2) < rows_mod_gemv_bs)
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    areg_lower[k] = A[(DIM_X / 2) + j + k * lda];
            }
        }
        else
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                areg_lower[k] = A[(DIM_X / 2) + j + k * lda];
        }

        // compute upper
        if(bx == gridDim.x - 1)
        {
            if(tx_ < rows_mod_gemv_bs)
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    res_1_ += areg_upper[k] * x[(ty_ * elements_per_thread + k) * incx];
            }
        }
        else
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                res_1_ += areg_upper[k] * x[(ty_ * elements_per_thread + k) * incx];
        }

        A += DIM_X * lda;

        // read upper from next block
        if(Vblocks != count - 1)
        {
            if(bx == gridDim.x - 1)
            {
                if(tx_ < rows_mod_gemv_bs)
                {
#pragma unroll
                    for(int k = 0; k < elements_per_thread; k++)
                        areg_upper[k] = A[j + k * lda];
                }
            }
            else
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    areg_upper[k] = A[j + k * lda];
            }
        }

        // compute lower
        if(bx == gridDim.x - 1)
        {
            if(tx_ + (DIM_X / 2) < rows_mod_gemv_bs)
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    res_2_ += areg_lower[k] * x[(ty_ * elements_per_thread + k) * incx];
            }
        }
        else
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                res_2_ += areg_lower[k] * x[(ty_ * elements_per_thread + k) * incx];
        }
        x += DIM_X * incx;
    } // end of main loop

    //////////////////
    // process last irregular tile

    if((cols_mod_gemv_bs != 0) && (by == gridDim.y - 1))
    {
        {
            int offset = count * DIM_X * size_t(lda);
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
            {
                areg_upper[k] = T(0);
                areg_lower[k] = T(0);
            }

            const int num_active_thread_cols = cols_mod_gemv_bs / elements_per_thread;

            //load upper
            if(bx == gridDim.x - 1)
            {
                if(ty_ < num_active_thread_cols)
                {
                    if(tx_ < rows_mod_gemv_bs)
                    {
#pragma unroll
                        for(int k = 0; k < elements_per_thread; k++)
                            areg_upper[k] = A_[offset + j + k * lda];
                    }
                }
                else if(ty_ == num_active_thread_cols)
                {
                    if(tx_ < rows_mod_gemv_bs)
                    {
#pragma unroll
                        for(int k = 0; k < irregular_cols; k++)
                            areg_upper[k] = A_[offset + j + k * lda];
                    }
                }
            }
            else
            {
                if(ty_ < num_active_thread_cols)
                {
#pragma unroll
                    for(int k = 0; k < elements_per_thread; k++)
                        areg_upper[k] = A_[offset + j + k * lda];
                }
                else if(ty_ == num_active_thread_cols)
                {
#pragma unroll
                    for(int k = 0; k < irregular_cols; k++)
                        areg_upper[k] = A_[offset + j + k * lda];
                }
            }

            // load lower
            if(bx == gridDim.x - 1)
            {
                if(ty_ < num_active_thread_cols)
                {
                    if(tx_ + (DIM_X / 2) < rows_mod_gemv_bs)
                    {
#pragma unroll
                        for(int k = 0; k < elements_per_thread; k++)
                            areg_lower[k] = A_[offset + j + k * lda + (DIM_X / 2)];
                    }
                }
                else if(ty_ == num_active_thread_cols)
                {
                    if(tx_ + (DIM_X / 2) < rows_mod_gemv_bs)
                    {
#pragma unroll
                        for(int k = 0; k < irregular_cols; k++)
                            areg_lower[k] = A_[offset + j + k * lda + (DIM_X / 2)];
                    }
                }
            }
            else
            {
                if(ty_ < num_active_thread_cols)
                {
#pragma unroll
                    for(int k = 0; k < elements_per_thread; k++)
                        areg_lower[k] = A_[offset + j + k * lda + (DIM_X / 2)];
                }
                else if(ty_ == num_active_thread_cols)
                {
#pragma unroll
                    for(int k = 0; k < irregular_cols; k++)
                        areg_lower[k] = A_[offset + j + k * lda + (DIM_X / 2)];
                }
            }
        } // end of if by == gridDim.x-1

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            res_1_ += areg_upper[k] * xbuff[ty_ * elements_per_thread + k];

// compute lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            res_2_ += areg_lower[k] * xbuff[ty_ * elements_per_thread + k];

    } // end of if  cols_mod_gemv_bs != 0

    // final reduction
    la[ty_ * DIM_X + tx_]               = res_1_;
    la[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
    __syncthreads();

    if(ty == 0)
    {
        res_1_ = T(0);
#pragma unroll
        for(int k = 0; k < 2 * DIM_Y; k++)
            res_1_ += la[k * DIM_X + tx];

        if(bx == gridDim.x - 1)
        {
            if(tx < rows_mod_gemv_bs)
                atomicAdd(&y[tx * incy], (res_1_ * alpha));
        }
        else
        {
            atomicAdd(&y[tx * incy], (res_1_ * alpha));
        }
    }
}

template <int DIM_X,
          int DIM_Y,
          int elements_per_thread,
          typename T,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCBLAS_KERNEL_ILF void gemvn_double_buffered_kernel_calc(rocblas_int rows,
                                                          rocblas_int cols,
                                                          T           alpha,
                                                          const T* __restrict__ A,
                                                          rocblas_int lda,
                                                          const T* __restrict__ x,
                                                          rocblas_int incx,
                                                          T* __restrict__ y,
                                                          rocblas_int incy)
{
}

template <int DIM_X,
          int DIM_Y,
          int elements_per_thread,
          typename T,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCBLAS_KERNEL_ILF void
    gemvn_double_buffered_generic_kernel_calc(rocblas_int rows,
                                              rocblas_int cols,
                                              T           alpha,
                                              const T*    A,
                                              rocblas_int lda,
                                              const T* __restrict__ x,
                                              rocblas_int incx,
                                              T* __restrict__ y,
                                              rocblas_int       incy,
                                              const rocblas_int irregular_cols,
                                              const rocblas_int rows_mod_gemv_bs,
                                              const rocblas_int cols_mod_gemv_bs)
{
}

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
    rocblas_int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    if(!alpha)
    {
        if(thread_id < DIM_X * 4)
        {
            rocblas_int ind = blockIdx.x * DIM_X * 4 + thread_id;
            if(ind < m)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = threadIdx.x;
    rocblas_int ty = threadIdx.y;

    rocblas_int ind;

    __shared__ T sdata[DIM_X * 4 * DIM_Y];

    T res_A[4];
    T res_x[4];

    res_A[0] = res_A[1] = res_A[2] = res_A[3] = T{0};

    ind = blockIdx.x * DIM_X * 4 + tx;

    rocblas_int n_tail = n % (4 * DIM_Y);
    rocblas_int col;

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

        ind = blockIdx.x * DIM_X * 4 + thread_id;

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
    rocblas_int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    if(!alpha)
    {
        if(thread_id < DIM_X)
        {
            rocblas_int ind = blockIdx.x * DIM_X + thread_id;
            if(ind < m)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = blockIdx.x * DIM_X + tx;

    __shared__ rocblas_double_complex sdata[DIM_X * DIM_Y];

    rocblas_double_complex res_A;
    rocblas_double_complex res_x;

    res_A = res_x = rocblas_double_complex{0, 0};

    rocblas_int n_tail = n % (DIM_Y);
    rocblas_int col;

    for(col = ty; col < (n - n_tail); col += DIM_Y)
    {

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

        ind = blockIdx.x * DIM_X + thread_id;

        if(ind < m)
        {
            y[ind * incy]
                = beta ? alpha * sdata[thread_id] + beta * y[ind * incy] : alpha * sdata[thread_id];
        }
    }
}

template <bool CONJ, rocblas_int NB_X, typename T, typename U>
ROCBLAS_KERNEL_ILF void gemvt_kernel_calc(rocblas_int m,
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
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

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
template <bool CONJ, rocblas_int NB_X, typename T, typename U>
ROCBLAS_KERNEL_ILF void gemvt_warp_reduce_kernel_calc(rocblas_int m,
                                                      rocblas_int n,
                                                      U           alpha,
                                                      const T* __restrict__ A,
                                                      rocblas_int lda,
                                                      const T* __restrict__ x,
                                                      rocblas_int incx,
                                                      U           beta,
                                                      T* __restrict__ y,
                                                      rocblas_int incy)
{
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

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

    rocblas_int tx = threadIdx.x;

    // offset blocks * cols * batch
    workspace += size_t(gridDim.x) * n * blockIdx.y;

    // We need to short-circuit if alpha==0 and not propagate NaNs
    if(!alpha)
    {
        if(tx == 0)
            for(int i = 0; i < n; i++)
                workspace[blockIdx.x + size_t(i) * gridDim.x] = 0;
        return;
    }

    int row = tx * WIN + blockIdx.x * NB_X * WIN;
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
                workspace[blockIdx.x + size_t(k + i) * gridDim.x] = alpha * sum[k];
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
            workspace[blockIdx.x + size_t(i) * gridDim.x] = alpha * sum[0];
    }
}

template <rocblas_int NB, rocblas_int WIN, typename T, typename U, typename W>
ROCBLAS_KERNEL(NB)
rocblas_gemvt_sn_reduce(rocblas_int    n_sums,
                        U              beta_device_host,
                        rocblas_stride stride_beta,
                        W* __restrict__ ya,
                        rocblas_stride shifty,
                        rocblas_int    incy,
                        rocblas_stride stridey,
                        T* __restrict__ workspace)
{
    T*   y    = load_ptr_batch(ya, blockIdx.z, shifty, stridey);
    auto beta = load_scalar(beta_device_host, blockIdx.z, stride_beta);

    T sum{0};

    size_t offset = size_t(n_sums) * (gridDim.y * blockIdx.z + blockIdx.y);
    workspace += offset;

    int inc = blockDim.x * WIN;

    int i         = threadIdx.x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += workspace[i + j];
    }
    if(threadIdx.x < remainder)
    {
        sum += workspace[n_sums - 1 - threadIdx.x];
    }
    sum = rocblas_dot_block_reduce<NB>(sum);

    if(threadIdx.x == 0)
    {
        y[ptrdiff_t(blockIdx.y) * incy]
            = beta ? (y[ptrdiff_t(blockIdx.y) * incy] * beta) + sum : sum;
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

    rocblas_int tx = threadIdx.x;

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
          rocblas_int elements_per_thread,
          typename T,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
gemvn_double_buffered_kernel(rocblas_int    m,
                             rocblas_int    n,
                             U              alpha_device_host,
                             rocblas_stride stride_alpha,
                             const V*       Aa,
                             rocblas_stride shifta,
                             rocblas_int    lda,
                             rocblas_stride strideA,
                             const V*       xa,
                             rocblas_stride shiftx,
                             rocblas_int    incx,
                             rocblas_stride stridex,
                             W*             ya,
                             rocblas_stride shifty,
                             rocblas_int    incy,
                             rocblas_stride stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    assert(DIM_X * DIM_Y == num_threads);

    auto alpha = load_scalar(alpha_device_host, blockIdx.z, stride_alpha);

    if(!alpha)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.z, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.z, shiftx, stridex);

    T* y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    gemvn_double_buffered_kernel_calc<DIM_X, DIM_Y, elements_per_thread>(
        m, n, alpha, A, lda, x, incx, y, incy);
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          typename T,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
gemvn_double_buffered_generic_kernel(rocblas_int    m,
                                     rocblas_int    n,
                                     U              alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     const V*       Aa,
                                     rocblas_stride shifta,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     const V*       xa,
                                     rocblas_stride shiftx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     W*             ya,
                                     rocblas_stride shifty,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    irregular_cols,
                                     rocblas_int    rows_mod_gemv_bs,
                                     rocblas_int    cols_mod_gemv_bs)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    assert(DIM_X * DIM_Y == num_threads);

    auto alpha = load_scalar(alpha_device_host, blockIdx.z, stride_alpha);

    if(!alpha)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.z, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.z, shiftx, stridex);

    T* y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    gemvn_double_buffered_generic_kernel_calc<DIM_X, DIM_Y, elements_per_thread, T>(
        m, n, alpha, A, lda, x, incx, y, incy, irregular_cols, rows_mod_gemv_bs, cols_mod_gemv_bs);
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          typename T_lda,
          typename T,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
gemvn_kernel(rocblas_int    m,
             rocblas_int    n,
             U              alpha_device_host,
             rocblas_stride stride_alpha,
             const V*       Aa,
             rocblas_stride shifta,
             T_lda          lda,
             rocblas_stride strideA,
             const V*       xa,
             rocblas_stride shiftx,
             rocblas_int    incx,
             rocblas_stride stridex,
             U              beta_device_host,
             rocblas_stride stride_beta,
             W*             ya,
             rocblas_stride shifty,
             rocblas_int    incy,
             rocblas_stride stridey)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, blockIdx.y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.y, shiftx, stridex);

    T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    gemvn_kernel_calc<DIM_X, DIM_Y, T_lda>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// lda always cast to size_t so single kernel
template <bool CONJ, rocblas_int NB_X, typename T, typename U, typename V, typename W>
ROCBLAS_KERNEL(NB_X)
gemvt_kernel(rocblas_int    m,
             rocblas_int    n,
             U              alpha_device_host,
             rocblas_stride stride_alpha,
             const V*       Aa,
             rocblas_stride shifta,
             rocblas_int    lda,
             rocblas_stride strideA,
             const V*       xa,
             rocblas_stride shiftx,
             rocblas_int    incx,
             rocblas_stride stridex,
             U              beta_device_host,
             rocblas_stride stride_beta,
             W*             ya,
             rocblas_stride shifty,
             rocblas_int    incy,
             rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, blockIdx.y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.y, shiftx, stridex);

    T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    gemvt_kernel_calc<CONJ, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

//Optimized kernel for GEMV transpose case when m or n is less than 6000
template <bool CONJ, rocblas_int NB_X, typename T, typename U, typename V, typename W>
ROCBLAS_KERNEL(NB_X)
gemvt_warp_reduce_kernel(rocblas_int    m,
                         rocblas_int    n,
                         U              alpha_device_host,
                         rocblas_stride stride_alpha,
                         const V*       Aa,
                         rocblas_stride shifta,
                         rocblas_int    lda,
                         rocblas_stride strideA,
                         const V*       xa,
                         rocblas_stride shiftx,
                         rocblas_int    incx,
                         rocblas_stride stridex,
                         U              beta_device_host,
                         rocblas_stride stride_beta,
                         W*             ya,
                         rocblas_stride shifty,
                         rocblas_int    incy,
                         rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, blockIdx.y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.y, shiftx, stridex);

    T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    gemvt_warp_reduce_kernel_calc<CONJ, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <bool        CONJ,
          rocblas_int NB_X,
          rocblas_int WIN,
          typename T_lda,
          typename T,
          typename U,
          typename V>
ROCBLAS_KERNEL(NB_X)
gemvt_sn_kernel(rocblas_int    m,
                rocblas_int    n,
                U              alpha_device_host,
                rocblas_stride stride_alpha,
                const V*       Aa,
                rocblas_stride shifta,
                T_lda          lda,
                rocblas_stride strideA,
                const V*       xa,
                rocblas_stride shiftx,
                rocblas_int    incx,
                rocblas_stride stridex,
                T*             workspace)
{
    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.y, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.y, shiftx, stridex);

    gemvt_sn_kernel_calc<CONJ, NB_X, WIN, T_lda>(m, n, alpha, A, lda, x, incx, workspace);
}

template <bool CONJ, rocblas_int NB_X, typename T, typename U, typename V, typename W>
ROCBLAS_KERNEL(NB_X)
gemvtsm_kernel(rocblas_int    m,
               rocblas_int    n,
               U              alpha_device_host,
               rocblas_stride stride_alpha,
               const V*       Aa,
               rocblas_stride shifta,
               rocblas_int    lda,
               rocblas_stride strideA,
               const V*       xa,
               rocblas_stride shiftx,
               rocblas_int    incx,
               rocblas_stride stridex,
               U              beta_device_host,
               rocblas_stride stride_beta,
               W*             ya,
               rocblas_stride shifty,
               rocblas_int    incy,
               rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, blockIdx.x, stride_alpha);
    auto beta  = load_scalar(beta_device_host, blockIdx.x, stride_beta);

    if(!alpha && beta == 1)
        return;

    // batch in blockIdx.x not y
    const T* A = cond_load_ptr_batch(alpha, Aa, blockIdx.x, shifta, strideA);
    const T* x = cond_load_ptr_batch(alpha, xa, blockIdx.x, shiftx, stridex);

    T* y = load_ptr_batch(ya, blockIdx.x, shifty, stridey);

    gemvtsm_kernel_calc<CONJ, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}
