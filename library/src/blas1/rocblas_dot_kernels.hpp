/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#pragma once

#include "rocblas_block_sizes.h"
#include "rocblas_dot.hpp"

template <typename T>
constexpr int rocblas_dot_one_block_threshold()
{
    int single_block_threshold = 32768;
    if(std::is_same_v<T, float>)
        single_block_threshold = 31000;
    else if(std::is_same_v<T, rocblas_float_complex>)
        single_block_threshold = 16000;
    else if(std::is_same_v<T, double>)
        single_block_threshold = 13000;
    else if(std::is_same_v<T, rocblas_double_complex>)
        single_block_threshold = 10000;
    return single_block_threshold;
}

template <bool ONE_BLOCK, typename V, typename T>
__inline__ __device__ void
    rocblas_dot_save_sum(V sum, V* __restrict__ workspace, T* __restrict__ out)
{
    if(threadIdx.x == 0)
    {
        if(ONE_BLOCK || gridDim.x == 1) // small N avoid second kernel
            out[blockIdx.y] = T(sum);
        else
            workspace[blockIdx.x + size_t(blockIdx.y) * gridDim.x] = sum;
    }
}

template <bool ONE_BLOCK, int NB, int WIN, bool CONJ, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1(rocblas_int n,
                        const U __restrict__ xa,
                        rocblas_stride shiftx,
                        rocblas_stride stridex,
                        const U __restrict__ ya,
                        rocblas_stride shifty,
                        rocblas_stride stridey,
                        V* __restrict__ workspace,
                        T* __restrict__ out)
{
    const auto* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const auto* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = !ONE_BLOCK ? blockDim.x * gridDim.x : blockDim.x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
    {
        sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
    }

    sum = rocblas_dot_block_reduce<NB>(sum);

    rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <bool ONE_BLOCK, int NB, int WIN, bool CONJ, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1by2(rocblas_int n,
                           const U __restrict__ xa,
                           rocblas_stride shiftx,
                           rocblas_stride stridex,
                           const U __restrict__ ya,
                           rocblas_stride shifty,
                           rocblas_stride stridey,
                           V* __restrict__ workspace,
                           T* __restrict__ out)
{
    const auto* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const auto* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    V   sum = 0;
    int i   = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

    // sum WIN elements per thread
    int inc = !ONE_BLOCK ? blockDim.x * gridDim.x : blockDim.x;

    if constexpr(
        std::is_same_v<
            T,
            rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, rocblas_float>)
    {
        i *= 2;
        inc *= 2;
        for(int j = 0; j < WIN && i < n - 1; j++, i += inc)
        {
#pragma unroll
            for(int k = 0; k < 2; ++k)
            {
                sum += V(y[i + k]) * V(CONJ ? conj(x[i + k]) : x[i + k]);
            }
        }
        // If `n` is odd then the computation of last element is covered below.
        if(n % 2 && i == n - 1)
        {
            sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
        }
    }
    else
    {
        for(int j = 0; j < WIN && i < n; j++, i += inc)
        {
            sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
        }
    }

    sum = rocblas_dot_block_reduce<NB>(sum);

    rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <typename API_INT,
          bool ONE_BLOCK,
          int  NB,
          int  WIN,
          bool CONJ,
          typename T,
          typename U,
          typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel(rocblas_int n,
                   const U __restrict__ xa,
                   rocblas_stride shiftx,
                   API_INT        incx,
                   rocblas_stride stridex,
                   const U __restrict__ ya,
                   rocblas_stride shifty,
                   API_INT        incy,
                   rocblas_stride stridey,
                   V* __restrict__ workspace,
                   T* __restrict__ out)
{
    const auto* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const auto* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = blockDim.x * gridDim.x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
    {
        sum += V(y[i * int64_t(incy)])
               * V(CONJ ? conj(x[i * int64_t(incx)]) : x[i * int64_t(incx)]);
    }
    sum = rocblas_dot_block_reduce<NB>(sum);

    rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <typename API_INT,
          bool ONE_BLOCK,
          int  NB,
          int  WIN,
          bool CONJ,
          typename T,
          typename U,
          typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_magsq(rocblas_int n,
                         const U __restrict__ xa,
                         rocblas_stride shiftx,
                         API_INT        incx,
                         rocblas_stride stridex,
                         V* __restrict__ workspace,
                         T* __restrict__ out)
{
    const auto* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);

    int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = blockDim.x * gridDim.x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
    {
        int64_t idx = i * int64_t(incx);
        sum += V(x[idx]) * V(CONJ ? conj(x[idx]) : x[idx]);
    }
    sum = rocblas_dot_block_reduce<NB>(sum);

    rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <int NB, int WIN, typename V, typename T = V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_reduce(int n_sums, V* __restrict__ in, T* __restrict__ out)
{
    V sum = 0;

    size_t offset = size_t(blockIdx.y) * n_sums;
    in += offset;

    int inc = blockDim.x * gridDim.x * WIN;

    int i         = threadIdx.x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += in[i + j];
    }
    if(threadIdx.x < remainder)
    {
        sum += in[n_sums - 1 - threadIdx.x];
    }

    sum = rocblas_dot_block_reduce<NB>(sum);
    if(threadIdx.x == 0)
        out[blockIdx.y] = T(sum);
}

// assume workspace has already been allocated, recommended for repeated calling of dot_strided_batched product
// routine
template <typename API_INT, int NB, bool CONJ, typename T, typename U, typename V>
rocblas_status rocblas_internal_dot_launcher(rocblas_handle __restrict__ handle,
                                             API_INT n,
                                             const U __restrict__ x,
                                             rocblas_stride offsetx,
                                             API_INT        incx,
                                             rocblas_stride stridex,
                                             const U __restrict__ y,
                                             rocblas_stride offsety,
                                             API_INT        incy,
                                             rocblas_stride stridey,
                                             API_INT        batch_count,
                                             T* __restrict__ results,
                                             V* __restrict__ workspace)
{

    // One or two kernels are used to finish the reduction
    // kernel 1 write partial results per thread block in workspace, number of partial results is blocks
    // kernel 2 if blocks > 1 the partial results in workspace are reduced to output

    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode && batch_count > 0)
        {
            RETURN_IF_HIP_ERROR(
                hipMemsetAsync(&results[0], 0, batch_count * sizeof(T), handle->get_stream()));
        }
        else
        {
            for(int i = 0; i < batch_count; i++)
            {
                results[i] = T(0);
            }
        }

        return rocblas_status_success;
    }

    static constexpr int WIN = rocblas_dot_WIN<T>();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    int64_t shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
    int64_t shifty = incy < 0 ? offsety - int64_t(incy) * (n - 1) : offsety;

    static constexpr int single_block_threshold = rocblas_dot_one_block_threshold<T>();

    if(n <= single_block_threshold)
    {
        // we only reduce the block count to 1 so safe to ignore extra workspace allocated in caller

        static constexpr int NB_OB  = 1024;
        static constexpr int WIN_OB = 32; // 32K max n threshold, assert guard below

        rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB_OB * WIN_OB);
        assert(blocks == 1);
        static constexpr bool ONE_BLOCK = true;

        dim3 grid(blocks, batch_count);
        dim3 threads(NB_OB);

        T* output = results; // device mode output directly to results
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            size_t offset = size_t(batch_count) * blocks;
            output        = (T*)(workspace + offset);
        }

        if(x != y || incx != incy || offsetx != offsety || stridex != stridey)
        {
            if(incx == 1 && incy == 1)
            {
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_dot_kernel_inc1by2<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
                    grid,
                    threads,
                    0,
                    handle->get_stream(),
                    n,
                    x,
                    shiftx,
                    stridex,
                    y,
                    shifty,
                    stridey,
                    workspace,
                    output);
            }
            else
            {
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_dot_kernel<API_INT, ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
                    grid,
                    threads,
                    0,
                    handle->get_stream(),
                    n,
                    x,
                    shiftx,
                    incx,
                    stridex,
                    y,
                    shifty,
                    incy,
                    stridey,
                    workspace,
                    output);
            }
        }
        else // x dot x
        {
            ROCBLAS_LAUNCH_KERNEL(
                (rocblas_dot_kernel_magsq<API_INT, ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
                grid,
                threads,
                0,
                handle->get_stream(),
                n,
                x,
                shiftx,
                incx,
                stridex,
                workspace,
                output);
        }

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
    }
    else
    {
        static constexpr bool ONE_BLOCK = false;

        rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB * WIN);
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);

        T* output = results;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            size_t offset = size_t(batch_count) * blocks;
            output        = (T*)(workspace + offset);
        }

        if(x != y || incx != incy || offsetx != offsety || stridex != stridey)
        {
            if(incx == 1 && incy == 1)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_inc1<ONE_BLOCK, NB, WIN, CONJ, T>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      x,
                                      shiftx,
                                      stridex,
                                      y,
                                      shifty,
                                      stridey,
                                      workspace,
                                      output);
            }
            else
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel<API_INT, ONE_BLOCK, NB, WIN, CONJ, T>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      x,
                                      shiftx,
                                      incx,
                                      stridex,
                                      y,
                                      shifty,
                                      incy,
                                      stridey,
                                      workspace,
                                      output);
            }
        }
        else // x dot x
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_magsq<API_INT, ONE_BLOCK, NB, WIN, CONJ, T>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  x,
                                  shiftx,
                                  incx,
                                  stridex,
                                  workspace,
                                  output);
        }

        if(blocks > 1) // if single block first kernel did all work
            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_reduce<NB, WIN>),
                                  dim3(1, batch_count),
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  blocks,
                                  workspace,
                                  output);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
    }
    return rocblas_status_success;
}

// for ex interface and _64 reuse
#ifdef INST_DOT_EX_LAUNCHER
#error INST_DOT_EX_LAUNCHER already defined
#endif

#define INST_DOT_EX_LAUNCHER(API_INT_, NB_, CONJ_, T_, U_, V_)           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status             \
        rocblas_internal_dot_launcher<API_INT_, NB_, CONJ_, T_, U_, V_>( \
            rocblas_handle __restrict__ handle,                          \
            API_INT_ n,                                                  \
            U_ __restrict__ x,                                           \
            rocblas_stride offsetx,                                      \
            API_INT_       incx,                                         \
            rocblas_stride stridex,                                      \
            U_ __restrict__ y,                                           \
            rocblas_stride offsety,                                      \
            API_INT_       incy,                                         \
            rocblas_stride stridey,                                      \
            API_INT_       batch_count,                                  \
            T_* __restrict__ results,                                    \
            V_* __restrict__ workspace);
