/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas1/reduction.hpp"
#include "blas1/rocblas_iamax_iamin_kernels.hpp"
#include "blas1/rocblas_reduction.hpp"

#include "rocblas_iamax_iamin_64.hpp" // rocblas_int API called

// iamax, iamin kernels

template <int N, typename REDUCE, typename T>
__inline__ __device__ rocblas_index_64_value_t<T>
                      rocblas_wavefront_reduce_method_64(rocblas_index_64_value_t<T> x)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        rocblas_index_64_value_t<T> y{};
        y.index = __shfl_down(x.index, offset);
        y.value = __shfl_down(x.value, offset);
        REDUCE{}(x, y);
        offset >>= 1;
    }
    return x;
}

template <int NB, typename REDUCE, typename T>
__inline__ __device__ T rocblas_shuffle_block_reduce_method_64(T val)
{
    __shared__ T psums[warpSize];

    rocblas_int wavefront = threadIdx.x / warpSize;
    rocblas_int wavelet   = threadIdx.x % warpSize;

    if(wavefront == 0)
        psums[wavelet] = T{};
    __syncthreads();

    val = rocblas_wavefront_reduce_method_64<warpSize, REDUCE>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int num_wavefronts = NB / warpSize;
    val = (threadIdx.x < num_wavefronts) ? psums[wavelet] : T{};
    if(wavefront == 0)
        val = rocblas_wavefront_reduce_method_64<num_wavefronts, REDUCE>(val); // sum wavefront sums

    return val;
}

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <int NB, typename FETCH, typename REDUCE, typename TPtrX, typename To>
ROCBLAS_KERNEL(NB)
rocblas_iamax_iamin_kernel_part1_64(rocblas_int    n,
                                    TPtrX          xvec,
                                    rocblas_stride shiftx,
                                    int64_t        incx,
                                    rocblas_stride stridex,
                                    int64_t        batch_offset,
                                    int64_t        indexOffset,
                                    To*            workspace)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    To      sum;

    const auto* x = load_ptr_batch(xvec, blockIdx.y, shiftx, stridex);

    // bound
    if(tid < n)
        sum = FETCH{}(x[tid * incx], tid + 1 + indexOffset); // 1 based indexing
    else
        sum = rocblas_default_value<To>{}(); // pad with default value

    sum = rocblas_shuffle_block_reduce_method_64<NB, REDUCE>(sum);

    if(threadIdx.x == 0)
        workspace[blockIdx.y * batch_offset + blockIdx.x] = sum;
}

// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <int NB, typename REDUCE, typename To, typename Tr>
ROCBLAS_KERNEL(NB)
rocblas_iamax_iamin_kernel_part2_64(int64_t nblocks,
                                    int64_t batch_offset,
                                    To*     workspace,
                                    Tr*     result)
{
    int tx = threadIdx.x;
    To  winner;

    if(tx < nblocks)
    {
        To* work = workspace + blockIdx.y * batch_offset;
        winner   = work[tx];

        // bound, loop
        for(int64_t i = tx + NB; i < nblocks; i += NB)
            REDUCE{}(winner, work[i]);
    }
    else
    { // pad with default value
        winner = rocblas_default_value<To>{}();
    }

    winner = rocblas_shuffle_block_reduce_method_64<NB, REDUCE>(winner);

    // Store result on device or in workspace
    if(tx == 0)
        result[blockIdx.y] = winner.index;
}
