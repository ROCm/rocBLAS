/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/reduction_strided_batched.hpp"
#include "handle.hpp"
#include "rocblas.h"
#include "rocblas_amax_amin.hpp"
#include "utility.hpp"
#include <type_traits>
#include <utility>

// iamax, iamin kernels

template <int N, typename REDUCE, typename T>
__inline__ __device__ rocblas_index_value_t<T> wavefront_reduce_method(rocblas_index_value_t<T> x)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        rocblas_index_value_t<T> y{};
        y.index = __shfl_down(x.index, offset);
        y.value = __shfl_down(x.value, offset);
        REDUCE{}(x, y);
        offset >>= 1;
    }
    return x;
}

template <rocblas_int NB, typename REDUCE, typename T>
__inline__ __device__ T rocblas_shuffle_block_reduce_method(T val)
{
    __shared__ T psums[warpSize];

    rocblas_int wavefront = hipThreadIdx_x / warpSize;
    rocblas_int wavelet   = hipThreadIdx_x % warpSize;

    if(wavefront == 0)
        psums[wavelet] = T{};
    __syncthreads();

    val = wavefront_reduce_method<warpSize, REDUCE>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int num_wavefronts = NB / warpSize;
    val = (hipThreadIdx_x < num_wavefronts) ? psums[wavelet] : T{};
    if(wavefront == 0)
        val = wavefront_reduce_method<num_wavefronts, REDUCE>(val); // sum wavefront sums

    return val;
}

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE = rocblas_reduce_sum,
          typename TPtrX,
          typename To>
ROCBLAS_KERNEL(NB)
rocblas_iaminmax_reduction_strided_batched_kernel_part1(rocblas_int    n,
                                                        rocblas_int    nblocks,
                                                        TPtrX          xvec,
                                                        rocblas_stride shiftx,
                                                        rocblas_int    incx,
                                                        rocblas_stride stridex,
                                                        To*            workspace)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    To        sum;

    const auto* x = load_ptr_batch(xvec, hipBlockIdx_y, shiftx, stridex);

    // bound
    if(tid < n)
        sum = FETCH{}(x[tid * incx], tid);
    else
        sum = rocblas_default_value<To>{}(); // pad with default value

    sum = rocblas_shuffle_block_reduce_method<NB, REDUCE>(sum);

    if(hipThreadIdx_x == 0)
        workspace[hipBlockIdx_y * nblocks + hipBlockIdx_x] = sum;
}

// kernel 2 is used from non-strided reduction_batched see include file
// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <rocblas_int NB,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename To,
          typename Tr>
ROCBLAS_KERNEL(NB)
rocblas_iaminmax_reduction_strided_batched_kernel_part2(rocblas_int nblocks,
                                                        To*         workspace,
                                                        Tr*         result)
{
    rocblas_int tx = hipThreadIdx_x;
    To          sum;

    if(tx < nblocks)
    {
        To* work = workspace + hipBlockIdx_y * nblocks;
        sum      = work[tx];

        // bound, loop
        for(rocblas_int i = tx + NB; i < nblocks; i += NB)
            REDUCE{}(sum, work[i]);
    }
    else
    { // pad with default value
        sum = rocblas_default_value<To>{}();
    }

    sum = rocblas_shuffle_block_reduce_method<NB, REDUCE>(sum);

    // Store result on device or in workspace
    if(tx == 0)
        result[hipBlockIdx_y] = Tr(FINALIZE{}(sum));
}

/*! \brief

    \details
    rocblas_iaminmax_reduction_strided_batched computes a reduction over multiple vectors x_i
              Template parameters allow threads per block, data, and specific phase kernel overrides
              At least two kernels are needed to finish the reduction
              kernel 1 write partial result per thread block in workspace, blocks partial results
              kernel 2 gathers all the partial result in workspace and finishes the final reduction.
    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
              number of elements in each vector x_i
    @param[in]
    x         pointer to the first vector x_i on the GPU.
    @param[in]
    shiftx    rocblas_int
              specifies a base offset increment for the start of each x_i.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   rocblas_int
              specifies the pointer increment between batches for x. stridex must be >= n*incx
    @param[in]
    batch_count rocblas_int
              number of instances in the batch
    @param[out]
    workspace To*
              temporary GPU buffer for inidividual block results for each batch
              and results buffer in case result pointer is to host memory
              Size must be (blocks+1)*batch_count*sizeof(To)
    @param[out]
    result
              pointers to array of batch_count size for results. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_iaminmax_reduction_strided_batched(rocblas_handle __restrict__ handle,
                                                          rocblas_int    n,
                                                          TPtrX          x,
                                                          rocblas_stride shiftx,
                                                          rocblas_int    incx,
                                                          rocblas_stride stridex,
                                                          rocblas_int    batch_count,
                                                          To*            workspace,
                                                          Tr*            result)
{
    rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB);

    hipLaunchKernelGGL((rocblas_iaminmax_reduction_strided_batched_kernel_part1<NB, FETCH, REDUCE>),
                       dim3(blocks, batch_count),
                       NB,
                       0,
                       handle->get_stream(),
                       n,
                       blocks,
                       x,
                       shiftx,
                       incx,
                       stridex,
                       workspace);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL(
            (rocblas_iaminmax_reduction_strided_batched_kernel_part2<NB, REDUCE, FINALIZE>),
            dim3(1, batch_count),
            NB,
            0,
            handle->get_stream(),
            blocks,
            workspace,
            result);
    }
    else
    {
        // If in host pointer mode, workspace is converted to Tr* and the result is
        // placed there, and then copied from device to host. If To is a class type,
        // it must be a standard layout type and its first member must be of type Tr.
        static_assert(std::is_standard_layout<To>{}, "To must be a standard layout type");

        bool reduceKernel = blocks > 1 || batch_count > 1;
        if(reduceKernel)
        {
            hipLaunchKernelGGL(
                (rocblas_iaminmax_reduction_strided_batched_kernel_part2<NB, REDUCE, FINALIZE>),
                dim3(1, batch_count),
                NB,
                0,
                handle->get_stream(),
                blocks,
                workspace,
                (Tr*)(workspace + size_t(batch_count) * blocks));
        }

        if(std::is_same<FINALIZE, rocblas_finalize_identity>{} || reduceKernel)
        {
            // If FINALIZE is trivial or kernel part2 was called, result is in the
            // beginning of workspace[0]+offset, and can be copied directly.
            size_t offset = reduceKernel ? size_t(batch_count) * blocks : 0;
            RETURN_IF_HIP_ERROR(hipMemcpy(
                result, workspace + offset, batch_count * sizeof(Tr), hipMemcpyDeviceToHost));
        }
        else
        {
            // If FINALIZE is not trivial and kernel part2 was not called, then
            // workspace[0] needs to be finalized on host.
            auto res = std::make_unique<To[]>(batch_count);
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&res[0], workspace, batch_count * sizeof(To), hipMemcpyDeviceToHost));
            for(rocblas_int i = 0; i < batch_count; i++)
                result[i] = Tr(FINALIZE{}(res[i]));
        }
    }

    return rocblas_status_success;
}
