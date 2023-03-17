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

#include "reduction.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_iamax_iamin.hpp"
#include "rocblas_reduction.hpp"

// iamax, iamin kernels

template <int N, typename REDUCE, typename T>
__inline__ __device__ rocblas_index_value_t<T>
                      rocblas_wavefront_reduce_method(rocblas_index_value_t<T> x)
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

    rocblas_int wavefront = threadIdx.x / warpSize;
    rocblas_int wavelet   = threadIdx.x % warpSize;

    if(wavefront == 0)
        psums[wavelet] = T{};
    __syncthreads();

    val = rocblas_wavefront_reduce_method<warpSize, REDUCE>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int num_wavefronts = NB / warpSize;
    val = (threadIdx.x < num_wavefronts) ? psums[wavelet] : T{};
    if(wavefront == 0)
        val = rocblas_wavefront_reduce_method<num_wavefronts, REDUCE>(val); // sum wavefront sums

    return val;
}

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <rocblas_int NB, typename FETCH, typename REDUCE, typename TPtrX, typename To>
ROCBLAS_KERNEL(NB)
rocblas_iamax_iamin_kernel_part1(rocblas_int    n,
                                 rocblas_int    nblocks,
                                 TPtrX          xvec,
                                 rocblas_stride shiftx,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 To*            workspace)
{
    ptrdiff_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    To        sum;

    const auto* x = load_ptr_batch(xvec, blockIdx.y, shiftx, stridex);

    // bound
    if(tid < n)
        sum = FETCH{}(x[tid * incx], tid);
    else
        sum = rocblas_default_value<To>{}(); // pad with default value

    sum = rocblas_shuffle_block_reduce_method<NB, REDUCE>(sum);

    if(threadIdx.x == 0)
        workspace[blockIdx.y * nblocks + blockIdx.x] = sum;
}

// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <rocblas_int NB, typename REDUCE, typename FINALIZE, typename To, typename Tr>
ROCBLAS_KERNEL(NB)
rocblas_iamax_iamin_kernel_part2(rocblas_int nblocks, To* workspace, Tr* result)
{
    rocblas_int tx = threadIdx.x;
    To          sum;

    if(tx < nblocks)
    {
        To* work = workspace + blockIdx.y * nblocks;
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
        result[blockIdx.y] = Tr(FINALIZE{}(sum));
}

/*! \brief

    \details
    rocblas_internal_iamax_iamin_template computes a reduction over multiple vectors x_i
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
          typename REDUCE,
          typename FINALIZE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_internal_iamax_iamin_template(rocblas_handle handle,
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

    hipLaunchKernelGGL((rocblas_iamax_iamin_kernel_part1<NB, FETCH, REDUCE>),
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
        hipLaunchKernelGGL((rocblas_iamax_iamin_kernel_part2<NB, REDUCE, FINALIZE>),
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
            hipLaunchKernelGGL((rocblas_iamax_iamin_kernel_part2<NB, REDUCE, FINALIZE>),
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

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T*                  x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_template<ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amax,
                                                 rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_batched_template(rocblas_handle            handle,
                                            rocblas_int               n,
                                            const T* const*           x,
                                            rocblas_stride            shiftx,
                                            rocblas_int               incx,
                                            rocblas_stride            stridex,
                                            rocblas_int               batch_count,
                                            rocblas_int*              result,
                                            rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_template<ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amax,
                                                 rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T*                  x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_template<ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amin,
                                                 rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_batched_template(rocblas_handle            handle,
                                            rocblas_int               n,
                                            const T* const*           x,
                                            rocblas_stride            shiftx,
                                            rocblas_int               incx,
                                            rocblas_stride            stridex,
                                            rocblas_int               batch_count,
                                            rocblas_int*              result,
                                            rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_template<ROCBLAS_IAMAX_NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amin,
                                                 rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

// clang-format off
#ifdef INSTANTIATE_IAMAX_TEMPLATE
#error INSTANTIATE_IAMAX_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMAX_TEMPLATE(T_, S_)                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                         \
    rocblas_status rocblas_internal_iamax_template<T_, S_>(rocblas_handle    handle,  \
                                                          rocblas_int    n,           \
                                                          const T_*      x,           \
                                                          rocblas_stride shiftx,      \
                                                          rocblas_int    incx,        \
                                                          rocblas_stride stridex,     \
                                                          rocblas_int    batch_count, \
                                                          rocblas_int*   workspace,   \
                                                          rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMAX_TEMPLATE(float, float)
INSTANTIATE_IAMAX_TEMPLATE(double, double)
INSTANTIATE_IAMAX_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMAX_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMAX_TEMPLATE

#ifdef INSTANTIATE_IAMAX_BATCHED_TEMPLATE
#error INSTANTIATE_IAMAX_BATCHED_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMAX_BATCHED_TEMPLATE(T_, S_)                                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                \
    rocblas_status rocblas_internal_iamax_batched_template<T_, S_>(rocblas_handle   handle,  \
                                                               rocblas_int      n,           \
                                                               const T_* const* x,           \
                                                               rocblas_stride   shiftx,      \
                                                               rocblas_int      incx,        \
                                                               rocblas_stride   stridex,     \
                                                               rocblas_int      batch_count, \
                                                               rocblas_int*     workspace,   \
                                                               rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMAX_BATCHED_TEMPLATE(float, float)
INSTANTIATE_IAMAX_BATCHED_TEMPLATE(double, double)
INSTANTIATE_IAMAX_BATCHED_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMAX_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMAX_BATCHED_TEMPLATE

#ifdef INSTANTIATE_IAMIN_TEMPLATE
#error INSTANTIATE_IAMIN_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMIN_TEMPLATE(T_, S_)                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                         \
    rocblas_status rocblas_internal_iamin_template<T_, S_>(rocblas_handle    handle,  \
                                                          rocblas_int    n,           \
                                                          const T_*      x,           \
                                                          rocblas_stride shiftx,      \
                                                          rocblas_int    incx,        \
                                                          rocblas_stride stridex,     \
                                                          rocblas_int    batch_count, \
                                                          rocblas_int*   workspace,   \
                                                          rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMIN_TEMPLATE(float, float)
INSTANTIATE_IAMIN_TEMPLATE(double, double)
INSTANTIATE_IAMIN_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMIN_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMIN_TEMPLATE

#ifdef INSTANTIATE_IAMIN_BATCHED_TEMPLATE
#error INSTANTIATE_IAMIN_BATCHED_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_IAMIN_BATCHED_TEMPLATE(T_, S_)                                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                \
    rocblas_status rocblas_internal_iamin_batched_template<T_, S_>(rocblas_handle   handle,  \
                                                               rocblas_int      n,           \
                                                               const T_* const* x,           \
                                                               rocblas_stride   shiftx,      \
                                                               rocblas_int      incx,        \
                                                               rocblas_stride   stridex,     \
                                                               rocblas_int      batch_count, \
                                                               rocblas_int*     workspace,   \
                                                               rocblas_index_value_t<S_>* result);

INSTANTIATE_IAMIN_BATCHED_TEMPLATE(float, float)
INSTANTIATE_IAMIN_BATCHED_TEMPLATE(double, double)
INSTANTIATE_IAMIN_BATCHED_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_IAMIN_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_IAMIN_BATCHED_TEMPLATE

// clang-format on
