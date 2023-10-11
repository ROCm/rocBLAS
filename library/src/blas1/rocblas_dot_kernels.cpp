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

#include "check_numerics_vector.hpp"
#include "logging.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_dot.hpp"

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

template <bool        ONE_BLOCK,
          rocblas_int NB,
          rocblas_int WIN,
          bool        CONJ,
          typename T,
          typename U,
          typename V>
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
    const T* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

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

template <
    bool        ONE_BLOCK,
    rocblas_int NB,
    rocblas_int WIN,
    bool        CONJ,
    typename T,
    typename U,
    typename V,
    std::enable_if_t<
        !std::is_same_v<
            T,
            rocblas_half> && !std::is_same_v<T, rocblas_bfloat16> && !std::is_same_v<T, rocblas_float>,
        int> = 0>
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
    const T* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

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

template <
    bool        ONE_BLOCK,
    rocblas_int NB,
    rocblas_int WIN,
    bool        CONJ,
    typename T,
    typename U,
    typename V,
    std::enable_if_t<
        std::is_same_v<
            T,
            rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, rocblas_float>,
        int> = 0>
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
    const T* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;
    i *= 2;

    V sum = 0;

    // sum WIN elements per thread
    int inc = !ONE_BLOCK ? blockDim.x * gridDim.x : blockDim.x;
    inc *= 2;
    for(int j = 0; j < WIN && i < n - 1; j++, i += inc)
    {
#pragma unroll
        for(rocblas_int k = 0; k < 2; ++k)
        {
            sum += V(y[i + k]) * V(CONJ ? conj(x[i + k]) : x[i + k]);
        }
    }
    // If `n` is odd then the computation of last element is covered below.
    if(n % 2 && i == n - 1)
    {
        sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
    }

    sum = rocblas_dot_block_reduce<NB>(sum);

    rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <bool        ONE_BLOCK,
          rocblas_int NB,
          rocblas_int WIN,
          bool        CONJ,
          typename T,
          typename U,
          typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel(rocblas_int n,
                   const U __restrict__ xa,
                   rocblas_stride shiftx,
                   rocblas_int    incx,
                   rocblas_stride stridex,
                   const U __restrict__ ya,
                   rocblas_stride shifty,
                   rocblas_int    incy,
                   rocblas_stride stridey,
                   V* __restrict__ workspace,
                   T* __restrict__ out)
{
    const T* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

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

template <bool        ONE_BLOCK,
          rocblas_int NB,
          rocblas_int WIN,
          bool        CONJ,
          typename T,
          typename U,
          typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_magsq(rocblas_int n,
                         const U __restrict__ xa,
                         rocblas_stride shiftx,
                         rocblas_int    incx,
                         rocblas_stride stridex,
                         V* __restrict__ workspace,
                         T* __restrict__ out)
{
    const T* x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);

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

template <rocblas_int NB, rocblas_int WIN, typename V, typename T = V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_reduce(rocblas_int n_sums, V* __restrict__ in, T* __restrict__ out)
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
template <rocblas_int NB, bool CONJ, typename T, typename U, typename V>
rocblas_status rocblas_internal_dot_template(rocblas_handle __restrict__ handle,
                                             rocblas_int n,
                                             const U __restrict__ x,
                                             rocblas_stride offsetx,
                                             rocblas_int    incx,
                                             rocblas_stride stridex,
                                             const U __restrict__ y,
                                             rocblas_stride offsety,
                                             rocblas_int    incy,
                                             rocblas_stride stridey,
                                             rocblas_int    batch_count,
                                             T* __restrict__ results,
                                             V* __restrict__ workspace)
{

    // One or two kernels are used to finish the reduction
    // kernel 1 write partial results per thread block in workspace, number of partial results is blocks
    // kernel 2 if blocks > 1 the partial results in workspace are reduced to output

    static constexpr int WIN = rocblas_dot_WIN<T>();

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

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    int64_t shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
    int64_t shifty = incy < 0 ? offsety - int64_t(incy) * (n - 1) : offsety;

    int single_block_threshold = 32768;
    if(std::is_same_v<T, float>)
        single_block_threshold = 31000;
    else if(std::is_same_v<T, rocblas_float_complex>)
        single_block_threshold = 16000;
    else if(std::is_same_v<T, double>)
        single_block_threshold = 13000;
    else if(std::is_same_v<T, rocblas_double_complex>)
        single_block_threshold = 10000;

    if(n <= single_block_threshold)
    {
        // we only reduce the block count to 1 so safe to ignore extra workspace allocated in caller

        static constexpr int NB_OB  = 1024;
        static constexpr int WIN_OB = 32; // 32K max n threshold, assert guard below

        static constexpr bool ONE_BLOCK = true;

        rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB_OB * WIN_OB);
        assert(blocks == 1);

        dim3   grid(blocks, batch_count);
        dim3   threads(NB_OB);
        size_t offset = size_t(batch_count) * blocks;
        T*     output = results;
        if(handle->pointer_mode != rocblas_pointer_mode_device)
        {
            output = (T*)(workspace + offset);
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
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
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
            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_magsq<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
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

        if(handle->pointer_mode != rocblas_pointer_mode_device)
        {
            // Changed to hipMemcpy for pointer mode host to match legacy BLAS.
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&results[0], output, sizeof(T) * batch_count, hipMemcpyDeviceToHost));
        }
    }
    else
    {
        static constexpr bool ONE_BLOCK = false;
        rocblas_int           blocks    = rocblas_reduction_kernel_block_count(n, NB * WIN);
        dim3                  grid(blocks, batch_count);
        dim3                  threads(NB);
        size_t                offset = size_t(batch_count) * blocks;
        T*                    output = results;
        if(handle->pointer_mode != rocblas_pointer_mode_device)
        {
            output = (T*)(workspace + offset);
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
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel<ONE_BLOCK, NB, WIN, CONJ, T>),
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
            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_magsq<ONE_BLOCK, NB, WIN, CONJ, T>),
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

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            if(blocks > 1) // if single block first kernel did all work
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_reduce<NB, WIN>),
                                      dim3(1, batch_count),
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      blocks,
                                      workspace,
                                      results);
        }
        else
        {
            if(blocks > 1) // if single block first kernel did all work
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_reduce<NB, WIN>),
                                      dim3(1, batch_count),
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      blocks,
                                      workspace,
                                      output);
            // Changed to hipMemcpy for pointer mode host to match legacy BLAS.
            RETURN_IF_HIP_ERROR(
                hipMemcpy(&results[0], output, sizeof(T) * batch_count, hipMemcpyDeviceToHost));
        }
    }
    return rocblas_status_success;
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_template(rocblas_handle __restrict__ handle,
                                  rocblas_int n,
                                  const T* __restrict__ x,
                                  rocblas_stride offsetx,
                                  rocblas_int    incx,
                                  rocblas_stride stridex,
                                  const T* __restrict__ y,
                                  rocblas_stride offsety,
                                  rocblas_int    incy,
                                  rocblas_stride stridey,
                                  rocblas_int    batch_count,
                                  T* __restrict__ results,
                                  Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_template<ROCBLAS_DOT_NB, false>(handle,
                                                                n,
                                                                x,
                                                                offsetx,
                                                                incx,
                                                                stridex,
                                                                y,
                                                                offsety,
                                                                incy,
                                                                stridey,
                                                                batch_count,
                                                                results,
                                                                workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_template(rocblas_handle __restrict__ handle,
                                   rocblas_int n,
                                   const T* __restrict__ x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T* __restrict__ y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T* __restrict__ results,
                                   Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_template<ROCBLAS_DOT_NB, true>(handle,
                                                               n,
                                                               x,
                                                               offsetx,
                                                               incx,
                                                               stridex,
                                                               y,
                                                               offsety,
                                                               incy,
                                                               stridey,
                                                               batch_count,
                                                               results,
                                                               workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_batched_template(rocblas_handle __restrict__ handle,
                                          rocblas_int n,
                                          const T* const* __restrict__ x,
                                          rocblas_stride offsetx,
                                          rocblas_int    incx,
                                          rocblas_stride stridex,
                                          const T* const* __restrict__ y,
                                          rocblas_stride offsety,
                                          rocblas_int    incy,
                                          rocblas_stride stridey,
                                          rocblas_int    batch_count,
                                          T* __restrict__ results,
                                          Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_template<ROCBLAS_DOT_NB, false>(handle,
                                                                n,
                                                                x,
                                                                offsetx,
                                                                incx,
                                                                stridex,
                                                                y,
                                                                offsety,
                                                                incy,
                                                                stridey,
                                                                batch_count,
                                                                results,
                                                                workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_batched_template(rocblas_handle __restrict__ handle,
                                           rocblas_int n,
                                           const T* const* __restrict__ x,
                                           rocblas_stride offsetx,
                                           rocblas_int    incx,
                                           rocblas_stride stridex,
                                           const T* const* __restrict__ y,
                                           rocblas_stride offsety,
                                           rocblas_int    incy,
                                           rocblas_stride stridey,
                                           rocblas_int    batch_count,
                                           T* __restrict__ results,
                                           Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_template<ROCBLAS_DOT_NB, true>(handle,
                                                               n,
                                                               x,
                                                               offsetx,
                                                               incx,
                                                               stridex,
                                                               y,
                                                               offsety,
                                                               incy,
                                                               stridey,
                                                               batch_count,
                                                               results,
                                                               workspace);
}

template <typename T>
rocblas_status rocblas_dot_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          rocblas_int    n,
                                          T              x,
                                          rocblas_stride offset_x,
                                          rocblas_int    inc_x,
                                          rocblas_stride stride_x,
                                          T              y,
                                          rocblas_stride offset_y,
                                          rocblas_int    inc_y,
                                          rocblas_stride stride_y,
                                          rocblas_int    batch_count,
                                          const int      check_numerics,
                                          bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// If there are any changes in template parameters in the files *dot*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off
#ifdef INSTANTIATE_DOT_TEMPLATE
#error INSTANTIATE_DOT_TEMPLATE already defined
#endif

#define INSTANTIATE_DOT_TEMPLATE(T_, Tex_)                                                      \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                       \
rocblas_status rocblas_internal_dot_template<T_, Tex_>(rocblas_handle __restrict__ handle,      \
                                                       rocblas_int                 n,           \
                                                       const T_*      __restrict__ x,           \
                                                       rocblas_stride              offsetx,     \
                                                       rocblas_int                 incx,        \
                                                       rocblas_stride              stridex,     \
                                                       const T_*      __restrict__ y,           \
                                                       rocblas_stride              offsety,     \
                                                       rocblas_int                 incy,        \
                                                       rocblas_stride              stridey,     \
                                                       rocblas_int                 batch_count, \
                                                       T_*            __restrict__ results,     \
                                                       Tex_*          __restrict__ workspace);

INSTANTIATE_DOT_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_DOT_TEMPLATE(rocblas_bfloat16, float)
INSTANTIATE_DOT_TEMPLATE(float, float)
INSTANTIATE_DOT_TEMPLATE(double, double)
INSTANTIATE_DOT_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOT_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOT_TEMPLATE

#ifdef INSTANTIATE_DOTC_TEMPLATE
#error INSTANTIATE_DOTC_TEMPLATE already defined
#endif

#define INSTANTIATE_DOTC_TEMPLATE(T_, Tex_)                                                 \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                   \
rocblas_status rocblas_internal_dotc_template<T_, Tex_>(rocblas_handle __restrict__ handle, \
                                                  rocblas_int                 n,            \
                                                  const T_*      __restrict__ x,            \
                                                  rocblas_stride              offsetx,      \
                                                  rocblas_int                 incx,         \
                                                  rocblas_stride              stridex,      \
                                                  const T_*      __restrict__ y,            \
                                                  rocblas_stride              offsety,      \
                                                  rocblas_int                 incy,         \
                                                  rocblas_stride              stridey,      \
                                                  rocblas_int                 batch_count,  \
                                                  T_*            __restrict__ results,      \
                                                  Tex_*          __restrict__ workspace);

INSTANTIATE_DOTC_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOTC_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOTC_TEMPLATE

#ifdef INSTANTIATE_DOT_BATCHED_TEMPLATE
#error INSTANTIATE_DOT_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_DOT_BATCHED_TEMPLATE(T_, Tex_)                                                        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                                 \
rocblas_status rocblas_internal_dot_batched_template<T_, Tex_>(rocblas_handle   __restrict__ handle,      \
                                                               rocblas_int                   n,           \
                                                               const T_* const* __restrict__ x,           \
                                                               rocblas_stride                offsetx,     \
                                                               rocblas_int                   incx,        \
                                                               rocblas_stride                stridex,     \
                                                               const T_* const* __restrict__ y,           \
                                                               rocblas_stride                offsety,     \
                                                               rocblas_int                   incy,        \
                                                               rocblas_stride                stridey,     \
                                                               rocblas_int                   batch_count, \
                                                               T_*              __restrict__ results,     \
                                                               Tex_*            __restrict__ workspace);

INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_bfloat16, float)
INSTANTIATE_DOT_BATCHED_TEMPLATE(float, float)
INSTANTIATE_DOT_BATCHED_TEMPLATE(double, double)
INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOT_BATCHED_TEMPLATE

#ifdef INSTANTIATE_DOTC_BATCHED_TEMPLATE
#error INSTANTIATE_DOTC_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_DOTC_BATCHED_TEMPLATE(T_, Tex_)                                                        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                                  \
rocblas_status rocblas_internal_dotc_batched_template<T_, Tex_>(rocblas_handle   __restrict__ handle,      \
                                                                rocblas_int                   n,           \
                                                                const T_* const* __restrict__ x,           \
                                                                rocblas_stride                offsetx,     \
                                                                rocblas_int                   incx,        \
                                                                rocblas_stride                stridex,     \
                                                                const T_* const* __restrict__ y,           \
                                                                rocblas_stride                offsety,     \
                                                                rocblas_int                   incy,        \
                                                                rocblas_stride                stridey,     \
                                                                rocblas_int                   batch_count, \
                                                                T_*              __restrict__ results,     \
                                                                Tex_*            __restrict__ workspace);

INSTANTIATE_DOTC_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOTC_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOTC_BATCHED_TEMPLATE

// for ex interface
#ifdef INSTANTIATE_DOT_EX_TEMPLATE
#error INSTANTIATE_DOT_EX_TEMPLATE already defined
#endif

#define INSTANTIATE_DOT_EX_TEMPLATE(NB_, CONJ_, T_, U_, V_) \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE \
rocblas_status rocblas_internal_dot_template<NB_, CONJ_, T_, U_, V_>(rocblas_handle __restrict__ handle,      \
                                                                     rocblas_int                 n,           \
                                                                     U_             __restrict__ x,           \
                                                                     rocblas_stride              offsetx,     \
                                                                     rocblas_int                 incx,        \
                                                                     rocblas_stride              stridex,     \
                                                                     U_             __restrict__ y,           \
                                                                     rocblas_stride              offsety,     \
                                                                     rocblas_int                 incy,        \
                                                                     rocblas_stride              stridey,     \
                                                                     rocblas_int                 batch_count, \
                                                                     T_*            __restrict__ results,     \
                                                                     V_*            __restrict__ workspace);

// Mixed precision for dot_ex
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, false, rocblas_half, rocblas_half const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, false, rocblas_half, rocblas_half const* const*, float)

// real types are "supported" in dotc_ex
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const* const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const*, rocblas_half)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const* const*, rocblas_half)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, rocblas_bfloat16, rocblas_bfloat16 const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, rocblas_bfloat16, rocblas_bfloat16 const* const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, float, float const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, float, float const* const*, float)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, double, double const*, double)
INSTANTIATE_DOT_EX_TEMPLATE(ROCBLAS_DOT_NB, true, double, double const* const*, double)

#undef INSTANTIATE_DOT_EX_TEMPLATE

#ifdef INSTANTIATE_DOT_CHECK_NUMERICS
#error INSTANTIATE_DOT_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_DOT_CHECK_NUMERICS(T_) \
template rocblas_status rocblas_dot_check_numerics<T_>(const char* function_name, \
    rocblas_handle handle, \
    rocblas_int    n, \
    T_             x, \
    rocblas_stride offset_x, \
    rocblas_int    inc_x, \
    rocblas_stride stride_x, \
    T_             y, \
    rocblas_stride offset_y, \
    rocblas_int    inc_y, \
    rocblas_stride stride_y, \
    rocblas_int    batch_count, \
    const int      check_numerics, \
    bool           is_input);

INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_half const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_half const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_bfloat16 const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_bfloat16 const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(float const*)
INSTANTIATE_DOT_CHECK_NUMERICS(float const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(double const*)
INSTANTIATE_DOT_CHECK_NUMERICS(double const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_float_complex const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_float_complex const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_double_complex const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_double_complex const* const*)

#undef INSTANTIATE_DOT_CHECK_NUMERICS

// clang-format on
