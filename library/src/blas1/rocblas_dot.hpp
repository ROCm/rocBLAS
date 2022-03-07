/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_reduction.hpp"

template <bool ONE_BLOCK, typename V, typename T>
__inline__ __device__ void
    rocblas_dot_save_sum(V sum, V* __restrict__ workspace, T* __restrict__ out)
{
    if(hipThreadIdx_x == 0)
    {
        if(ONE_BLOCK || hipGridDim_x == 1) // small N avoid second kernel
            out[hipBlockIdx_y] = T(sum);
        else
            workspace[hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x] = sum;
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
                        rocblas_int    stridey,
                        V* __restrict__ workspace,
                        T* __restrict__ out)
{
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    int i = !ONE_BLOCK ? hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x : hipThreadIdx_x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = !ONE_BLOCK ? hipBlockDim_x * hipGridDim_x : hipBlockDim_x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
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
          typename V,
          std::enable_if_t<!std::is_same<T, rocblas_half>{} && !std::is_same<T, rocblas_bfloat16>{}
                               && !std::is_same<T, rocblas_float>{},
                           int> = 0>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1by2(rocblas_int n,
                           const U __restrict__ xa,
                           rocblas_stride shiftx,
                           rocblas_stride stridex,
                           const U __restrict__ ya,
                           rocblas_stride shifty,
                           rocblas_int    stridey,
                           V* __restrict__ workspace,
                           T* __restrict__ out)
{
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    int i = !ONE_BLOCK ? hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x : hipThreadIdx_x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = !ONE_BLOCK ? hipBlockDim_x * hipGridDim_x : hipBlockDim_x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
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
          typename V,
          std::enable_if_t<std::is_same<T, rocblas_half>{} || std::is_same<T, rocblas_bfloat16>{}
                               || std::is_same<T, rocblas_float>{},
                           int> = 0>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1by2(rocblas_int n,
                           const U __restrict__ xa,
                           rocblas_stride shiftx,
                           rocblas_stride stridex,
                           const U __restrict__ ya,
                           rocblas_stride shifty,
                           rocblas_int    stridey,
                           V* __restrict__ workspace,
                           T* __restrict__ out)
{
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    int i = !ONE_BLOCK ? hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x : hipThreadIdx_x;
    i *= 2;

    V sum = 0;

    // sum WIN elements per thread
    int inc = !ONE_BLOCK ? hipBlockDim_x * hipGridDim_x : hipBlockDim_x;
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
                   rocblas_int    stridey,
                   V* __restrict__ workspace,
                   T* __restrict__ out)
{
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    const T* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    int i = !ONE_BLOCK ? hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x : hipThreadIdx_x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = hipBlockDim_x * hipGridDim_x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
    {
        sum += V(y[i * incy]) * V(CONJ ? conj(x[i * incx]) : x[i * incx]);
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
    const T* x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);

    int i = !ONE_BLOCK ? hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x : hipThreadIdx_x;

    V sum = 0;

    // sum WIN elements per thread
    int inc = hipBlockDim_x * hipGridDim_x;
    for(int j = 0; j < WIN && i < n; j++, i += inc)
    {
        sum += V(x[i * incx]) * V(CONJ ? conj(x[i * incx]) : x[i * incx]);
    }
    sum = rocblas_dot_block_reduce<NB>(sum);

    rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <rocblas_int NB, rocblas_int WIN, typename V, typename T = V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_reduce(rocblas_int n_sums, V* __restrict__ in, T* __restrict__ out)
{
    V sum = 0;

    int offset = hipBlockIdx_y * n_sums;
    in += offset;

    int inc = hipBlockDim_x * hipGridDim_x * WIN;

    int i         = hipThreadIdx_x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += in[i + j];
    }
    if(hipThreadIdx_x < remainder)
    {
        sum += in[n_sums - 1 - hipThreadIdx_x];
    }

    sum = rocblas_dot_block_reduce<NB>(sum);
    if(hipThreadIdx_x == 0)
        out[hipBlockIdx_y] = T(sum);
}

// work item number (WIN) of elements to process
template <typename T>
constexpr int rocblas_dot_WIN()
{
    size_t nb = sizeof(T);

    int n = 8;
    if(nb >= 8)
        n = 2;
    else if(nb >= 4)
        n = 4;

    return n;
}

constexpr int rocblas_dot_WIN(size_t nb)
{
    int n = 8;
    if(nb >= 8)
        n = 2;
    else if(nb >= 4)
        n = 4;

    return n;
}

// assume workspace has already been allocated, recommended for repeated calling of dot_strided_batched product
// routine
template <rocblas_int NB, bool CONJ, typename T, typename U, typename V = T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_template(rocblas_handle __restrict__ handle,
                                  rocblas_int n,
                                  const U __restrict__ x,
                                  rocblas_int    offsetx,
                                  rocblas_int    incx,
                                  rocblas_stride stridex,
                                  const U __restrict__ y,
                                  rocblas_int    offsety,
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
            RETURN_IF_HIP_ERROR(hipMemset(results, 0, batch_count * sizeof(T)));
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
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    int single_block_threshold = 32768;
    if(std::is_same<T, float>{})
        single_block_threshold = 31000;
    else if(std::is_same<T, rocblas_float_complex>{})
        single_block_threshold = 16000;
    else if(std::is_same<T, double>{})
        single_block_threshold = 13000;
    else if(std::is_same<T, rocblas_double_complex>{})
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
                hipLaunchKernelGGL((rocblas_dot_kernel_inc1by2<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
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
                hipLaunchKernelGGL((rocblas_dot_kernel<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
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
            hipLaunchKernelGGL((rocblas_dot_kernel_magsq<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
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
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
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
                hipLaunchKernelGGL((rocblas_dot_kernel_inc1<ONE_BLOCK, NB, WIN, CONJ, T>),
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
                hipLaunchKernelGGL((rocblas_dot_kernel<ONE_BLOCK, NB, WIN, CONJ, T>),
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
            hipLaunchKernelGGL((rocblas_dot_kernel_magsq<ONE_BLOCK, NB, WIN, CONJ, T>),
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
                hipLaunchKernelGGL((rocblas_dot_kernel_reduce<NB, WIN>),
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
                hipLaunchKernelGGL((rocblas_dot_kernel_reduce<NB, WIN>),
                                   dim3(1, batch_count),
                                   threads,
                                   0,
                                   handle->get_stream(),
                                   blocks,
                                   workspace,
                                   output);

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
        }
    }
    return rocblas_status_success;
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
