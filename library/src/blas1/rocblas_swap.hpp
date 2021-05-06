/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <typename T>
__forceinline__ __device__ __host__ void rocblas_swap_vals(T* __restrict__ x, T* __restrict__ y)
{
    T tmp = *y;
    *y    = *x;
    *x    = tmp;
}

template <typename UPtr>
ROCBLAS_KERNEL void rocblas_swap_kernel(rocblas_int    n,
                                        UPtr           xa,
                                        ptrdiff_t      offsetx,
                                        rocblas_int    incx,
                                        rocblas_stride stridex,
                                        UPtr           ya,
                                        ptrdiff_t      offsety,
                                        rocblas_int    incy,
                                        rocblas_stride stridey)
{
    auto*     x   = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto*     y   = load_ptr_batch(ya, hipBlockIdx_y, offsety, stridey);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        rocblas_swap_vals(x + tid * incx, y + tid * incy);
    }
}

//! @brief Optimized kernel for the floating points.
//!
template <rocblas_int NB, typename UPtr>
ROCBLAS_KERNEL __launch_bounds__(NB) void sswap_2_kernel(rocblas_int n,
                                                         UPtr __restrict__ xa,
                                                         ptrdiff_t      offsetx,
                                                         rocblas_stride stridex,
                                                         UPtr __restrict__ ya,
                                                         ptrdiff_t      offsety,
                                                         rocblas_stride stridey)
{
    ptrdiff_t tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 2;
    auto*     x   = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
    auto*     y   = load_ptr_batch(ya, hipBlockIdx_y, offsety, stridey);
    if(tid < n - 1)
    {
        for(rocblas_int j = 0; j < 2; ++j)
        {
            rocblas_swap_vals(x + tid + j, y + tid + j);
        }
    }
    if(n % 2 != 0 && tid == n - 1)
    {
        rocblas_swap_vals(x + tid, y + tid);
    }
}

template <rocblas_int NB, typename U>
rocblas_status rocblas_swap_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     U              y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count = 1)
{
    // Quick return if possible.
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    static constexpr bool using_rocblas_float
        = std::is_same<U, rocblas_float*>{} || std::is_same<U, rocblas_float* const*>{};

    if(!using_rocblas_float || incx != 1 || incy != 1)
    {
        // in case of negative inc shift pointer to end of data for negative indexing tid*inc
        ptrdiff_t shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
        ptrdiff_t shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);

        hipLaunchKernelGGL(rocblas_swap_kernel,
                           blocks,
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
                           stridey);
    }
    else
    {
        // Kernel function for improving the performance of SSWAP when incx==1 and incy==1
        ptrdiff_t shiftx = offsetx - 0;
        ptrdiff_t shifty = offsety - 0;

        int  blocks = 1 + ((n - 1) / (NB * 2));
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);

        hipLaunchKernelGGL(sswap_2_kernel<NB>,
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
                           stridey);
    }
    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_swap_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           T              y,
                                           rocblas_int    offset_y,
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
