/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <bool CONJ, typename U, typename V>
__global__ void copy_kernel(rocblas_int    n,
                            const U        xa,
                            ptrdiff_t      shiftx,
                            rocblas_int    incx,
                            rocblas_stride stridex,
                            V              ya,
                            ptrdiff_t      shifty,
                            rocblas_int    incy,
                            rocblas_stride stridey)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto* x   = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    auto*       y   = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);
    if(tid < n)
    {

        y[tid * incy] = CONJ ? conj(x[tid * incx]) : x[tid * incx];
    }
}

//! @brief Optimized kernel for the floating points.
//!
template <rocblas_int NB, typename U, typename V>
__global__ __launch_bounds__(NB) void scopy_2_kernel(rocblas_int n,
                                                     const U __restrict xa,
                                                     ptrdiff_t      shiftx,
                                                     rocblas_stride stridex,
                                                     V __restrict ya,
                                                     ptrdiff_t      shifty,
                                                     rocblas_stride stridey)
{
    ptrdiff_t   tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 2;
    const auto* x   = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    auto*       y   = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);
    if(tid < n - 1)
    {
        for(rocblas_int j = 0; j < 2; ++j)
        {
            y[tid + j] = x[tid + j];
        }
    }
    if(n % 2 != 0 && tid == n - 1)
        y[tid] = x[tid];
}

template <bool CONJ, rocblas_int NB, typename U, typename V>
rocblas_status rocblas_copy_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     U              x,
                                     rocblas_int    offsetx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     V              y,
                                     rocblas_int    offsety,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    // Quick return if possible.
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    if(!x || !y)
        return rocblas_status_invalid_pointer;

    static constexpr bool using_rocblas_float
        = std::is_same<V, rocblas_float*>{} || std::is_same<V, rocblas_float* const*>{};

    if(!using_rocblas_float || incx != 1 || incy != 1)
    {
        // In case of negative inc shift pointer to end of data for negative indexing tid*inc
        ptrdiff_t shiftx = offsetx - ((incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0);
        ptrdiff_t shifty = offsety - ((incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0);

        int  blocks = (n - 1) / NB + 1;
        dim3 grid(blocks, batch_count);
        dim3 threads(NB);

        hipLaunchKernelGGL(copy_kernel<CONJ>,
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
                           stridey);
    }
    else
    {
        // Kernel function for improving the performance of SCOPY when incx==1 and incy==1

        // In case of negative inc shift pointer to end of data for negative indexing tid*inc
        ptrdiff_t shiftx = offsetx - 0;
        ptrdiff_t shifty = offsety - 0;

        int         blocks = 1 + ((n - 1) / (NB * 2));
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);
        hipStream_t scopy_stream = handle->get_stream();

        hipLaunchKernelGGL(scopy_2_kernel<NB>,
                           grid,
                           threads,
                           0,
                           scopy_stream,
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

template <typename T, typename U>
rocblas_status rocblas_copy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
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
