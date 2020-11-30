/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
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

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = offsetx - ((incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0);
    ptrdiff_t shifty = offsety - ((incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0);

    int         blocks = (n - 1) / NB + 1;
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);
    hipStream_t my_stream = handle->get_stream();

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    hipLaunchKernelGGL(copy_kernel<CONJ>,
                       grid,
                       threads,
                       0,
                       my_stream,
                       n,
                       x,
                       shiftx,
                       incx,
                       stridex,
                       y,
                       shifty,
                       incy,
                       stridey);

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
    rocblas_status check_numerics_status = rocblas_check_numerics_vector_template(function_name,
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

    check_numerics_status = rocblas_check_numerics_vector_template(function_name,
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
