/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "definitions.h"
#include "handle.h"

#define NB_X 256

template <typename T>
__global__ void swap_kernel(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    T tmp;
    if(incx >= 0 && incy >= 0)
    {
        if(tid < n)
        {
            tmp           = y[tid * incy];
            y[tid * incy] = x[tid * incx];
            x[tid * incx] = tmp;
        }
    }
    else if(incx < 0 && incy < 0)
    {
        if(tid < n)
        {
            tmp                     = y[(1 - n + tid) * incy];
            y[(1 - n + tid) * incy] = x[(1 - n + tid) * incx];
            x[(1 - n + tid) * incx] = tmp;
        }
    }
    else if(incx >= 0)
    {
        if(tid < n)
        {
            tmp                     = y[(1 - n + tid) * incy];
            y[(1 - n + tid) * incy] = x[tid * incx];
            x[tid * incx]           = tmp;
        }
    }
    else
    {
        if(tid < n)
        {
            tmp                     = y[tid * incy];
            y[tid * incy]           = x[(1 - n + tid) * incx];
            x[(1 - n + tid) * incx] = tmp;
        }
    }
}

/*! \brief BLAS Level 1 API

    \details
    swap  interchange vector x[i] and y[i], for  i = 1 , … , n

        y := x; x := y

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
              if n <= 0 quick return with rocblas_status_success
    @param[inout]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[inout]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template <class T>
rocblas_status rocblas_swap_template(
    rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    log_function(
        handle, replaceX<T>("rocblas_Xswap"), n, (const void*&)x, incx, (const void*&)y, incy);

    if(x == nullptr)
        return rocblas_status_invalid_pointer;
    else if(y == nullptr)
        return rocblas_status_invalid_pointer;
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    /*
     * Quick return if possible.
     */
    if(n <= 0)
        return rocblas_status_success;

    int blocks = (n - 1) / NB_X + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    hipLaunchKernelGGL(
        swap_kernel, dim3(grid), dim3(threads), 0, rocblas_stream, n, x, incx, y, incy);

    return rocblas_status_success;
}

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_sswap(
    rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{

    return rocblas_swap_template<float>(handle, n, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_dswap(
    rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{

    return rocblas_swap_template<double>(handle, n, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_cswap(rocblas_handle handle,
                                        rocblas_int n,
                                        rocblas_float_complex* x,
                                        rocblas_int incx,
                                        rocblas_float_complex* y,
                                        rocblas_int incy)
{

    return rocblas_swap_template<rocblas_float_complex>(handle, n, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_zswap(rocblas_handle handle,
                                        rocblas_int n,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{

    return rocblas_swap_template<rocblas_double_complex>(handle, n, x, incx, y, incy);
}

/* ============================================================================================ */
