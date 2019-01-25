/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {
static constexpr int NB = 256;

template <typename T>
__global__ void swap_kernel(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    ssize_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tmp      = y[tid * incy];
        y[tid * incy] = x[tid * incx];
        x[tid * incx] = tmp;
    }
}

template <typename>
constexpr char rocblas_swap_name[] = "unknown";
template <>
constexpr char rocblas_swap_name<float>[] = "rocblas_sswap";
template <>
constexpr char rocblas_swap_name<double>[] = "rocblas_dswap";
template <>
constexpr char rocblas_swap_name<rocblas_half>[] = "rocblas_hswap";
template <>
constexpr char rocblas_swap_name<rocblas_float_complex>[] = "rocblas_cswap";
template <>
constexpr char rocblas_swap_name<rocblas_double_complex>[] = "rocblas_zswap";

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
rocblas_status
rocblas_swap(rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    auto layer_mode = handle->layer_mode;

    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_swap_name<T>, n, x, incx, y, incy);
    if(layer_mode & rocblas_layer_mode_log_bench)
        log_bench(handle,
                  "./rocblas-bench -f swap -r",
                  rocblas_precision_string<T>,
                  "-n",
                  n,
                  "--incx",
                  incx,
                  "--incy",
                  incy);
    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_swap_name<T>, "N", n, "incx", incx, "incy", incy);

    if(!x || !y)
        return rocblas_status_invalid_pointer;
    /*
     * Quick return if possible.
     */
    if(n <= 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;
    int blocks                 = (n - 1) / NB + 1;
    dim3 grid(blocks);
    dim3 threads(NB);

    if(incx < 0)
        x -= ssize_t(incx) * (n - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (n - 1);

    hipLaunchKernelGGL(swap_kernel, grid, threads, 0, rocblas_stream, n, x, incx, y, incy);

    return rocblas_status_success;
}

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sswap(
    rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_dswap(
    rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

#if 0 // complex not supported

rocblas_status rocblas_cswap(rocblas_handle handle,
                             rocblas_int n,
                             rocblas_float_complex* x,
                             rocblas_int incx,
                             rocblas_float_complex* y,
                             rocblas_int incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_zswap(rocblas_handle handle,
                             rocblas_int n,
                             rocblas_double_complex* x,
                             rocblas_int incx,
                             rocblas_double_complex* y,
                             rocblas_int incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

#endif

} // extern "C"
