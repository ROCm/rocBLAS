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
constexpr int NB = 256;

template <typename T>
__global__ void copy_kernel(rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    ssize_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(tid < n)
        y[tid * incy] = x[tid * incx];
}

template <typename>
constexpr char rocblas_copy_name[] = "unknown";
template <>
constexpr char rocblas_copy_name<float>[] = "rocblas_scopy";
template <>
constexpr char rocblas_copy_name<double>[] = "rocblas_dcopy";
template <>
constexpr char rocblas_copy_name<rocblas_half>[] = "rocblas_hcopy";
template <>
constexpr char rocblas_copy_name<rocblas_float_complex>[] = "rocblas_ccopy";
template <>
constexpr char rocblas_copy_name<rocblas_double_complex>[] = "rocblas_zcopy";

/*! \brief BLAS Level 1 API

    \details
    copy  copies the vector x[i] into the vector y[i], for  i = 1 , … , n

        y := x,

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
              if n <= 0 quick return with rocblas_status_success
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template <class T>
rocblas_status rocblas_copy_template(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_copy_name<T>, n, x, incx, y, incy);

    if(layer_mode & rocblas_layer_mode_log_bench)
        log_bench(handle,
                  "./rocblas-bench -f copy -r",
                  rocblas_precision_string<T>,
                  "-n",
                  n,
                  "--incx",
                  incx,
                  "--incy",
                  incy);

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_copy_name<T>, "N", n, "incx", incx, "incy", incy);

    if(!x || !y)
        return rocblas_status_invalid_pointer;

    /*
     * Quick return if possible.
     */
    if(n <= 0)
        return rocblas_status_success;

    int blocks = (n - 1) / NB + 1;
    dim3 grid(blocks);
    dim3 threads(NB);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(incx < 0)
        x -= ssize_t(incx) * (n - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (n - 1);

    hipLaunchKernelGGL(copy_kernel, grid, threads, 0, rocblas_stream, n, x, incx, y, incy);

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

rocblas_status rocblas_scopy(rocblas_handle handle,
                             rocblas_int n,
                             const float* x,
                             rocblas_int incx,
                             float* y,
                             rocblas_int incy)
{
    return rocblas_copy_template(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_dcopy(rocblas_handle handle,
                             rocblas_int n,
                             const double* x,
                             rocblas_int incx,
                             double* y,
                             rocblas_int incy)
{
    return rocblas_copy_template(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_hcopy(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_half* x,
                             rocblas_int incx,
                             rocblas_half* y,
                             rocblas_int incy)
{
    return rocblas_copy_template(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_ccopy(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_float_complex* x,
                             rocblas_int incx,
                             rocblas_float_complex* y,
                             rocblas_int incy)
{
    return rocblas_copy_template(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_zcopy(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_double_complex* x,
                             rocblas_int incx,
                             rocblas_double_complex* y,
                             rocblas_int incy)
{
    return rocblas_copy_template(handle, n, x, incx, y, incy);
}

} // extern "C"
