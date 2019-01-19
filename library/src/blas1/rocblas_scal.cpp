/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <complex.h>
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {
constexpr int NB = 256;

template <typename T, typename U>
__global__ void scal_kernel(rocblas_int n, U alpha_device_host, T* x, rocblas_int incx)
{
    auto alpha  = load_scalar(alpha_device_host);
    ssize_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        x[tid * incx] *= alpha;
}

template <typename>
constexpr char rocblas_scal_name[] = "unknown";
template <>
constexpr char rocblas_scal_name<float>[] = "rocblas_sscal";
template <>
constexpr char rocblas_scal_name<double>[] = "rocblas_dscal";
template <>
constexpr char rocblas_scal_name<rocblas_float_complex>[] = "rocblas_cscal";
template <>
constexpr char rocblas_scal_name<rocblas_double_complex>[] = "rocblas_zscal";

/*! \brief BLAS Level 1 API

    \details
    scal  scal the vector x[i] with scalar alpha, for  i = 1 , … , n

        x := alpha * x ,

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
              quick return if n <= 0.
    @param[in]
    alpha     specifies the scalar alpha.
    @param[inout]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
              quick return if incx <= 0.

    ********************************************************************/

template <class T>
rocblas_status
rocblas_scal(rocblas_handle handle, rocblas_int n, const T* alpha, T* x, rocblas_int incx)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!alpha)
        return rocblas_status_invalid_pointer;
    auto layer_mode = handle->layer_mode;
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_scal_name<T>, n, *alpha, x, incx);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f scal -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--alpha",
                      *alpha);
    }
    else
    {
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_scal_name<T>, n, alpha, x, incx);
    }
    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_scal_name<T>, "N", n, "incx", incx);

    if(!x)
        return rocblas_status_invalid_pointer;

    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0)
        return rocblas_status_success;

    rocblas_int blocks = (n - 1) / NB + 1;
    dim3 threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(scal_kernel, blocks, threads, 0, rocblas_stream, n, alpha, x, incx);
    else // alpha is on host
        hipLaunchKernelGGL(scal_kernel, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx);

    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status
rocblas_sscal(rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

#if 0 // complex not supported

rocblas_status rocblas_cscal(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_float_complex* alpha,
                             rocblas_float_complex* x,
                             rocblas_int incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

rocblas_status rocblas_zscal(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_double_complex* alpha,
                             rocblas_double_complex* x,
                             rocblas_int incx)
{
    return rocblas_scal(handle, n, alpha, x, incx);
}

#endif

} // extern "C"
