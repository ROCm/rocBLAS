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

template <typename>
constexpr char rocblas_axpy_name[] = "unknown";
template <>
constexpr char rocblas_axpy_name<float>[] = "rocblas_saxpy";
template <>
constexpr char rocblas_axpy_name<double>[] = "rocblas_daxpy";
template <>
constexpr char rocblas_axpy_name<rocblas_half>[] = "rocblas_haxpy";
template <>
constexpr char rocblas_axpy_name<rocblas_float_complex>[] = "rocblas_caxpy";
template <>
constexpr char rocblas_axpy_name<rocblas_double_complex>[] = "rocblas_zaxpy";

template <typename T>
void rocblas_axpy_log(rocblas_handle handle,
                      rocblas_int n,
                      const T* alpha,
                      const T* x,
                      rocblas_int incx,
                      T* y,
                      rocblas_int incy)
{
    auto layer_mode = handle->layer_mode;
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_axpy_name<T>, n, *alpha, x, incx, y, incy);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f axpy -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--alpha",
                      *alpha,
                      "--incx",
                      incx,
                      "--incy",
                      incx);
    }
    else if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_axpy_name<T>, n, alpha, x, incx, y, incy);

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_axpy_name<T>, "N", n, "incx", incx, "incy", incy);
}

template <typename T, typename U>
__global__ void axpy_kernel(
    rocblas_int n, U alpha_device_host, const T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    auto alpha  = load_scalar(alpha_device_host);
    ssize_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // bound
    if(tid < n)
        y[tid * incy] += alpha * x[tid * incx];
}

/*! \brief BLAS Level 1 API

    \details
    axpy   compute y := alpha * x + y

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
              if n <= 0 quick return with rocblas_status_success
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[inout]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template <class T>
rocblas_status rocblas_axpy(rocblas_handle handle,
                            rocblas_int n,
                            const T* alpha,
                            const T* x,
                            rocblas_int incx,
                            T* y,
                            rocblas_int incy)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!alpha)
        return rocblas_status_invalid_pointer;
    rocblas_axpy_log(handle, n, alpha, x, incx, y, incy);
    if(!x || !y)
        return rocblas_status_invalid_pointer;
    if(n <= 0) // Quick return if possible. Not Argument error
        return rocblas_status_success;

    int blocks = (n - 1) / NB + 1;
    dim3 threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(incx < 0)
        x -= ssize_t(incx) * (n - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (n - 1);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(
            axpy_kernel, blocks, threads, 0, rocblas_stream, n, alpha, x, incx, y, incy);
    else if(*alpha) // alpha is on host
        hipLaunchKernelGGL(
            axpy_kernel, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx, y, incy);

    return rocblas_status_success;
}

template <typename T, typename U>
__global__ void haxpy_mlt_8(int n_mlt_8, U alpha_device_host, const T* x, T* y)
{
    int tid       = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    auto alpha_h2 = load_scalar(alpha_device_host);

    rocblas_half2 y0, y1, y2, y3;
    rocblas_half2 x0, x1, x2, x3;
    rocblas_half2 z0, z1, z2, z3;

    if(tid * 8 < n_mlt_8)
    {
        y0[0] = y[tid][0];
        y0[1] = y[tid][1];
        y1[0] = y[tid][2];
        y1[1] = y[tid][3];
        y2[0] = y[tid][4];
        y2[1] = y[tid][5];
        y3[0] = y[tid][6];
        y3[1] = y[tid][7];

        x0[0] = x[tid][0];
        x0[1] = x[tid][1];
        x1[0] = x[tid][2];
        x1[1] = x[tid][3];
        x2[0] = x[tid][4];
        x2[1] = x[tid][5];
        x3[0] = x[tid][6];
        x3[1] = x[tid][7];

        z0 = rocblas_fmadd_half2(alpha_h2, x0, y0);
        z1 = rocblas_fmadd_half2(alpha_h2, x1, y1);
        z2 = rocblas_fmadd_half2(alpha_h2, x2, y2);
        z3 = rocblas_fmadd_half2(alpha_h2, x3, y3);

        y[tid][0] = z0[0];
        y[tid][1] = z0[1];
        y[tid][2] = z1[0];
        y[tid][3] = z1[1];
        y[tid][4] = z2[0];
        y[tid][5] = z2[1];
        y[tid][6] = z3[0];
        y[tid][7] = z3[1];
    }
}

template <typename T, typename U>
__global__ void haxpy_mod_8(int n_mod_8, U alpha_device_host, const T* x, T* y)
{
    auto alpha = load_scalar(alpha_device_host);
    int tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n_mod_8)
        y[tid] += alpha * x[tid];
}

template <>
rocblas_status rocblas_axpy(rocblas_handle handle,
                            rocblas_int n,
                            const rocblas_half* alpha,
                            const rocblas_half* x,
                            rocblas_int incx,
                            rocblas_half* y,
                            rocblas_int incy)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    rocblas_axpy_log(handle, n, alpha, x, incx, y, incy);
    if(!alpha || !x || !y)
        return rocblas_status_invalid_pointer;
    if(n <= 0) // Quick return if possible. Not Argument error
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;
    if(incx != 1 || incy != 1) // slow code, no rocblas_half8 or rocblas_half2
    {
        int blocks = (n - 1) / NB + 1;
        dim3 threads(NB);

        if(incx < 0)
            x += incx * (1 - n);
        if(incy < 0)
            y += incy * (1 - n);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
            hipLaunchKernelGGL(axpy_kernel,
                               blocks,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               (const _Float16*)alpha,
                               (const _Float16*)x,
                               incx,
                               (_Float16*)y,
                               incy);
        else if(*(const _Float16*)alpha) // alpha is on host
            hipLaunchKernelGGL(axpy_kernel,
                               blocks,
                               threads,
                               0,
                               rocblas_stream,
                               n,
                               *(const _Float16*)alpha,
                               (const _Float16*)x,
                               incx,
                               (_Float16*)y,
                               incy);
    }
    else
    {                                // rocblas_half8 load-store and rocblas_half2 arithmetic
        rocblas_int n_mod_8 = n & 7; // n mod 8
        rocblas_int n_mlt_8 = n & ~(rocblas_int)7; // multiple of 8
        int blocks          = (n / 8 - 1) / NB + 1;
        dim3 grid(blocks);
        dim3 threads(NB);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL(haxpy_mlt_8,
                               grid,
                               threads,
                               0,
                               rocblas_stream,
                               n_mlt_8,
                               (const rocblas_half2*)alpha,
                               (const rocblas_half8*)x,
                               (rocblas_half8*)y);

            if(n_mod_8) // cleanup non-multiple of 8
                hipLaunchKernelGGL(haxpy_mod_8,
                                   1,
                                   n_mod_8,
                                   0,
                                   rocblas_stream,
                                   n_mod_8,
                                   (const _Float16*)alpha,
                                   (const _Float16*)x + n_mlt_8,
                                   (_Float16*)y + n_mlt_8);
        }
        else if(*(const _Float16*)alpha) // alpha is on host
        {
            hipLaunchKernelGGL(haxpy_mlt_8,
                               grid,
                               threads,
                               0,
                               rocblas_stream,
                               n_mlt_8,
                               load_scalar((const rocblas_half2*)alpha),
                               (const rocblas_half8*)x,
                               (rocblas_half8*)y);

            if(n_mod_8) // cleanup non-multiple of 8
                hipLaunchKernelGGL(haxpy_mod_8,
                                   1,
                                   n_mod_8,
                                   0,
                                   rocblas_stream,
                                   n_mod_8,
                                   *(const _Float16*)alpha,
                                   (const _Float16*)x + n_mlt_8,
                                   (_Float16*)y + n_mlt_8);
        }
    }
    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {
rocblas_status rocblas_haxpy(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_half* alpha,
                             const rocblas_half* x,
                             rocblas_int incx,
                             rocblas_half* y,
                             rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

rocblas_status rocblas_saxpy(rocblas_handle handle,
                             rocblas_int n,
                             const float* alpha,
                             const float* x,
                             rocblas_int incx,
                             float* y,
                             rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

rocblas_status rocblas_daxpy(rocblas_handle handle,
                             rocblas_int n,
                             const double* alpha,
                             const double* x,
                             rocblas_int incx,
                             double* y,
                             rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

#if 0 // complex not supported
rocblas_status rocblas_caxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_float_complex* alpha,
                                        const rocblas_float_complex* x,
                                        rocblas_int incx,
                                        rocblas_float_complex* y,
                                        rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

rocblas_status rocblas_zaxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_double_complex* alpha,
                                        const rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}
#endif

} // extern "C"
