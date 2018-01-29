/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"
#include "handle.h"

#define NB_X 256

template <typename T>
__global__ void axpy_kernel_host_scalar(
    rocblas_int n, const T alpha, const T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(incx >= 0 && incy >= 0)
    {
        if(tid < n)
        {
            y[tid * incy] += (alpha)*x[tid * incx];
        }
    }
    else if(incx < 0 && incy < 0)
    {
        if(tid < n)
        {
            y[(1 - n + tid) * incy] += (alpha)*x[(1 - n + tid) * incx];
        }
    }
    else if(incx >= 0)
    {
        if(tid < n)
        {
            y[(1 - n + tid) * incy] += (alpha)*x[tid * incx];
        }
    }
    else
    {
        if(tid < n)
        {
            y[tid * incy] += (alpha)*x[(1 - n + tid) * incx];
        }
    }
}

template <typename T>
__global__ void axpy_kernel_device_scalar(
    rocblas_int n, const T* alpha, const T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(incx >= 0 && incy >= 0)
    {
        if(tid < n)
        {
            y[tid * incy] += (*alpha) * x[tid * incx];
        }
    }
    else if(incx < 0 && incy < 0)
    {
        if(tid < n)
        {
            y[(1 - n + tid) * incy] += (*alpha) * x[(1 - n + tid) * incx];
        }
    }
    else if(incx >= 0)
    {
        if(tid < n)
        {
            y[(1 - n + tid) * incy] += (*alpha) * x[tid * incx];
        }
    }
    else
    {
        if(tid < n)
        {
            y[tid * incy] += (*alpha) * x[(1 - n + tid) * incx];
        }
    }
}

__global__ void haxpy_mod_8_device_scalar(int n, const __fp16* alpha, const __fp16* x, __fp16* y)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    int index = ((n / 8) * 8) + tid;

    if(index < n)
        y[index] = (*alpha) * x[index] + y[index];
}

__global__ void haxpy_mod_8_host_scalar(int n, const __fp16 alpha, const __fp16* x, __fp16* y)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    int index = ((n / 8) * 8) + tid;

    if(index < n)
        y[index] = alpha * x[index] + y[index];
}

__global__ void
haxpy_mlt_8_device_scalar(int n_mlt_8, const __fp16* alpha, const half8* x, half8* y)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    half2 alpha_h2;
    alpha_h2[0] = (*alpha);
    alpha_h2[1] = (*alpha);

    half2 y0, y1, y2, y3;
    half2 x0, x1, x2, x3;
    half2 z0, z1, z2, z3;

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

__global__ void haxpy_mlt_8_host_scalar(int n_mlt_8, const half2 alpha, const half8* x, half8* y)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    half2 y0, y1, y2, y3;
    half2 x0, x1, x2, x3;
    half2 z0, z1, z2, z3;

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

        z0 = rocblas_fmadd_half2(alpha, x0, y0);
        z1 = rocblas_fmadd_half2(alpha, x1, y1);
        z2 = rocblas_fmadd_half2(alpha, x2, y2);
        z3 = rocblas_fmadd_half2(alpha, x3, y3);

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
rocblas_status rocblas_axpy_template(rocblas_handle handle,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* x,
                                     rocblas_int incx,
                                     T* y,
                                     rocblas_int incy)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_function(handle,
                     replaceX<T>("rocblas_Xaxpy"),
                     n,
                     *alpha,
                     (const void*&)x,
                     incx,
                     (const void*&)y,
                     incy);
        log_bench(handle,
                  "./rocblas-bench -f axpy -r",
                  replaceX<T>("X"),
                  "-n",
                  n,
                  "--alpha",
                  *alpha,
                  "--incx",
                  incx,
                  "--incy",
                  incx);
    }
    else
    {
        log_function(handle,
                     replaceX<T>("rocblas_Xaxpy"),
                     n,
                     (const void*&)alpha,
                     (const void*&)x,
                     incx,
                     (const void*&)y,
                     incy);
    }

    if(nullptr == alpha)
        return rocblas_status_invalid_pointer;
    else if(nullptr == x)
        return rocblas_status_invalid_pointer;
    else if(nullptr == y)
        return rocblas_status_invalid_pointer;

    if(n <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    int blocks = ((n - 1) / NB_X) + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL(axpy_kernel_device_scalar,
                           dim3(blocks),
                           dim3(threads),
                           0,
                           rocblas_stream,
                           n,
                           alpha,
                           x,
                           incx,
                           y,
                           incy);
    }
    else // alpha is on host
    {
        T scalar = *alpha;
        if(0.0 == scalar)
        {
            return rocblas_status_success;
        }

        hipLaunchKernelGGL(axpy_kernel_host_scalar,
                           dim3(blocks),
                           dim3(threads),
                           0,
                           rocblas_stream,
                           n,
                           scalar,
                           x,
                           incx,
                           y,
                           incy);
    }

    return rocblas_status_success;
}

template <class T>
rocblas_status rocblas_axpy_half(rocblas_handle handle,
                                 rocblas_int n,
                                 const T* alpha,
                                 const T* x,
                                 rocblas_int incx,
                                 T* y,
                                 rocblas_int incy)
{
    if(nullptr == alpha)
    {
        return rocblas_status_invalid_pointer;
    }
    else if(nullptr == x)
    {
        return rocblas_status_invalid_pointer;
    }
    else if(nullptr == y)
    {
        return rocblas_status_invalid_pointer;
    }
    else if(nullptr == handle)
    {
        return rocblas_status_invalid_handle;
    }

    if(n <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    if(1 != incx || 1 != incy) // slow code, no half8 or half2
    {
        int blocks = ((n - 1) / NB_X) + 1;

        dim3 grid(blocks, 1, 1);
        dim3 threads(NB_X, 1, 1);

        hipStream_t rocblas_stream;
        RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            hipLaunchKernelGGL(axpy_kernel_device_scalar,
                               dim3(blocks),
                               dim3(threads),
                               0,
                               rocblas_stream,
                               n,
                               (const __fp16*)alpha,
                               (const __fp16*)x,
                               incx,
                               (__fp16*)y,
                               incy);
        }
        else // alpha is on host
        {
            if(0 == *alpha)
            {
                return rocblas_status_success;
            }

            const __fp16 f16_alpha = *reinterpret_cast<const __fp16*>(alpha);
            hipLaunchKernelGGL(axpy_kernel_host_scalar,
                               dim3(blocks),
                               dim3(threads),
                               0,
                               rocblas_stream,
                               n,
                               f16_alpha,
                               (const __fp16*)x,
                               incx,
                               (__fp16*)y,
                               incy);
        }
    }
    else // half8 load-store and half2 arithmetic
    {
        rocblas_int n_mlt_8 = (n / 8) * 8; // multiple of 8
        rocblas_int n_mod_8 = n - n_mlt_8; // n mod 8

        int blocks = (((n / 8) - 1) / NB_X) + 1;

        dim3 grid(blocks, 1, 1);
        dim3 threads(NB_X, 1, 1);

        hipStream_t rocblas_stream;
        RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            hipLaunchKernelGGL(haxpy_mlt_8_device_scalar,
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocblas_stream,
                               n_mlt_8,
                               (const __fp16*)alpha,
                               (const half8*)x,
                               (half8*)y);

            if(0 != n_mod_8) // cleanup non-multiple of 8
            {
                hipLaunchKernelGGL(haxpy_mod_8_device_scalar,
                                   dim3(1, 1, 1),
                                   dim3(n_mod_8, 1, 1),
                                   0,
                                   rocblas_stream,
                                   n,
                                   (const __fp16*)alpha,
                                   (const __fp16*)x,
                                   (__fp16*)y);
            }
        }
        else // alpha is on host
        {
            if(0 == *alpha)
            {
                return rocblas_status_success;
            }

            half2 half2_alpha;
            half2_alpha[0] = *reinterpret_cast<const __fp16*>(alpha);
            half2_alpha[1] = *reinterpret_cast<const __fp16*>(alpha);

            hipLaunchKernelGGL(haxpy_mlt_8_host_scalar,
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocblas_stream,
                               n_mlt_8,
                               half2_alpha,
                               (const half8*)x,
                               (half8*)y);

            if(0 != n_mod_8) // cleanup non-multiple of 8
            {
                const __fp16 f16_alpha = *reinterpret_cast<const __fp16*>(alpha);
                hipLaunchKernelGGL(haxpy_mod_8_host_scalar,
                                   dim3(1, 1, 1),
                                   dim3(n_mod_8, 1, 1),
                                   0,
                                   rocblas_stream,
                                   n,
                                   f16_alpha,
                                   (const __fp16*)x,
                                   (__fp16*)y);
            }
        }
    }
    return rocblas_status_success;
}

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C89 wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_haxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_half* alpha,
                                        const rocblas_half* x,
                                        rocblas_int incx,
                                        rocblas_half* y,
                                        rocblas_int incy)
{
    return rocblas_axpy_half<rocblas_half>(handle, n, alpha, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_saxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const float* alpha,
                                        const float* x,
                                        rocblas_int incx,
                                        float* y,
                                        rocblas_int incy)
{
    return rocblas_axpy_template<float>(handle, n, alpha, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_daxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const double* alpha,
                                        const double* x,
                                        rocblas_int incx,
                                        double* y,
                                        rocblas_int incy)
{
    return rocblas_axpy_template<double>(handle, n, alpha, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_caxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_float_complex* alpha,
                                        const rocblas_float_complex* x,
                                        rocblas_int incx,
                                        rocblas_float_complex* y,
                                        rocblas_int incy)
{
    return rocblas_axpy_template<rocblas_float_complex>(handle, n, alpha, x, incx, y, incy);
}

extern "C" rocblas_status rocblas_zaxpy(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_double_complex* alpha,
                                        const rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    return rocblas_axpy_template<rocblas_double_complex>(handle, n, alpha, x, incx, y, incy);
}

/* ============================================================================================ */
