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

#define NB_X 256

template <typename T>
__global__ void scal_kernel_host_scalar(rocblas_int n, const T alpha, T* x, rocblas_int incx)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(tid < n)
    {
        x[tid * incx] = (alpha) * (x[tid * incx]);
    }
}

template <typename T>
__global__ void scal_kernel_device_scalar(rocblas_int n, const T* alpha, T* x, rocblas_int incx)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(tid < n)
    {
        x[tid * incx] = (*alpha) * (x[tid * incx]);
    }
}

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
rocblas_scal_template(rocblas_handle handle, rocblas_int n, const T* alpha, T* x, rocblas_int incx)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_function(handle, replaceX<T>("rocblas_Xscal"), n, *alpha, (const void*&)x, incx);

        log_bench(handle,
                  "./rocblas-bench -f scal -r",
                  replaceX<T>("X"),
                  "-n",
                  n,
                  "--incx",
                  incx,
                  "--alpha",
                  *alpha);
    }
    else
    {
        log_function(
            handle, replaceX<T>("rocblas_Xscal"), n, (const void*&)alpha, (const void*&)x, incx);
    }

    if(nullptr == x)
        return rocblas_status_invalid_pointer;
    if(nullptr == alpha)
        return rocblas_status_invalid_pointer;

    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0)
        return rocblas_status_success;

    rocblas_int blocks = (n - 1) / NB_X + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL(scal_kernel_device_scalar,
                           dim3(blocks),
                           dim3(threads),
                           0,
                           rocblas_stream,
                           n,
                           alpha,
                           x,
                           incx);
    }
    else // alpha is on host
    {
        T scalar = *alpha;
        hipLaunchKernelGGL(scal_kernel_host_scalar,
                           dim3(blocks),
                           dim3(threads),
                           0,
                           rocblas_stream,
                           n,
                           scalar,
                           x,
                           incx);
    }

    return rocblas_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status
rocblas_sscal(rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx)
{
    return rocblas_scal_template<float>(handle, n, alpha, x, incx);
}

extern "C" rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
{
    return rocblas_scal_template<double>(handle, n, alpha, x, incx);
}

extern "C" rocblas_status rocblas_cscal(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_float_complex* alpha,
                                        rocblas_float_complex* x,
                                        rocblas_int incx)
{
    return rocblas_scal_template<rocblas_float_complex>(handle, n, alpha, x, incx);
}

extern "C" rocblas_status rocblas_zscal(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_double_complex* alpha,
                                        rocblas_double_complex* x,
                                        rocblas_int incx)
{
    return rocblas_scal_template<rocblas_double_complex>(handle, n, alpha, x, incx);
}
