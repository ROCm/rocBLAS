/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include<complex.h>
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "definitions.h"
#include "handle.h"

#define NB_X 256

template <typename T>
__global__ void
scal_kernel_host_scalar(hipLaunchParm lp, rocblas_int n, const T alpha, T* x, rocblas_int incx)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // bound
    if(tid < n)
    {
        x[tid * incx] = (alpha) * (x[tid * incx]);
    }
}

template <typename T>
__global__ void
scal_kernel_device_scalar(hipLaunchParm lp, rocblas_int n, const T* alpha, T* x, rocblas_int incx)
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
    if(nullptr == x)
        return rocblas_status_invalid_pointer;
    if(nullptr == alpha)
        return rocblas_status_invalid_pointer;
    else if(nullptr == handle)
        return rocblas_status_invalid_handle;

    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0)
        return rocblas_status_success;

    rocblas_int blocks = (n - 1) / NB_X + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernel(HIP_KERNEL_NAME(scal_kernel_device_scalar),
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
        hipLaunchKernel(HIP_KERNEL_NAME(scal_kernel_host_scalar),
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
    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            fprintf(handle->rocblas_logfile, "rocblas_sscal,%d,%f,%p,%d", n, *alpha, (void*)x, incx);
        }
        else
        {
            if(handle->layer_mode & rocblas_layer_mode_logging_synch)
            {
                float temp;
                hipMemcpy(&temp, alpha, sizeof(float), hipMemcpyDeviceToHost);
                fprintf(handle->rocblas_logfile, "rocblas_sscal,%d,%f,%p,%d", n, temp, (void*)x, incx);
            }
            else
            {
                fprintf(handle->rocblas_logfile, "rocblas_sscal,%d,%p,%p,%d", n, alpha, (void*)x, incx);
            }
        }
    }

    rocblas_status status = rocblas_scal_template<float>(handle, n, alpha, x, incx);

    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->layer_mode & rocblas_layer_mode_logging_synch)
        {
            fprintf(handle->rocblas_logfile, ",%d\n", status);
            fflush(handle->rocblas_logfile);
        }
    }

    return status;
}

extern "C" rocblas_status 
rocblas_dscal(rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
{
    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            fprintf(handle->rocblas_logfile, "rocblas_dscal,%d,%lf,%p,%d", n, *alpha, (void*)x, incx);
        }
        else
        {
            if(handle->layer_mode & rocblas_layer_mode_logging_synch)
            {
                float temp;
                hipMemcpy(&temp, alpha, sizeof(float), hipMemcpyDeviceToHost);
                fprintf(handle->rocblas_logfile, "rocblas_dscal,%d,%lf,%p,%d", n, temp, (void*)x, incx);
            }
            else
            {
                fprintf(handle->rocblas_logfile, "rocblas_dscal,%d,%p,%p,%d", n, alpha, (void*)x, incx);
            }
        }
    }

    rocblas_status status = rocblas_scal_template<double>(handle, n, alpha, x, incx);

    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->layer_mode & rocblas_layer_mode_logging_synch)
        {
            fprintf(handle->rocblas_logfile, ",%d\n", status);
            fflush(handle->rocblas_logfile);
        }
    }

    return status;
}

extern "C" rocblas_status
rocblas_cscal(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* alpha, rocblas_float_complex* x, rocblas_int incx)
{
    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            fprintf(handle->rocblas_logfile, "rocblas_cscal,%d,%f%+fi,%p,%d", n, (*alpha).x, (*alpha).y, (void*)x, incx);
        }
        else
        {
            if(handle->layer_mode & rocblas_layer_mode_logging_synch)
            {
                rocblas_float_complex temp;
                hipMemcpy(&temp, alpha, sizeof(rocblas_float_complex), hipMemcpyDeviceToHost);
                fprintf(handle->rocblas_logfile, "rocblas_cscal,%d,%f%+fi,%p,%d", n, temp.x, temp.y, (void*)x, incx);
            }
            else
            {
                fprintf(handle->rocblas_logfile, "rocblas_cscal,%d,%p,%p,%d", n, alpha, (void*)x, incx);
            }
        }
    }

    rocblas_status status = rocblas_scal_template<rocblas_float_complex>(handle, n, alpha, x, incx);

    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->layer_mode & rocblas_layer_mode_logging_synch)
        {
            fprintf(handle->rocblas_logfile, ",%d\n", status);
            fflush(handle->rocblas_logfile);
        }
    }

    return status;
}

extern "C" rocblas_status
rocblas_zscal(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* alpha, rocblas_double_complex* x, rocblas_int incx)
{
    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            fprintf(handle->rocblas_logfile, "rocblas_zscal,%d,%lf%+lfi,%p,%d", n, (*alpha).x, (*alpha).y, (void*)x, incx);
        }
        else
        {
            if(handle->layer_mode & rocblas_layer_mode_logging_synch)
            {
                rocblas_double_complex temp;
                hipMemcpy(&temp, alpha, sizeof(rocblas_double_complex), hipMemcpyDeviceToHost);
                fprintf(handle->rocblas_logfile, "rocblas_zscal,%d,%lf%+lfi,%p,%d", n, temp.x, temp.y, (void*)x, incx);
            }
            else
            {
                fprintf(handle->rocblas_logfile, "rocblas_zscal,%d,%p,%p,%d", n, alpha, (void*)x, incx);
            }
        }
    }

    rocblas_status status = rocblas_scal_template<rocblas_double_complex>(handle, n, alpha, x, incx);

    if(handle->layer_mode & rocblas_layer_mode_logging)
    {
        if(handle->layer_mode & rocblas_layer_mode_logging_synch)
        {
            fprintf(handle->rocblas_logfile, ",%d\n", status);
            fflush(handle->rocblas_logfile);
        }
    }

    return status;
}
