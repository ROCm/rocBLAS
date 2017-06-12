/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include <hip/hip_runtime.h>

 

#include "rocblas.h"
 
#include "definitions.h"

#define NB_X 256

template<typename T>
__global__ void
axpy_kernel_host_scalar(hipLaunchParm lp,
    rocblas_int n,
    const T alpha,
    const T *x, rocblas_int incx,
    T *y,  rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    //bound
    if ( tid < n ) {
        y[tid * incy] +=  (alpha) * (x[tid * incx]);
    }
}

template<typename T>
__global__ void
axpy_kernel_device_scalar(hipLaunchParm lp,
    rocblas_int n,
    const T *alpha,
    const T *x, rocblas_int incx,
    T *y,  rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    //bound
    if ( tid < n ) {
        y[tid * incy] +=  (*alpha) * (x[tid * incx]);
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

template<class T>
rocblas_status
rocblas_axpy_template(rocblas_handle handle,
    rocblas_int n,
    const T *alpha,
    const T *x, rocblas_int incx,
    T *y,  rocblas_int incy)
{

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( alpha == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( x == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incx < 0 )
        return rocblas_status_invalid_size;
    else if ( y == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incy < 0 )
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */

    if ( n == 0 )
        return rocblas_status_success;

    int blocks = (n-1)/ NB_X + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    if( rocblas_pointer_to_mode((void*)alpha) == rocblas_pointer_mode_device ){
        hipLaunchKernel(HIP_KERNEL_NAME(axpy_kernel_device_scalar), dim3(blocks), dim3(threads), 0, rocblas_stream, n, alpha, x, incx, y, incy);
    }
    else{// alpha is on host
        T scalar = *alpha;
        hipLaunchKernel(HIP_KERNEL_NAME(axpy_kernel_host_scalar), dim3(blocks), dim3(threads), 0, rocblas_stream, n, scalar, x, incx, y, incy);
    }

    return rocblas_status_success;
}



/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C89 wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_saxpy(rocblas_handle handle,
    rocblas_int n,
    const float *alpha,
    const float *x, rocblas_int incx,
    float *y,  rocblas_int incy){

    return rocblas_axpy_template<float>(handle, n, alpha, x, incx, y, incy);
}

extern "C"
rocblas_status
rocblas_daxpy(rocblas_handle handle,
    rocblas_int n,
    const double *alpha,
    const double *x, rocblas_int incx,
    double *y,  rocblas_int incy){

    return rocblas_axpy_template<double>(handle, n, alpha, x, incx, y, incy);
}

extern "C"
rocblas_status
rocblas_caxpy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex *y,  rocblas_int incy){

    return rocblas_axpy_template<rocblas_float_complex>(handle, n, alpha, x, incx, y, incy);
}

extern "C"
rocblas_status
rocblas_zaxpy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex *y,  rocblas_int incy){

    return rocblas_axpy_template<rocblas_double_complex>(handle, n, alpha, x, incx, y, incy);
}


/* ============================================================================================ */
