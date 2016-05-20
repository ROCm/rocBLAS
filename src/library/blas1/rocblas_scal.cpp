/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"

#define NB_X 256

template<typename T>
__global__ void
scal_kernel_host_scalar(hipLaunchParm lp,
    rocblas_int n,
    const T alpha,
    T *x, rocblas_int incx)
{
    int tx  = hipThreadIdx_x;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    //bound
    if ( tid < n ) {
        x[tid * incx] =  (alpha) * (x[tid * incx]);
    }
}

template<typename T>
__global__ void
scal_kernel_device_scalar(hipLaunchParm lp,
    rocblas_int n,
    const T *alpha,
    T *x, rocblas_int incx)
{
    int tx  = hipThreadIdx_x;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    //bound
    if ( tid < n ) {
        x[tid * incx] =  (*alpha) * (x[tid * incx]);
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
    @param[in]
    alpha     specifies the scalar alpha.
    @param[inout]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.


    ********************************************************************/

template<class T>
rocblas_status
rocblas_scal_template(rocblas_handle handle,
    rocblas_int n,
    const T *alpha,
    T *x, rocblas_int incx)
{

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( x == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incx < 0 )
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */

    if ( n == 0 )
        return rocblas_status_success;

    int blocks = (n-1)/ NB_X + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads(NB_X, 1, 1);

    if( rocblas_get_pointer_location((void*)alpha) == rocblas_mem_location_device ){
        hipLaunchKernel(HIP_KERNEL_NAME(scal_kernel_device_scalar), dim3(blocks), dim3(threads), 0, 0 , n, alpha, x, incx);
    }
    else{
        printf("i am in host branch\n");
        T scalar = *alpha;
        hipLaunchKernel(HIP_KERNEL_NAME(scal_kernel_host_scalar), dim3(blocks), dim3(threads), 0, 0 , n, scalar, x, incx);
    }

    return rocblas_status_success;
}

/* ============================================================================================ */

    /*
     * ===========================================================================
     *    template interface
     *    template specialization
     * ===========================================================================
     */


template<>
rocblas_status
rocblas_scal<float>(rocblas_handle handle,
    rocblas_int n,
    const float *alpha,
    float *x, rocblas_int incx){

    return rocblas_scal_template<float>(handle, n, alpha, x, incx);
}

template<>
rocblas_status
rocblas_scal<double>(rocblas_handle handle,
    rocblas_int n,
    const double *alpha,
    double *x, rocblas_int incx){

    return rocblas_scal_template<double>(handle, n, alpha, x, incx);
}

template<>
rocblas_status
rocblas_scal<rocblas_float_complex>(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *alpha,
    rocblas_float_complex *x, rocblas_int incx){

    return rocblas_scal_template<rocblas_float_complex>(handle, n, alpha, x, incx);
}

template<>
rocblas_status
rocblas_scal<rocblas_double_complex>(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *alpha,
    rocblas_double_complex *x, rocblas_int incx){

    return rocblas_scal_template<rocblas_double_complex>(handle, n, alpha, x, incx);
}

/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C89 wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_sscal(rocblas_handle handle,
    rocblas_int n,
    const float *alpha,
    float *x, rocblas_int incx){

    return rocblas_scal<float>(handle, n, alpha, x, incx);
}

extern "C"
rocblas_status
rocblas_dscal(rocblas_handle handle,
    rocblas_int n,
    const double *alpha,
    double *x, rocblas_int incx){

    return rocblas_scal<double>(handle, n, alpha, x, incx);
}


extern "C"
rocblas_status
rocblas_cscal(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *alpha,
    rocblas_float_complex *x, rocblas_int incx){

    return rocblas_scal<rocblas_float_complex>(handle, n, alpha, x, incx);
}

extern "C"
rocblas_status
rocblas_zscal(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *alpha,
    rocblas_double_complex *x, rocblas_int incx){

    return rocblas_scal<rocblas_double_complex>(handle, n, alpha, x, incx);
}



/* ============================================================================================ */
