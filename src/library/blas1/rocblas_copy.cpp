/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"


#define NB_X 256

template<class T>
__global__ void
copy_kernel(hipLaunchParm lp,
    rocblas_int n,
    const T *x, rocblas_int incx,
    T* y,  rocblas_int incy)
{
    int tx  = hipThreadIdx_x;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    //bound
    if ( tid < n ) {
        y[tid*incy] =  x[tid * incx];
    }
}


/*! \brief BLAS Level 1 API

    \details
    copy  copies the vector x[i] into the vector y[i], for  i = 1 , … , n

        y := x,

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
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

template<class T>
rocblas_status
rocblas_copy_template(rocblas_handle handle,
    rocblas_int n,
    const T *x, rocblas_int incx,
    T* y,       rocblas_int incy)
{

    if ( n < 0 )
        return rocblas_invalid_dim;
    else if ( x == NULL )
        return rocblas_invalid_vecX;
    else if ( incx < 0 )
        return rocblas_invalid_incx;
    else if ( y == NULL )
        return rocblas_invalid_vecY;
    else if ( incy < 0 )
        return rocblas_invalid_incy;

    /*
     * Quick return if possible.
     */
    if ( n == 0)
        return rocblas_success;

    int blocks = (n-1)/ NB_X + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads( NB_X, 1, 1 );

    //hipLaunchKernel(HIP_KERNEL_NAME(copy_kernel), dim3(grid), dim3(threads), 0, 0 , n, x, incx, y, incy);

    return rocblas_success;
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
rocblas_copy<float>(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    float* y,       rocblas_int incy){

    return rocblas_copy_template<float>(handle, n, x, incx, y, incy);
}

template<>
rocblas_status
rocblas_copy<double>(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    double* y,       rocblas_int incy){

    return rocblas_copy_template<double>(handle, n, x, incx, y, incy);
}

template<>
rocblas_status
rocblas_copy<rocblas_float_complex>(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex* y,       rocblas_int incy){

    return rocblas_copy_template<rocblas_float_complex>(handle, n, x, incx, y, incy);
}

template<>
rocblas_status
rocblas_copy<rocblas_double_complex>(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex* y,       rocblas_int incy){

    return rocblas_copy_template<rocblas_double_complex>(handle, n, x, incx, y, incy);
}
/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_scopy(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    float* y,       rocblas_int incy){

    return rocblas_copy<float>(handle, n, x, incx, y, incy);
}


extern "C"
rocblas_status
rocblas_dcopy(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    double* y,       rocblas_int incy){

    return rocblas_copy<double>(handle, n, x, incx, y, incy);
}

extern "C"
rocblas_status
rocblas_ccopy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex* y,       rocblas_int incy){

    return rocblas_copy<rocblas_float_complex>(handle, n, x, incx, y, incy);
}

extern "C"
rocblas_status
rocblas_zcopy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex* y,       rocblas_int incy){

    return rocblas_copy<rocblas_double_complex>(handle, n, x, incx, y, incy);
}



/* ============================================================================================ */
