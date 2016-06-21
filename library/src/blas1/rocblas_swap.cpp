/* ************************************************************************
 * swapright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"


#define NB_X 256

template<typename T>
__global__ void
swap_kernel(hipLaunchParm lp,
    rocblas_int n,
    T *x, rocblas_int incx,
    T* y,  rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    T tmp;
    //bound
    if ( tid < n ) {
        tmp = y[tid*incy];
        y[tid*incy] =  x[tid * incx];
        x[tid*incx] =  tmp;
    }
}


/*! \brief BLAS Level 1 API

    \details
    swap  interchange vector x[i] and y[i], for  i = 1 , … , n

        y := x; x := y

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
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

template<class T>
rocblas_status
rocblas_swap_template(rocblas_handle handle,
    rocblas_int n,
    T *x, rocblas_int incx,
    T* y, rocblas_int incy)
{

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( x == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incx < 0 )
        return rocblas_status_invalid_size;
    else if ( y == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incy < 0 )
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    int blocks = (n-1)/ NB_X + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads( NB_X, 1, 1 );

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    hipLaunchKernel(HIP_KERNEL_NAME(swap_kernel), dim3(grid), dim3(threads), 0, rocblas_stream, n, x, incx, y, incy);

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
rocblas_swap<float>(rocblas_handle handle,
    rocblas_int n,
    float *x, rocblas_int incx,
    float* y, rocblas_int incy){

    return rocblas_swap_template<float>(handle, n, x, incx, y, incy);
}

template<>
rocblas_status
rocblas_swap<double>(rocblas_handle handle,
    rocblas_int n,
    double *x, rocblas_int incx,
    double* y, rocblas_int incy){

    return rocblas_swap_template<double>(handle, n, x, incx, y, incy);
}

template<>
rocblas_status
rocblas_swap<rocblas_float_complex>(rocblas_handle handle,
    rocblas_int n,
    rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex* y, rocblas_int incy){

    return rocblas_swap_template<rocblas_float_complex>(handle, n, x, incx, y, incy);
}

template<>
rocblas_status
rocblas_swap<rocblas_double_complex>(rocblas_handle handle,
    rocblas_int n,
    rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex* y, rocblas_int incy){

    return rocblas_swap_template<rocblas_double_complex>(handle, n, x, incx, y, incy);
}

/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_sswap(rocblas_handle handle,
    rocblas_int n,
    float *x, rocblas_int incx,
    float* y, rocblas_int incy){

    return rocblas_swap<float>(handle, n, x, incx, y, incy);
}


extern "C"
rocblas_status
rocblas_dswap(rocblas_handle handle,
    rocblas_int n,
    double *x, rocblas_int incx,
    double* y, rocblas_int incy){

    return rocblas_swap<double>(handle, n, x, incx, y, incy);
}


extern "C"
rocblas_status
rocblas_cswap(rocblas_handle handle,
    rocblas_int n,
    rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex* y, rocblas_int incy){

    return rocblas_swap<rocblas_float_complex>(handle, n, x, incx, y, incy);
}

extern "C"
rocblas_status
rocblas_zswap(rocblas_handle handle,
    rocblas_int n,
    rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex* y, rocblas_int incy){

    return rocblas_swap<rocblas_double_complex>(handle, n, x, incx, y, incy);
}

/* ============================================================================================ */
