/* ************************************************************************
 * trtriright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"
#include "trtri_device.h"

//flag indicate whether write into A or invA
template<typename T, rocblas_int NB, rocblas_int flag>
__global__ void
trtri_kernel(hipLaunchParm lp,
    rocblas_fill uplo,
    rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda,
    T *invA, rocblas_int ldinvA)
{
    trtri_device<T, NB, flag>(uplo, diag, n, A, lda, invA, ldinvA);
}



//because of shared memory size, the NB_X must be <= 64
#define NB_X 32

//assume invA has already been allocated, recommened for repeated calling of trtri product routine
template<typename T>
rocblas_status
rocblas_trtri_template_workspace(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda,
    T *invA, rocblas_int ldinvA)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( A == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( lda < n )
        return rocblas_status_invalid_size;
    else if ( invA == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( ldinvA < n )
        return rocblas_status_invalid_size;
    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    if(n > NB_X ){
        printf("n is %d, n must be less than %d, will return\n", n, NB_X);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipLaunchKernel(HIP_KERNEL_NAME(trtri_kernel<T, NB_X, 1>), dim3(grid), dim3(threads), 0, 0 , uplo, diag, n, A, lda, invA, ldinvA);

    return rocblas_status_success;

}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix  A

        inv(A);

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
    @param[in]
    diag      rocblas_diagonal.
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         rocblas_int.
    @param[in,output]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.

    ********************************************************************/

template<typename T>
rocblas_status
rocblas_trtri_template(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda)
{

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( A == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( lda < n )
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    if(n > NB_X ){
        printf("n is %d must be less than %d, will exit\n", n, NB_X);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipLaunchKernel(HIP_KERNEL_NAME(trtri_kernel<T, NB_X, 0>), dim3(grid), dim3(threads), 0, 0 , uplo, diag, n, A, lda, nullptr, 0);

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
rocblas_trtri<float>(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    float *A, rocblas_int lda){

    return rocblas_trtri_template<float>(handle, uplo, diag, n, A, lda);
}

template<>
rocblas_status
rocblas_trtri<double>(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    double *A, rocblas_int lda){

    return rocblas_trtri_template<double>(handle, uplo, diag, n, A, lda);
}


/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_strtri(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    float *A, rocblas_int lda){

    return rocblas_trtri<float>(handle, uplo, diag, n, A, lda);
}

extern "C"
rocblas_status
rocblas_dtrtri(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    double *A, rocblas_int lda){

    return rocblas_trtri<double>(handle, uplo, diag, n, A, lda);
}

/* ============================================================================================ */
