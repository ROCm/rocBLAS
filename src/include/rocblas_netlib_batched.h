/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_BATCHED_H_
#define _ROCBLAS_BATCHED_H_

#include <rocblas_types.h>

/*!\file
 * \brief rocblas_netlib.h provides the Batched Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.
*/


#ifdef __cplusplus
extern "C" {
#endif

    /*
     * ===========================================================================
     *    level 2 Batched BLAS
     * ===========================================================================
     */


/*! \brief batched BLAS Level 2 API

    \details
    Assume individual matrix X[i] is stored continuously in memory of X, for i= 0,1,..,batchCount-1.
    We use X to refer to individual X[i] in the following documentation

    xGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    trans     rocblas_operation
    @param[in]
    m         rocblas_int
    @param[in]
    n         rocblas_int
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[in]
    beta      specifies the scalar beta.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.
    @param[in]
    batchCount
              rocblas_int
              specifies the number of matrices.

    ********************************************************************/
    rocblas_status rocblas_sgemv_batched(rocblas_handle handle,
                     rocblas_operation trans,
                     rocblas_int m, rocblas_int n,
                     const float *alpha,
                     const float *A, rocblas_int lda,
                     const float *x, rocblas_int incx,
                     const float *beta,
                     float *y, rocblas_int incy,
                     rocblas_int batchCount);

    rocblas_status rocblas_dgemv_batched(rocblas_handle handle,
                     rocblas_operation trans,
                     rocblas_int m, rocblas_int n,
                     const double *alpha,
                     const double *A, rocblas_int lda,
                     const double *x, rocblas_int incx,
                     const double *beta,
                     double *y, rocblas_int incy,
                     rocblas_int batchCount);

    rocblas_status rocblas_cgemv_batched(rocblas_handle handle,
                     rocblas_operation trans,
                     rocblas_int m, rocblas_int n,
                     const rocblas_float_complex *alpha,
                     const rocblas_float_complex *A, rocblas_int lda,
                     const rocblas_float_complex *x, rocblas_int incx,
                     const rocblas_float_complex *beta,
                     rocblas_float_complex *y, rocblas_int incy,
                     rocblas_int batchCount);

    rocblas_status rocblas_zgemv_batched(rocblas_handle handle,
                     rocblas_operation trans,
                     rocblas_int m, rocblas_int n,
                     const rocblas_double_complex *alpha,
                     const rocblas_double_complex *A, rocblas_int lda,
                     const rocblas_double_complex *x, rocblas_int incx,
                     const rocblas_double_complex *beta,
                     rocblas_double_complex *y, rocblas_int incy,
                     rocblas_int batchCount);



    /*
     * ===========================================================================
     *    level 3 Batched BLAS
     * ===========================================================================
     */
    // moved to regular netlib.h


#ifdef __cplusplus
}
#endif

#endif  /* _ROCBLAS_BATCHED_H_ */
