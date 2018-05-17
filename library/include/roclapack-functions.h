/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCLAPACK_FUNCTIONS_H
#define _ROCLAPACK_FUNCTIONS_H

#include "rocblas-types.h"

/*!\file
 * \brief rocblas_netlib.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based
 * NVIDIA GPUs.
 *  This file exposes C89 BLAS interface
*/

/*
 * ===========================================================================
 *   READEME: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

#ifdef __cplusplus
extern "C" {
#endif
    
    /*! \brief LAPACK API

    \details
    potf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

        A = U' * U ,  if UPLO = 'U', or
        A = L  * L',  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper or lower
    @param[in]
    n         the matrix dimensions
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.


    ********************************************************************/
    
ROCBLAS_EXPORT rocblas_status
roclapack_spotf2(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda);

/*! \brief LAPACK API

    \details
    potf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

        A = U' * U ,  if UPLO = 'U', or
        A = L  * L',  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper or lower
    @param[in]
    n         the matrix dimensions
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.


    ********************************************************************/
    
ROCBLAS_EXPORT rocblas_status
roclapack_dpotf2(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda);

#ifdef __cplusplus
}
#endif

#endif /* _ROCLAPACK_FUNCTIONS_H */

