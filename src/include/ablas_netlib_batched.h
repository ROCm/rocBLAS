/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#pragma once
#ifndef _ABLAS_BATCHED_H_
#define _ABLAS_BATCHED_H_

#include <ablas_types.h>

/*!\file
 * \brief ablas_netlib.h provides the Batched Basic Linear Algebra Subprograms of Level 1, 2 and 3,
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
    handle    ablas_handle.
              handle to the ablas library context queue.
    @param[in]
    trans     ablas_transpose
    @param[in]
    m         ablas_int
    @param[in]
    n         ablas_int
    @param[in]
    alpha    
              specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       ablas_int
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
    incy      ablas_int
              specifies the increment for the elements of y.
    @param[in]
    batchCount
              ablas_int
              specifies the number of matrices.

    ********************************************************************/
    ablas_int ablas_sgemv_batched(ablas_handle handle,
                     ablas_transpose trans, 
                     ablas_int m, ablas_int n, 
                     const float *alpha, 
                     const float *A, ablas_int lda, 
                     const float *x, ablas_int incx, 
                     const float *beta, 
                     float *y, ablas_int incy, 
                     ablas_int batchCount);

    ablas_int ablas_dgemv_batched(ablas_handle handle,
                     ablas_transpose trans, 
                     ablas_int m, ablas_int n, 
                     const double *alpha, 
                     const double *A, ablas_int lda, 
                     const double *x, ablas_int incx, 
                     const double *beta, 
                     ablas_double *y, ablas_int incy,
                     ablas_int batchCount);

    ablas_int ablas_cgemv_batched(ablas_handle handle,
                     ablas_transpose trans, 
                     ablas_int m, ablas_int n, 
                     const ablas_float_complex *alpha, 
                     const ablas_float_complex *A, ablas_int lda, 
                     const ablas_float_complex *x, ablas_int incx, 
                     const ablas_float_complex *beta, 
                     ablas_float_complex *y, ablas_int incy,
                     ablas_int batchCount);

    ablas_int ablas_zgemv_batched(ablas_handle handle,
                     ablas_transpose trans, 
                     ablas_int m, ablas_int n, 
                     const ablas_double_complex *alpha, 
                     const ablas_double_complex *A, ablas_int lda, 
                     const ablas_double_complex *x, ablas_int incx, 
                     const ablas_double_complex *beta, 
                     ablas_double_complex *y, ablas_int incy, 
                     ablas_int batchCount);



	/*
	 * ===========================================================================
	 *    level 3 Batched BLAS
	 * ===========================================================================
	 */


/*! \brief batched BLAS Level 3 API

    \details 
    Assume individual matrix X[i] is stored continuously in memory of X, for i= 0,1,..,batchCount-1. 
    We use X to refer to individual X[i] in the following documentation

    xGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    @param[in]
    handle    ablas_handle.
              handle to the ablas library context queue.
    @param[in]
    transA    ablas_transpose
			  specifies the form of op( A )
    @param[in]
    transB    ablas_transpose
              specifies the form of op( B )
    @param[in]
    m         ablas_int.
    @param[in]
    n         ablas_int.
    @param[in]
    k         ablas_int. 
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       ablas_int
              specifies the leading dimension of A.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       ablas_int
              specifies the leading dimension of B.    
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       ablas_int
              specifies the leading dimension of C.
    @param[in]
    batchCount
              ablas_int
              specifies the number of matrices.

    ********************************************************************/

    void ablas_sgemm_batched(ablas_handle handle,
                     	     ablas_transpose transa, ablas_transpose transb, 
			     ablas_int m, ablas_int n, ablas_int k, 
			     const float *alpha, 
			     const float *A, ablas_int lda, 
                             const float *B, ablas_int ldb, 
                             const float *beta, 
	                     float *C, ablas_int ldc, 
                             ablas_int batchCount);

    void ablas_dgemm_batched(ablas_handle handle,
                     	     ablas_transpose transa, ablas_transpose transb, 
			     ablas_int m, ablas_int n, ablas_int k, 
			     const double *alpha, 
			     const double *A, ablas_int lda, 
                             const double *B, ablas_int ldb, 
                             const double *beta, 
	                     float *C, ablas_int ldc, 
                             ablas_int batchCount);

    void ablas_cgemm_batched(ablas_handle handle,
                             ablas_transpose transa, ablas_transpose transb, 
			     ablas_int m, ablas_int n, ablas_int k, 
			     const ablas_float_complex *alpha, 
			     const ablas_float_complex *A, ablas_int lda, 
                             const ablas_float_complex *B, ablas_int ldb, 
                             const ablas_float_complex *beta, 
	                     ablas_float_complex *C, ablas_int ldc, 
                             ablas_int batchCount);

    void ablas_zgemm_batched(ablas_handle handle,
			     ablas_transpose transa, ablas_transpose transb, 
			     ablas_int m, ablas_int n, ablas_int k, 
			     const ablas_double_complex *alpha, 
			     const ablas_double_complex *A, ablas_int lda, 
                             const ablas_double_complex *B, ablas_int ldb, 
                             const ablas_double_complex *beta, 
	                     ablas_double_complex *C, ablas_int ldc,
                             ablas_int batchCount);


#ifdef __cplusplus
}
#endif

#endif  /* _ABLAS_BATCHED_H_ */
