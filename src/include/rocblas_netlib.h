/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_NETLIB_H_
#define _ROCBLAS_NETLIB_H_

#include <rocblas_types.h>


/*!\file
 * \brief rocblas_netlib.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.   
*/

    /*
     * ===========================================================================
     *   READEME: Please follow the naming convention
     *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
     *     lower case for vector, e.g. vector x, y    GEMV (y = A*x)

     *   rocblas_handle is defined as rocblas_queue (hipStream_t) right now 
     *   to allow multiple BLAS routines called in different streams(or called queues)
     *   on the same GPU device. This feature is called streaming supported by CUBLAS,too. 
     * ===========================================================================
     */


#ifdef __cplusplus
extern "C" {
#endif


    /*
     * ===========================================================================
     *    level 1 BLAS
     * ===========================================================================
     */

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

    rocblas_status
    rocblas_sscal(rocblas_handle handle,
        rocblas_int n,
        const float *alpha,
        float *x, rocblas_int incx);

    rocblas_status
    rocblas_dscal(rocblas_handle handle,
        rocblas_int n,
        const double *alpha,
        double *x, rocblas_int incx);

    rocblas_status
    rocblas_cscal(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *alpha,
        rocblas_float_complex *x, rocblas_int incx);

    rocblas_status
    rocblas_zscal(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *alpha,
        rocblas_double_complex *x, rocblas_int incx);


/*! \brief BLAS Level 1 API

    \details 
    copy  copies the vector x into the vector y, for  i = 1 , … , n 

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

    rocblas_status
    rocblas_scopy(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx, 
        float* y,       rocblas_int incy);

    rocblas_status
    rocblas_dcopy(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx, 
        double* y,       rocblas_int incy);




    /*
     * ===========================================================================
     *    level 2 BLAS
     * ===========================================================================
     */

/*! \brief BLAS Level 2 API

    \details 
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
    trans     rocblas_transpose
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

    ********************************************************************/
    rocblas_status rocblas_sgemv(rocblas_handle handle,
                     rocblas_transpose trans, 
                     rocblas_int m, rocblas_int n, 
                     const float *alpha, 
                     const float *A, rocblas_int lda, 
                     const float *x, rocblas_int incx, 
                     const float *beta, 
                     float *y, rocblas_int incy);

    rocblas_status rocblas_dgemv(rocblas_handle handle,
                     rocblas_transpose trans, 
                     rocblas_int m, rocblas_int n, 
                     const double *alpha, 
                     const double *A, rocblas_int lda, 
                     const double *x, rocblas_int incx, 
                     const double *beta, 
                     double *y, rocblas_int incy);

    rocblas_status rocblas_cgemv(rocblas_handle handle,
                     rocblas_transpose trans, 
                     rocblas_int m, rocblas_int n, 
                     const rocblas_float_complex *alpha, 
                     const rocblas_float_complex *A, rocblas_int lda, 
                     const rocblas_float_complex *x, rocblas_int incx, 
                     const rocblas_float_complex *beta, 
                     rocblas_float_complex *y, rocblas_int incy);

    rocblas_status rocblas_zgemv(rocblas_handle handle,
                     rocblas_transpose trans, 
                     rocblas_int m, rocblas_int n, 
                     const rocblas_double_complex *alpha, 
                     const rocblas_double_complex *A, rocblas_int lda, 
                     const rocblas_double_complex *x, rocblas_int incx, 
                     const rocblas_double_complex *beta, 
                     rocblas_double_complex *y, rocblas_int incy);


/*! \brief BLAS Level 2 API

    \details 
    xHE(SY)MV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n Hermitian(Symmetric) matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_uplo.
              specifies whether the upper or lower
    @param[in]
    n         rocblas_int.
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

    ********************************************************************/
    rocblas_status rocblas_ssymv(rocblas_handle handle,
                     rocblas_uplo uplo, 
                     rocblas_int n, 
                     const float *alpha, 
                     const float *A, rocblas_int lda, 
                     const float *x, rocblas_int incx, 
                     const float *beta, 
                     float *y, rocblas_int incy);

    rocblas_status rocblas_dsymv(rocblas_handle handle,
                     rocblas_uplo uplo, 
                     rocblas_int n, 
                     const double *alpha, 
                     const double *A, rocblas_int lda, 
                     const double *x, rocblas_int incx, 
                     const double *beta, 
                     double *y, rocblas_int incy);

    rocblas_status rocblas_chemv(rocblas_handle handle,
                     rocblas_uplo uplo, 
                     rocblas_int n, 
                     const rocblas_float_complex *alpha, 
                     const rocblas_float_complex *A, rocblas_int lda, 
                     const rocblas_float_complex *x, rocblas_int incx, 
                     const rocblas_float_complex *beta, 
                     rocblas_float_complex *y, rocblas_int incy);

    rocblas_status rocblas_zhemv(rocblas_handle handle,
                     rocblas_uplo uplo, 
                     rocblas_int n, 
                     const rocblas_double_complex *alpha, 
                     const rocblas_double_complex *A, rocblas_int lda, 
                     const rocblas_double_complex *x, rocblas_int incx, 
                     const rocblas_double_complex *beta, 
                     rocblas_double_complex *y, rocblas_int incy);


    /*
     * ===========================================================================
     *    level 3 BLAS
     * ===========================================================================
     */


/*! \brief BLAS Level 3 API

    \details 
    xGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_transpose
              specifies the form of op( A )
    @param[in]
    transB    rocblas_transpose
              specifies the form of op( B )
    @param[in]
    m         rocblas_int.
    @param[in]
    n         rocblas_int.
    @param[in]
    k         rocblas_int. 
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.    
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.

    ********************************************************************/
    rocblas_status rocblas_sgemm(rocblas_handle handle,
                     rocblas_transpose transa, rocblas_transpose transb, 
                     rocblas_int m, rocblas_int n, rocblas_int k, 
                     const float *alpha, 
                     const float *A, rocblas_int lda, 
                     const float *B, rocblas_int ldb, 
                     const float *beta, 
                     float *C, rocblas_int ldc);

    rocblas_status rocblas_dgemm(rocblas_handle handle,
                     rocblas_transpose transa, rocblas_transpose transb, 
                     rocblas_int m, rocblas_int n, rocblas_int k, 
                     const double *alpha, 
                     const double *A, rocblas_int lda, 
                     const double *B, rocblas_int ldb, 
                     const double *beta, 
                     float *C, rocblas_int ldc);

    rocblas_status rocblas_cgemm(rocblas_handle handle,
                     rocblas_transpose transa, rocblas_transpose transb, 
                     rocblas_int m, rocblas_int n, rocblas_int k, 
                     const rocblas_float_complex *alpha, 
                     const rocblas_float_complex *A, rocblas_int lda, 
                     const rocblas_float_complex *B, rocblas_int ldb, 
                     const rocblas_float_complex *beta, 
                     rocblas_float_complex *C, rocblas_int ldc);

    rocblas_status rocblas_zgemm(rocblas_handle handle,
                     rocblas_transpose transa, rocblas_transpose transb, 
                     rocblas_int m, rocblas_int n, rocblas_int k, 
                     const rocblas_double_complex *alpha, 
                     const rocblas_double_complex *A, rocblas_int lda, 
                     const rocblas_double_complex *B, rocblas_int ldb, 
                     const rocblas_double_complex *beta, 
                     rocblas_double_complex *C, rocblas_int ldc);



#ifdef __cplusplus
}
#endif

#endif  /* _ROCBLAS_NETLIB_H_ */
