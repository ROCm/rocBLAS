/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_FUNCTIONS_H_
#define _ROCBLAS_FUNCTIONS_H_

#include "rocblas_types.h"


/*!\file
 * \brief rocblas_netlib.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.
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

ROCBLAS_EXPORT rocblas_status
rocblas_sscal(rocblas_handle handle,
    rocblas_int n,
    const float *alpha,
    float *x, rocblas_int incx);

ROCBLAS_EXPORT rocblas_status
rocblas_dscal(rocblas_handle handle,
    rocblas_int n,
    const double *alpha,
    double *x, rocblas_int incx);

ROCBLAS_EXPORT rocblas_status
rocblas_cscal(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *alpha,
    rocblas_float_complex *x, rocblas_int incx);

ROCBLAS_EXPORT rocblas_status
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

ROCBLAS_EXPORT rocblas_status
rocblas_scopy(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    float* y,       rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_dcopy(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    double* y,       rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_ccopy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex* y,       rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_zcopy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex* y,       rocblas_int incy);


/*! \brief BLAS Level 1 API

    \details
    dot(u)  perform dot product of vector x and y

        result = x * y;

    dotc  perform dot product of complex vector x and complex y

        result = conjugate (x) * y;

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the dot product. either on the host CPU or device GPU.
              return is 0.0 if n <= 0.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status
rocblas_sdot(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    const float *y, rocblas_int incy,
    float *result);

ROCBLAS_EXPORT rocblas_status
rocblas_ddot(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    const double *y, rocblas_int incy,
    double *result);

ROCBLAS_EXPORT rocblas_status
rocblas_cdotu(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *y, rocblas_int incy,
    rocblas_float_complex *result);

ROCBLAS_EXPORT rocblas_status
rocblas_zdotu(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *y, rocblas_int incy,
    rocblas_double_complex *result);


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

ROCBLAS_EXPORT rocblas_status
rocblas_sswap(rocblas_handle handle,
    rocblas_int n,
    float *x, rocblas_int incx,
    float* y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_dswap(rocblas_handle handle,
    rocblas_int n,
    double *x, rocblas_int incx,
    double* y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_cswap(rocblas_handle handle,
    rocblas_int n,
    rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex* y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_zswap(rocblas_handle handle,
    rocblas_int n,
    rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex* y, rocblas_int incy);


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

ROCBLAS_EXPORT rocblas_status
rocblas_saxpy(rocblas_handle handle,
    rocblas_int n,
    const float *alpha,
    const float *x, rocblas_int incx,
    float *y,  rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_daxpy(rocblas_handle handle,
    rocblas_int n,
    const double *alpha,
    const double *x, rocblas_int incx,
    double *y,  rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_caxpy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_float_complex *y,  rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_zaxpy(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex *y,  rocblas_int incy);


/*! \brief BLAS Level 1 API

    \details
    asum computes the sum of the magnitudes of elements of a real vector x,
         or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the asum product. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status
rocblas_sasum(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    float *result);

ROCBLAS_EXPORT rocblas_status
rocblas_dasum(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    double *result);

ROCBLAS_EXPORT rocblas_status
rocblas_scasum(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    float *result);

ROCBLAS_EXPORT rocblas_status
rocblas_dzasum(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    double *result);



/*! \brief BLAS Level 1 API

    \details
    nrm2 computes the euclidean norm of a real or complex vector
              := sqrt( x'*x ) for real vector
              := sqrt( x**H*x ) for complex vector

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the nrm2 product. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status
rocblas_snrm2(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    float *result);

ROCBLAS_EXPORT rocblas_status
rocblas_dnrm2(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    double *result);

ROCBLAS_EXPORT rocblas_status
rocblas_scnrm2(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    float *result);

ROCBLAS_EXPORT rocblas_status
rocblas_dznrm2(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    double *result);


/*! \brief BLAS Level 1 API

    \details
    amax finds the first index of the element of maximum magnitude of real vector x
         or the sum of magnitude of the real and imaginary parts of elements if x is a complex vector

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the amax product. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status
rocblas_samax(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    rocblas_int *result);

ROCBLAS_EXPORT rocblas_status
rocblas_damax(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    rocblas_int *result);

ROCBLAS_EXPORT rocblas_status
rocblas_scamax(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_int *result);

ROCBLAS_EXPORT rocblas_status
rocblas_dzamax(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_int *result);

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

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status
rocblas_sgemv(rocblas_handle handle,
                 rocblas_operation trans,
                 rocblas_int m, rocblas_int n,
                 const float *alpha,
                 const float *A, rocblas_int lda,
                 const float *x, rocblas_int incx,
                 const float *beta,
                 float *y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_dgemv(rocblas_handle handle,
                 rocblas_operation trans,
                 rocblas_int m, rocblas_int n,
                 const double *alpha,
                 const double *A, rocblas_int lda,
                 const double *x, rocblas_int incx,
                 const double *beta,
                 double *y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_cgemv(rocblas_handle handle,
                 rocblas_operation trans,
                 rocblas_int m, rocblas_int n,
                 const rocblas_float_complex *alpha,
                 const rocblas_float_complex *A, rocblas_int lda,
                 const rocblas_float_complex *x, rocblas_int incx,
                 const rocblas_float_complex *beta,
                 rocblas_float_complex *y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_zgemv(rocblas_handle handle,
                 rocblas_operation trans,
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
    uplo      rocblas_fill.
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
ROCBLAS_EXPORT rocblas_status
rocblas_ssymv(rocblas_handle handle,
                 rocblas_fill uplo,
                 rocblas_int n,
                 const float *alpha,
                 const float *A, rocblas_int lda,
                 const float *x, rocblas_int incx,
                 const float *beta,
                 float *y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_dsymv(rocblas_handle handle,
                 rocblas_fill uplo,
                 rocblas_int n,
                 const double *alpha,
                 const double *A, rocblas_int lda,
                 const double *x, rocblas_int incx,
                 const double *beta,
                 double *y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_chemv(rocblas_handle handle,
                 rocblas_fill uplo,
                 rocblas_int n,
                 const rocblas_float_complex *alpha,
                 const rocblas_float_complex *A, rocblas_int lda,
                 const rocblas_float_complex *x, rocblas_int incx,
                 const rocblas_float_complex *beta,
                 rocblas_float_complex *y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status
rocblas_zhemv(rocblas_handle handle,
                 rocblas_fill uplo,
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
    transA    rocblas_operation
              specifies the form of op( A )
    @param[in]
    transB    rocblas_operation
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
ROCBLAS_EXPORT rocblas_status
rocblas_hgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lda,
    const rocblas_half *B, rocblas_int ldb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_sgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lda,
    const float *B, rocblas_int ldb,
    const float *beta,
          float *C, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_dgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lda,
    const double *B, rocblas_int ldb,
    const double *beta,
          double *C, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_qgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lda,
    const rocblas_half_complex *B, rocblas_int ldb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_cgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_zgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ldc);


/***************************************************************************
 * strided - specify leading stride
 * lsa - non-1 leading stride of a
 * lsb - non-1 leading stride of b
 * lsc - non-1 leading stride of c
 **************************************************************************/
ROCBLAS_EXPORT rocblas_status
rocblas_hgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_half *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int lsc, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_sgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lsa, rocblas_int lda,
    const float *B, rocblas_int lsb, rocblas_int ldb,
    const float *beta,
          float *C, rocblas_int lsc, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_dgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lsa, rocblas_int lda,
    const double *B, rocblas_int lsb, rocblas_int ldb,
    const double *beta,
          double *C, rocblas_int lsc, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_qgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_half_complex *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int lsc, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_cgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int lsc, rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status
rocblas_zgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int lsc, rocblas_int ldc);

/***************************************************************************
 * batched
 * bsa - "batch stride a": stride from the start of one "A" matrix to the next
 * bsb
 * bsc
 * batch_count - numbers of gemm's in the batch
 **************************************************************************/
ROCBLAS_EXPORT rocblas_status
rocblas_hgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_half *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_sgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lda, rocblas_int bsa,
    const float *B, rocblas_int ldb, rocblas_int bsb,
    const float *beta,
          float *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_dgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lda, rocblas_int bsa,
    const double *B, rocblas_int ldb, rocblas_int bsb,
    const double *beta,
          double *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_qgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_half_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_cgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_float_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_zgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_double_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );


/***************************************************************************
 * strided & batched
 * lsa - non-1 leading stride of a
 * lsb - non-1 leading stride of b
 * lsc - non-1 leading stride of c
 * bsa - "batch stride a": stride from the start of one "A" matrix to the next
 * bsb
 * bsc
 * batch_count - numbers of gemm's in the batch
 **************************************************************************/
ROCBLAS_EXPORT rocblas_status
rocblas_hgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_half *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_sgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const float *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const float *beta,
          float *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_dgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const double *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const double *beta,
          double *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_qgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_half_complex *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_cgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_float_complex *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_zgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_double_complex *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count );

#ifdef __cplusplus
}
#endif

#endif  /* _ROCBLAS_FUNCTIONS_H_ */
