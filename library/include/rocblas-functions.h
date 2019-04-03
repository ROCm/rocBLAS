/* ************************************************************************
 * Copyright 2016-2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_FUNCTIONS_H_
#define _ROCBLAS_FUNCTIONS_H_

#include "rocblas-types.h"

/*!\file
 * \brief rocblas_functions.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based
 * NVIDIA GPUs.
 *  This file exposes C89 BLAS interface
*/

/*
 * ===========================================================================
 *   README: Please follow the naming convention
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
rocblas_sscal(rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx);

ROCBLAS_EXPORT rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx);

/* not implemented
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
*/

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

ROCBLAS_EXPORT rocblas_status rocblas_scopy(rocblas_handle handle,
                                            rocblas_int n,
                                            const float* x,
                                            rocblas_int incx,
                                            float* y,
                                            rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_dcopy(rocblas_handle handle,
                                            rocblas_int n,
                                            const double* x,
                                            rocblas_int incx,
                                            double* y,
                                            rocblas_int incy);

/* not implemented
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
*/

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

ROCBLAS_EXPORT rocblas_status rocblas_sdot(rocblas_handle handle,
                                           rocblas_int n,
                                           const float* x,
                                           rocblas_int incx,
                                           const float* y,
                                           rocblas_int incy,
                                           float* result);

ROCBLAS_EXPORT rocblas_status rocblas_ddot(rocblas_handle handle,
                                           rocblas_int n,
                                           const double* x,
                                           rocblas_int incx,
                                           const double* y,
                                           rocblas_int incy,
                                           double* result);

/* not implemented
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
*/

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

ROCBLAS_EXPORT rocblas_status rocblas_sswap(
    rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_dswap(
    rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy);

/* not implemented
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
*/

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

ROCBLAS_EXPORT rocblas_status rocblas_haxpy(rocblas_handle handle,
                                            rocblas_int n,
                                            const rocblas_half* alpha,
                                            const rocblas_half* x,
                                            rocblas_int incx,
                                            rocblas_half* y,
                                            rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_saxpy(rocblas_handle handle,
                                            rocblas_int n,
                                            const float* alpha,
                                            const float* x,
                                            rocblas_int incx,
                                            float* y,
                                            rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_daxpy(rocblas_handle handle,
                                            rocblas_int n,
                                            const double* alpha,
                                            const double* x,
                                            rocblas_int incx,
                                            double* y,
                                            rocblas_int incy);

/* not implemented
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
*/

/*! \brief BLAS Level 1 API

    \details
    asum computes the sum of the magnitudes of elements of a real vector x,
         or the sum of magnitudes of the real and imaginary parts of elements if x is a complex
   vector

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

ROCBLAS_EXPORT rocblas_status rocblas_sasum(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result);

ROCBLAS_EXPORT rocblas_status rocblas_dasum(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result);

/* not implemented
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
*/

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

ROCBLAS_EXPORT rocblas_status rocblas_snrm2(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result);

ROCBLAS_EXPORT rocblas_status rocblas_dnrm2(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result);

/* not implemented
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
*/

/*! \brief BLAS Level 1 API

    \details
    amax finds the first index of the element of maximum magnitude of real vector x
         or the sum of magnitude of the real and imaginary parts of elements if x is a complex
   vector

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
              store the amax index. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_isamax(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_idamax(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result);

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_iscamax(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_int *result);

ROCBLAS_EXPORT rocblas_status
rocblas_idzamax(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_int *result);
*/

/*! \brief BLAS Level 1 API

    \details
    amin finds the first index of the element of minimum magnitude of real vector x
         or the sum of magnitude of the real and imaginary parts of elements if x is a complex
   vector

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
              store the amin index. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_isamin(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_idamin(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result);

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_iscamin(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_int *result);

ROCBLAS_EXPORT rocblas_status
rocblas_idzamin(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_int *result);
*/

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
ROCBLAS_EXPORT rocblas_status rocblas_sgemv(rocblas_handle handle,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const float* alpha,
                                            const float* A,
                                            rocblas_int lda,
                                            const float* x,
                                            rocblas_int incx,
                                            const float* beta,
                                            float* y,
                                            rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_dgemv(rocblas_handle handle,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const double* alpha,
                                            const double* A,
                                            rocblas_int lda,
                                            const double* x,
                                            rocblas_int incx,
                                            const double* beta,
                                            double* y,
                                            rocblas_int incy);

/*! \brief BLAS Level 2 API

    \details
    trsv solves

         A*x = alpha*b or A**T*x = alpha*b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     rocblas_operation

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         rocblas_int
              m specifies the number of rows of b. m >= 0.

    @param[in]
    alpha
              specifies the scalar alpha.

    @param[in]
    A         pointer storing matrix A on the GPU,
              of dimension ( lda, m )

    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
              lda = max( 1, m ).

    @param[in]
    x         pointer storing vector x on the GPU.

    @param[in]
    incx      specifies the increment for the elements of x.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_strsv(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal diag,
                                            rocblas_int m,
                                            const float* A,
                                            rocblas_int lda,
                                            float* x,
                                            rocblas_int incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsv(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal diag,
                                            rocblas_int m,
                                            const double* A,
                                            rocblas_int lda,
                                            double* x,
                                            rocblas_int incx);

/* not implemented
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
*/

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
/* not implemented
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
*/

/* not implemented
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
*/

/*! \brief BLAS Level 2 API

    \details
    xGER performs the matrix-vector operations

        A := A + alpha*x*y**T

    where alpha is a scalars, x and y are vectors, and A is an
    m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    m         rocblas_int
    @param[in]
    n         rocblas_int
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
    @param[in]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_sger(rocblas_handle handle,
                                           rocblas_int m,
                                           rocblas_int n,
                                           const float* alpha,
                                           const float* x,
                                           rocblas_int incx,
                                           const float* y,
                                           rocblas_int incy,
                                           float* A,
                                           rocblas_int lda);

ROCBLAS_EXPORT rocblas_status rocblas_dger(rocblas_handle handle,
                                           rocblas_int m,
                                           rocblas_int n,
                                           const double* alpha,
                                           const double* x,
                                           rocblas_int incx,
                                           const double* y,
                                           rocblas_int incy,
                                           double* A,
                                           rocblas_int lda);

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_cger(rocblas_handle handle,
                 rocblas_int m, rocblas_int n,
                 const rocblas_float_complex *alpha,
                 const rocblas_float_complex *x, rocblas_int incx,
                 const rocblas_float_complex *y, rocblas_int incy,
                       rocblas_float_complex *A, rocblas_int lda);

ROCBLAS_EXPORT rocblas_status
rocblas_zger(rocblas_handle handle,
                 rocblas_int m, rocblas_int n,
                 const rocblas_double_complex *alpha,
                 const rocblas_double_complex *x, rocblas_int incx,
                 const rocblas_double_complex *y, rocblas_int incy,
                       rocblas_double_complex *A, rocblas_int lda);
*/

/*! \brief BLAS Level 2 API

    \details
    xSYR performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalars, x is a vector, and A is an
    n by n symmetric matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
    @param[inout]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ssyr(rocblas_handle handle,
                                           rocblas_fill uplo,
                                           rocblas_int n,
                                           const float* alpha,
                                           const float* x,
                                           rocblas_int incx,
                                           float* A,
                                           rocblas_int lda);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr(rocblas_handle handle,
                                           rocblas_fill uplo,
                                           rocblas_int n,
                                           const double* alpha,
                                           const double* x,
                                           rocblas_int incx,
                                           double* A,
                                           rocblas_int lda);

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_csyr(rocblas_handle handle,
                 rocblas_int n,
                 const rocblas_float_complex *alpha,
                 const rocblas_float_complex *x, rocblas_int incx,
                       rocblas_float_complex *A, rocblas_int lda);

ROCBLAS_EXPORT rocblas_status
rocblas_zsyr(rocblas_handle handle,
                 rocblas_int n,
                 const rocblas_double_complex *alpha,
                 const rocblas_double_complex *x, rocblas_int incx,
                       rocblas_double_complex *A, rocblas_int lda);
*/

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix  A, namely, invA

        and write the result into invA;

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    diag      rocblas_diagonal.
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         rocblas_int.
              size of matrix A and invA
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[output]
    invA      pointer storing matrix invA on the GPU.
    @param[in]
    ldinvA    rocblas_int
              specifies the leading dimension of invA.

********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_strtri(rocblas_handle handle,
                                             rocblas_fill uplo,
                                             rocblas_diagonal diag,
                                             rocblas_int n,
                                             const float* A,
                                             rocblas_int lda,
                                             float* invA,
                                             rocblas_int ldinvA);

ROCBLAS_EXPORT rocblas_status rocblas_dtrtri(rocblas_handle handle,
                                             rocblas_fill uplo,
                                             rocblas_diagonal diag,
                                             rocblas_int n,
                                             const double* A,
                                             rocblas_int lda,
                                             double* invA,
                                             rocblas_int ldinvA);

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
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    stride_a  rocblas_int
             "batch stride a": stride from the start of one "A" matrix to the next
    @param[out]
    invA      pointer storing the inverse matrix A on the GPU.
              Partial inplace operation is supported, see below.
              If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
              the inverse of the upper triangular matrix, and the strictly lower
              triangular part of invA is cleared.
              If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
              the inverse of the lower triangular matrix, and the strictly upper
              triangular part of invA is cleared.
    @param[in]
    ldinvA    rocblas_int
              specifies the leading dimension of invA.
    @param[in]
    stride_invA rocblas_int
             "batch stride invA": stride from the start of one "invA" matrix to the next
    @param[in]
    batch_count       rocblas_int
              numbers of matrices in the batch
    ********************************************************************/
// assume invA has already been allocated, recommended for repeated calling of trtri product routine

ROCBLAS_EXPORT rocblas_status rocblas_strtri_batched(rocblas_handle handle,
                                                     rocblas_fill uplo,
                                                     rocblas_diagonal diag,
                                                     rocblas_int n,
                                                     const float* A,
                                                     rocblas_int lda,
                                                     rocblas_int stride_a,
                                                     float* invA,
                                                     rocblas_int ldinvA,
                                                     rocblas_int stride_invA,
                                                     rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrtri_batched(rocblas_handle handle,
                                                     rocblas_fill uplo,
                                                     rocblas_diagonal diag,
                                                     rocblas_int n,
                                                     const double* A,
                                                     rocblas_int lda,
                                                     rocblas_int stride_a,
                                                     double* invA,
                                                     rocblas_int ldinvA,
                                                     rocblas_int stride_invA,
                                                     rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    trsm solves

        op(A)*X = alpha*B or  X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices,
    A is triangular matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    The matrix X is overwritten on B.

    Note about memory allocation:
    When trsm is launched with a k evenly divisible by the internal block size of 128,
    and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
    memory found in the handle to increase overall performance. This memory can be managed by using
    the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
    used for temporary storage will default to 1 MB and may result in chunking, which in turn may
    reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
    to the desired chunk of right hand sides to be used at a time.

    (where k is m when rocblas_side_left and is n when rocblas_side_right)

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    side    rocblas_side.
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  rocblas_operation.
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       rocblas_int.
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       rocblas_int.
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            &zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            of dimension ( lda, k ), where k is m
            when  rocblas_side_left  and
            is  n  when  rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     rocblas_int.
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in,output]
    B       pointer storing matrix B on the GPU.

    @param[in]
    ldb    rocblas_int.
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_strsm(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal diag,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const float* alpha,
                                            const float* A,
                                            rocblas_int lda,
                                            float* B,
                                            rocblas_int ldb);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsm(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal diag,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const double* alpha,
                                            const double* A,
                                            rocblas_int lda,
                                            double* B,
                                            rocblas_int ldb);

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
    handle    rocblas_handle,
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_operation,
              specifies the form of op( A )
    @param[in]
    transB    rocblas_operation,
              specifies the form of op( B )
    @param[in]
    m         rocblas_int,
              number or rows of matrices op( A ) and C
    @param[in]
    n         rocblas_int,
              number of columns of matrices op( B ) and C
    @param[in]
    k         rocblas_int,
              number of columns of matrix op( A ) and number of rows of matrix op( B )
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int,
              specifies the leading dimension of A.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       rocblas_int,
              specifies the leading dimension of B.
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       rocblas_int,
              specifies the leading dimension of C.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_hgemm(rocblas_handle handle,
                                            rocblas_operation transa,
                                            rocblas_operation transb,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            const rocblas_half* alpha,
                                            const rocblas_half* A,
                                            rocblas_int lda,
                                            const rocblas_half* B,
                                            rocblas_int ldb,
                                            const rocblas_half* beta,
                                            rocblas_half* C,
                                            rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm(rocblas_handle handle,
                                            rocblas_operation transa,
                                            rocblas_operation transb,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            const float* alpha,
                                            const float* A,
                                            rocblas_int lda,
                                            const float* B,
                                            rocblas_int ldb,
                                            const float* beta,
                                            float* C,
                                            rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm(rocblas_handle handle,
                                            rocblas_operation transa,
                                            rocblas_operation transb,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            const double* alpha,
                                            const double* A,
                                            rocblas_int lda,
                                            const double* B,
                                            rocblas_int ldb,
                                            const double* beta,
                                            double* C,
                                            rocblas_int ldc);

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_qgemm(
    rocblas_handle handle,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lda,
    const rocblas_half_complex *B, rocblas_int ldb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int ldc);
*/

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_cgemm(
    rocblas_handle handle,
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
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ldc);
*/

/***************************************************************************
 * batched
 * stride_a - "batch stride a": stride from the start of one "A" matrix to the next
 * stride_b
 * stride_c
 * batch_count - numbers of gemm's in the batch
 **************************************************************************/

/*! \brief BLAS Level 3 API

    \details
    xGEMM_STRIDED_BATCHED performs one of the strided batched matrix-matrix operations

        C[i*stride_c] = alpha*op( A[i*stride_a] )*op( B[i*stride_b] ) + beta*C[i*stride_c], for i in
   [0,batch_count-1]

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are strided batched matrices, with
    op( A ) an m by k by batch_count strided_batched matrix,
    op( B ) an k by n by batch_count strided_batched matrix and
    C an m by n by batch_count strided_batched matrix.

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
              matrix dimention m.
    @param[in]
    n         rocblas_int.
              matrix dimention n.
    @param[in]
    k         rocblas_int.
              matrix dimention k.
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing strided batched matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of "A".
    @param[in]
    stride_a       rocblas_int
              stride from the start of one "A" matrix to the next
    @param[in]
    B         pointer storing strided batched matrix B on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of "B".
    @param[in]
    stride_b       rocblas_int
              stride from the start of one "B" matrix to the next
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing strided batched matrix C on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of "C".
    @param[in]
    stride_c       rocblas_int
              stride from the start of one "C" matrix to the next
    @param[in]
    batch_count
              rocblas_int
              number of gemm operatons in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_hgemm_strided_batched(rocblas_handle handle,
                                                            rocblas_operation transa,
                                                            rocblas_operation transb,
                                                            rocblas_int m,
                                                            rocblas_int n,
                                                            rocblas_int k,
                                                            const rocblas_half* alpha,
                                                            const rocblas_half* A,
                                                            rocblas_int lda,
                                                            rocblas_int stride_a,
                                                            const rocblas_half* B,
                                                            rocblas_int ldb,
                                                            rocblas_int stride_b,
                                                            const rocblas_half* beta,
                                                            rocblas_half* C,
                                                            rocblas_int ldc,
                                                            rocblas_int stride_c,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle,
                                                            rocblas_operation transa,
                                                            rocblas_operation transb,
                                                            rocblas_int m,
                                                            rocblas_int n,
                                                            rocblas_int k,
                                                            const float* alpha,
                                                            const float* A,
                                                            rocblas_int lda,
                                                            rocblas_int stride_a,
                                                            const float* B,
                                                            rocblas_int ldb,
                                                            rocblas_int stride_b,
                                                            const float* beta,
                                                            float* C,
                                                            rocblas_int ldc,
                                                            rocblas_int stride_c,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle,
                                                            rocblas_operation transa,
                                                            rocblas_operation transb,
                                                            rocblas_int m,
                                                            rocblas_int n,
                                                            rocblas_int k,
                                                            const double* alpha,
                                                            const double* A,
                                                            rocblas_int lda,
                                                            rocblas_int stride_a,
                                                            const double* B,
                                                            rocblas_int ldb,
                                                            rocblas_int stride_b,
                                                            const double* beta,
                                                            double* C,
                                                            rocblas_int ldc,
                                                            rocblas_int stride_c,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_hgemm_kernel_name(rocblas_handle handle,
                                                        rocblas_operation transa,
                                                        rocblas_operation transb,
                                                        rocblas_int m,
                                                        rocblas_int n,
                                                        rocblas_int k,
                                                        const rocblas_half* alpha,
                                                        const rocblas_half* A,
                                                        rocblas_int lda,
                                                        rocblas_int stride_a,
                                                        const rocblas_half* B,
                                                        rocblas_int ldb,
                                                        rocblas_int stride_b,
                                                        const rocblas_half* beta,
                                                        rocblas_half* C,
                                                        rocblas_int ldc,
                                                        rocblas_int stride_c,
                                                        rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm_kernel_name(rocblas_handle handle,
                                                        rocblas_operation transa,
                                                        rocblas_operation transb,
                                                        rocblas_int m,
                                                        rocblas_int n,
                                                        rocblas_int k,
                                                        const float* alpha,
                                                        const float* A,
                                                        rocblas_int lda,
                                                        rocblas_int stride_a,
                                                        const float* B,
                                                        rocblas_int ldb,
                                                        rocblas_int stride_b,
                                                        const float* beta,
                                                        float* C,
                                                        rocblas_int ldc,
                                                        rocblas_int stride_c,
                                                        rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm_kernel_name(rocblas_handle handle,
                                                        rocblas_operation transa,
                                                        rocblas_operation transb,
                                                        rocblas_int m,
                                                        rocblas_int n,
                                                        rocblas_int k,
                                                        const double* alpha,
                                                        const double* A,
                                                        rocblas_int lda,
                                                        rocblas_int stride_a,
                                                        const double* B,
                                                        rocblas_int ldb,
                                                        rocblas_int stride_b,
                                                        const double* beta,
                                                        double* C,
                                                        rocblas_int ldc,
                                                        rocblas_int stride_c,
                                                        rocblas_int batch_count);

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_qgemm_strided_batched(
    rocblas_handle handle,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lda, rocblas_int stride_a,
    const rocblas_half_complex *B, rocblas_int ldb, rocblas_int stride_b,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int ldc, rocblas_int stride_c,
    rocblas_int batch_count );
*/

/* not implemented
ROCBLAS_EXPORT rocblas_status
rocblas_cgemm_strided_batched(
    rocblas_handle handle,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda, rocblas_int stride_a,
    const rocblas_float_complex *B, rocblas_int ldb, rocblas_int stride_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ldc, rocblas_int stride_c,
    rocblas_int batch_count );

ROCBLAS_EXPORT rocblas_status
rocblas_zgemm_strided_batched(
    rocblas_handle handle,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_int stride_a,
    const rocblas_double_complex *B, rocblas_int ldb, rocblas_int stride_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ldc, rocblas_int stride_c,
    rocblas_int batch_count );
*/

/*! \brief BLAS Level 3 API

    \details
    xGEAM performs one of the matrix-matrix operations

        C = alpha*op( A ) + beta*op( B ),

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.

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
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    beta      specifies the scalar beta.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in, out]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_sgeam(rocblas_handle handle,
                                            rocblas_operation transa,
                                            rocblas_operation transb,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const float* alpha,
                                            const float* A,
                                            rocblas_int lda,
                                            const float* beta,
                                            const float* B,
                                            rocblas_int ldb,
                                            float* C,
                                            rocblas_int ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dgeam(rocblas_handle handle,
                                            rocblas_operation transa,
                                            rocblas_operation transb,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const double* alpha,
                                            const double* A,
                                            rocblas_int lda,
                                            const double* beta,
                                            const double* B,
                                            rocblas_int ldb,
                                            double* C,
                                            rocblas_int ldc);

/*
 * ===========================================================================
 *    BLAS extensions
 * ===========================================================================
 */

/*! \brief BLAS EX API

    \details
    GEMM_EX performs one of the matrix-matrix operations

        D = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C and D are m by n matrices.

    Supported types are as follows:
        - rocblas_datatype_f64_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f32_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r = c_type = d_type =
   compute_type

    Below are restrictions for rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r =
   c_type = d_type = compute_type:
        - k must be a multiple of 4
        - lda must be a multiple of 4 if transA == rocblas_operation_transpose
        - ldb must be a multiple of 4 if transB == rocblas_operation_none
        - for transA == rocblas_operation_transpose or transB == rocblas_operation_none the matrices
   A and B must
          have each 4 consecutive values in the k dimension packed. This packing can be achieved
   with the following
          pseudo-code. The code assumes the original matrices are in A and B, and the packed
   matrices are A_packed
          and B_packed. The size of the A_packed matrix is the same as the size of the A matrix, and
   the size of
          the B_packed matrix is the same as the size of the B matrix.

    @code
    if(trans_a == rocblas_operation_none)
    {
        int nb = 4;
        for(int i_m = 0; i_m < m; i_m++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                A_packed[i_k % nb + (i_m + (i_k / nb) * lda) * nb] = A[i_m + i_k * lda];
            }
        }
    }
    else
    {
        A_packed = A;
    }
    if(trans_b == rocblas_operation_transpose)
    {
        int nb = 4;
        for(int i_n = 0; i_n < m; i_n++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                B_packed[i_k % nb + (i_n + (i_k / nb) * lda) * nb] = B[i_n + i_k * lda];
            }
        }
    }
    else
    {
        B_packed = B;
    }
    @endcode

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_operation.
              specifies the form of op( A ).
    @param[in]
    transB    rocblas_operation
              specifies the form of op( B ).
    @param[in]
    m         rocblas_int.
              matrix dimension m.
    @param[in]
    n         rocblas_int.
              matrix dimension n.
    @param[in]
    k         rocblas_int.
              matrix dimension k.
    @param[in]
    alpha     const void *.
              specifies the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         void *.
              pointer storing matrix A on the GPU.
    @param[in]
    a_type    rocblas_datatype.
              specifies the datatype of matrix A.
    @param[in]
    lda       rocblas_int.
              specifies the leading dimension of A.
    @param[in]
    b         void *.
              pointer storing matrix B on the GPU.
    @param[in]
    b_type    rocblas_datatype.
              specifies the datatype of matrix B.
    @param[in]
    ldb       rocblas_int.
              specifies the leading dimension of B.
    @param[in]
    beta      const void *.
              specifies the scalar beta. Same datatype as compute_type.
    @param[in]
    c         void *.
              pointer storing matrix C on the GPU.
    @param[in]
    c_type    rocblas_datatype.
              specifies the datatype of matrix C.
    @param[in]
    ldc       rocblas_int.
              specifies the leading dimension of C.
    @param[out]
    d         void *.
              pointer storing matrix D on the GPU.
    @param[in]
    d_type    rocblas_datatype.
              specifies the datatype of matrix D.
    @param[in]
    ldd       rocblas_int.
              specifies the leading dimension of D.
    @param[in]
    compute_type
              rocblas_datatype.
              specifies the datatype of computation.
    @param[in]
    algo      rocblas_gemm_algo.
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              int32_t.
              reserved for future use.
    @param[in]
    flags     uint32_t.
              reserved for future use.
    @param[in, out]
    workspace_size
              size_t *.
              size of workspace.
    @parm[in]
    workspace void *.
              workspace.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                              rocblas_operation trans_a,
                                              rocblas_operation trans_b,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              const void* alpha,
                                              const void* a,
                                              rocblas_datatype a_type,
                                              rocblas_int lda,
                                              const void* b,
                                              rocblas_datatype b_type,
                                              rocblas_int ldb,
                                              const void* beta,
                                              const void* c,
                                              rocblas_datatype c_type,
                                              rocblas_int ldc,
                                              void* d,
                                              rocblas_datatype d_type,
                                              rocblas_int ldd,
                                              rocblas_datatype compute_type,
                                              rocblas_gemm_algo algo,
                                              int32_t solution_index,
                                              uint32_t flags,
                                              size_t* workspace_size,
                                              void* workspace);

/*! \brief BLAS EX API

    \details
    GEMM_STRIDED_BATCHED_EX performs one of the strided_batched matrix-matrix operations

        D[i*stride_d] = alpha*op(A[i*stride_a])*op(B[i*stride_b]) + beta*C[i*stride_c], for i in
   [0,batch_count-1]

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are strided_batched matrices, with
    op( A ) an m by k by batch_count strided_batched matrix,
    op( B ) a k by n by batch_count strided_batched matrix and
    C and D are m by n by batch_count strided_batched matrices.

    The strided_batched matrices are multiple matrices separated by a constant stride.
    The number of matrices is batch_count.

    Supported types are as follows:
        - rocblas_datatype_f64_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f32_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r = c_type = d_type =
   compute_type

    Below are restrictions for rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r =
   c_type = d_type = compute_type:
        - k must be a multiple of 4
        - lda must be a multiple of 4 if transA == rocblas_operation_transpose
        - ldb must be a multiple of 4 if transB == rocblas_operation_none
        - for transA == rocblas_operation_transpose or transB == rocblas_operation_none the matrices
   A and B must
          have each 4 consecutive values in the k dimension packed. This packing can be achieved
   with the following
          pseudo-code. The code assumes the original matrices are in A and B, and the packed
   matrices are A_packed
          and B_packed. The size of the A_packed matrix is the same as the size of the A matrix, and
   the size of
          the B_packed matrix is the same as the size of the B matrix.

    @code
    if(trans_a == rocblas_operation_none)
    {
        int nb = 4;
        for(int i_m = 0; i_m < m; i_m++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                A_packed[i_k % nb + (i_m + (i_k / nb) * lda) * nb] = A[i_m + i_k * lda];
            }
        }
    }
    else
    {
        A_packed = A;
    }
    if(trans_b == rocblas_operation_transpose)
    {
        int nb = 4;
        for(int i_n = 0; i_n < m; i_n++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                B_packed[i_k % nb + (i_n + (i_k / nb) * lda) * nb] = B[i_n + i_k * lda];
            }
        }
    }
    else
    {
        B_packed = B;
    }
    @endcode

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_operation.
              specifies the form of op( A ).
    @param[in]
    transB    rocblas_operation.
              specifies the form of op( B ).
    @param[in]
    m         rocblas_int.
              matrix dimension m.
    @param[in]
    n         rocblas_int.
              matrix dimension n.
    @param[in]
    k         rocblas_int.
              matrix dimension k.
    @param[in]
    alpha     const void *.
              specifies the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         void *.
              pointer storing matrix A on the GPU.
    @param[in]
    a_type    rocblas_datatype.
              specifies the datatype of matrix A.
    @param[in]
    lda       rocblas_int.
              specifies the leading dimension of A.
    @param[in]
    stride_a  rocblas_long.
              specifies stride from start of one "A" matrix to the next.
    @param[in]
    b         void *.
              pointer storing matrix B on the GPU.
    @param[in]
    b_type    rocblas_datatype.
              specifies the datatype of matrix B.
    @param[in]
    ldb       rocblas_int.
              specifies the leading dimension of B.
    @param[in]
    stride_b  rocblas_long.
              specifies stride from start of one "B" matrix to the next.
    @param[in]
    beta      const void *.
              specifies the scalar beta. Same datatype as compute_type.
    @param[in]
    c         void *.
              pointer storing matrix C on the GPU.
    @param[in]
    c_type    rocblas_datatype.
              specifies the datatype of matrix C.
    @param[in]
    ldc       rocblas_int.
              specifies the leading dimension of C.
    @param[in]
    stride_c  rocblas_long.
              specifies stride from start of one "C" matrix to the next.
    @param[out]
    d         void *.
              pointer storing matrix D on the GPU.
    @param[in]
    d_type    rocblas_datatype.
              specifies the datatype of matrix D.
    @param[in]
    ldd       rocblas_int.
              specifies the leading dimension of D.
    @param[in]
    stride_d  rocblas_long.
              specifies stride from start of one "D" matrix to the next.
    @param[in]
    batch_count
              rocblas_int.
              number of gemm operations in the batch.
    @param[in]
    compute_type
              rocblas_datatype.
              specifies the datatype of computation.
    @param[in]
    algo      rocblas_gemm_algo.
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              int32_t.
              reserved for future use.
    @param[in]
    flags     uint32_t.
              reserved for future use.
    @param[in, out]
    workspace_size
              size_t *.
              size of workspace.
    @parm[in]
    workspace void *.
              workspace.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle handle,
                                                              rocblas_operation trans_a,
                                                              rocblas_operation trans_b,
                                                              rocblas_int m,
                                                              rocblas_int n,
                                                              rocblas_int k,
                                                              const void* alpha,
                                                              const void* a,
                                                              rocblas_datatype a_type,
                                                              rocblas_int lda,
                                                              rocblas_long stride_a,
                                                              const void* b,
                                                              rocblas_datatype b_type,
                                                              rocblas_int ldb,
                                                              rocblas_long stride_b,
                                                              const void* beta,
                                                              const void* c,
                                                              rocblas_datatype c_type,
                                                              rocblas_int ldc,
                                                              rocblas_long stride_c,
                                                              void* d,
                                                              rocblas_datatype d_type,
                                                              rocblas_int ldd,
                                                              rocblas_long stride_d,
                                                              rocblas_int batch_count,
                                                              rocblas_datatype compute_type,
                                                              rocblas_gemm_algo algo,
                                                              int32_t solution_index,
                                                              uint32_t flags,
                                                              size_t* workspace_size,
                                                              void* workspace);

/*! BLAS EX API

    \details
    TRSM_EX solves

        op(A)*X = alpha*B or X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices,
    A is triangular matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    The matrix X is overwritten on B.

    TRSM_EX gives the user the ability to manage device memory and exposes the invA matrix to be
    reused between runs.
    Before trsm_ex can be used the user must setup the invA matrix and x_temp_workspace.

    Setting up invA:
    The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
    matrix A, followed by any smaller diagonal block that remains.
    To set up invA it is recommended that rocblas_trtri_batched be used with matrix A as the input.

    Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
    rocblas_side_left and is n when rocblas_side_right.

    To begin, rocblas_trtri_batched must be called on the full 128x128 sized diagonal blocks of
    matrix A. Below are the restricted parameters:
      - n = 128
      - ldinvA = 128
      - stride_invA = 128x128
      - batch_count = k / 128,

    Then any remaining block may be added:
      - n = k % 128
      - invA = invA + stride_invA * previous_batch_count
      - ldinvA = 128
      - batch_count = 1

    Setting up x_temp_workspace:
    When x_temp_workspace is a nullptr the API enters a setup mode to recommend the size needed for
    temporary memory to be stored. The suggested size depends on the rocblas_trsm_option specified
    and is stored in x_temp_size. Once x_temp_workspace has been assigned to sufficient device
    memory, the API may be called again. This time x_temp_size must specify the size of temporary
    device memory allocated.

    @param[in]
    handle  rocblas_handle.
            handle to the rocblas library context queue.

    @param[in]
    side    rocblas_side.
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a lower triangular matrix.

    @param[in]
    transA  rocblas_operation.
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       rocblas_int.
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       rocblas_int.
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            &zero then A is not referenced, and B need not be set before
            entry.

    @param[in]
    A       void *
            pointer storing matrix A on the GPU.
            of dimension ( lda, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     rocblas_int.
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in, out]
    B       void *
            pointer storing matrix B on the GPU.
            B is of dimension ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    ldb    rocblas_int.
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    @param[in]
    invA    void *
            pointer storing the inverse diagonal blocks of A on the GPU.
            invA is of dimension ( ld_invA, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right.
            ld_invA must be equal to 128.

    @param[in]
    ld_invA rocblas_int.
            ldb specifies the first dimension of invA. ld_invA >= max( 1, BLOCK ).

    @param[in]
    compute_type rocblas_datatype
            specifies the datatype of computation

    @param[in]
    option  rocblas_trsm_option
            enumerant specifying the selected trsm memory option.
            -	rocblas_trsm_high_performance
            -	rocblas_trsm_low_memory
            Trsm can choose algorithms that either use large work memory size in order
            to get high performance, or small work memory with reduced performance.
            User can inspect returned work memory size to fit their application needs.
    @param[in, out]
    x_temp_size size_t*
            During setup the suggested size of x_temp is returned with respect
            to the selected rocblas_trsm_option.
            During run x_temp_size specifies the size allocated for
            x_temp_workspace
            Note: Must use rocblas_trsm_high_performance suggest size
            If rocblas_side_left and m is not a multiple of 128
            If rocblas_side_right and n is not a multiple of 128
    @parm[in]
    x_temp_workspace void*
            During setup x_temp_workspace must hold a null pointer to signal
            the request for x_temp_size
            During run x_temp_workspace is a pointer to store temporary matrix X
            on the GPU.
            x_temp_workspace is of dimension ( m, x_temp_size/m )

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_trsm_ex(rocblas_handle handle,
                                              rocblas_side side,
                                              rocblas_fill uplo,
                                              rocblas_operation trans_a,
                                              rocblas_diagonal diag,
                                              rocblas_int m,
                                              rocblas_int n,
                                              const void* alpha,
                                              const void* a,
                                              rocblas_int lda,
                                              void* b,
                                              rocblas_int ldb,
                                              const void* invA,
                                              rocblas_int ld_invA,
                                              rocblas_datatype compute_type,
                                              rocblas_trsm_option option,
                                              size_t* x_temp_size,
                                              void* x_temp_workspace);

/*
 * ===========================================================================
 *    build information
 * ===========================================================================
 */

/*! \brief   loads char* buf with the rocblas library version. size_t len
    is the maximum length of char* buf.
    \details

    @param[in, out]
    buf             pointer to buffer for version string

    @param[in]
    len             length of buf

 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_version_string(char* buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* _ROCBLAS_FUNCTIONS_H_ */
