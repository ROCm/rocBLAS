/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once
#ifndef _CBLAS_INTERFACE_
#define _CBLAS_INTERFACE_

#include "rocblas.h"

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
 */

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */
template <typename T>
void cblas_iamax(rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

template <typename T>
void cblas_iamin(rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

template <typename T1, typename T2>
void cblas_asum(rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <typename T>
void cblas_axpy(rocblas_int n, const T alpha, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <typename T>
void cblas_copy(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <typename T>
void cblas_dot(
    rocblas_int n, const T* x, rocblas_int incx, const T* y, rocblas_int incy, T* result);

template <typename T1, typename T2>
void cblas_nrm2(rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <typename T>
void cblas_scal(rocblas_int n, const T alpha, T* x, rocblas_int incx);

template <typename T>
void cblas_swap(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <typename T>
void cblas_gemv(rocblas_operation transA,
                rocblas_int m,
                rocblas_int n,
                T alpha,
                T* A,
                rocblas_int lda,
                T* x,
                rocblas_int incx,
                T beta,
                T* y,
                rocblas_int incy);

template <typename T>
void cblas_symv(rocblas_fill uplo,
                rocblas_int n,
                T alpha,
                T* A,
                rocblas_int lda,
                T* x,
                rocblas_int incx,
                T beta,
                T* y,
                rocblas_int incy);

template <typename T>
void cblas_ger(rocblas_int m,
               rocblas_int n,
               T alpha,
               T* x,
               rocblas_int incx,
               T* y,
               rocblas_int incy,
               T* A,
               rocblas_int lda);

template <typename T>
void cblas_syr(
    rocblas_fill uplo, rocblas_int n, T alpha, T* x, rocblas_int incx, T* A, rocblas_int lda);

template <typename T>
void cblas_hemv(rocblas_fill uplo,
                rocblas_int n,
                T alpha,
                T* A,
                rocblas_int lda,
                T* x,
                rocblas_int incx,
                T beta,
                T* y,
                rocblas_int incy);

template <typename T>
void cblas_gemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int m,
                rocblas_int n,
                rocblas_int k,
                T alpha,
                T* A,
                rocblas_int lda,
                T* B,
                rocblas_int ldb,
                T beta,
                T* C,
                rocblas_int ldc);

template <typename T>
void cblas_trsm(rocblas_side side,
                rocblas_fill uplo,
                rocblas_operation transA,
                rocblas_diagonal diag,
                rocblas_int m,
                rocblas_int n,
                T alpha,
                const T* A,
                rocblas_int lda,
                T* B,
                rocblas_int ldb);

template <typename T>
rocblas_int cblas_potf2(rocblas_fill uplo, rocblas_int n, T* A, rocblas_int lda);

template <typename T>
rocblas_int cblas_trtri(char uplo, char diag, rocblas_int n, T* A, rocblas_int lda);

template <typename T>
void cblas_trmm(rocblas_side side,
                rocblas_fill uplo,
                rocblas_operation transA,
                rocblas_diagonal diag,
                rocblas_int m,
                rocblas_int n,
                T alpha,
                const T* A,
                rocblas_int lda,
                T* B,
                rocblas_int ldb);

template <typename T>
rocblas_int cblas_getrf(rocblas_int m, rocblas_int n, T* A, rocblas_int lda, rocblas_int* ipiv);

template <typename T>
rocblas_int cblas_potrf(char uplo, rocblas_int m, T* A, rocblas_int lda);
/* ============================================================================================ */

#endif /* _CBLAS_INTERFACE_ */
