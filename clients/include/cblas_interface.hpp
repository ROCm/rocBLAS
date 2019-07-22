/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _CBLAS_INTERFACE_
#define _CBLAS_INTERFACE_

#include "cblas.h"
#include "rocblas.h"

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
 */

extern "C" {
void strtri_(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri_(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
void ctrtri_(char* uplo, char* diag, int* n, rocblas_float_complex* A, int* lda, int* info);
void ztrtri_(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void spotrf_(char* uplo, int* m, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* m, double* A, int* lda, int* info);
void cpotrf_(char* uplo, int* m, rocblas_float_complex* A, int* lda, int* info);
void zpotrf_(char* uplo, int* m, rocblas_double_complex* A, int* lda, int* info);
}

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// iamax
template <typename T>
void cblas_iamax(rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

template <>
inline void cblas_iamax(rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result)
{
    *result = (rocblas_int)cblas_isamax(n, x, incx);
}

template <>
inline void cblas_iamax(rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result)
{
    *result = (rocblas_int)cblas_idamax(n, x, incx);
}

template <>
inline void cblas_iamax(rocblas_int                  n,
                        const rocblas_float_complex* x,
                        rocblas_int                  incx,
                        rocblas_int*                 result)
{
    *result = (rocblas_int)cblas_icamax(n, x, incx);
}

template <>
inline void cblas_iamax(rocblas_int                   n,
                        const rocblas_double_complex* x,
                        rocblas_int                   incx,
                        rocblas_int*                  result)
{
    *result = (rocblas_int)cblas_izamax(n, x, incx);
}

// asum
template <typename T1, typename T2>
void cblas_asum(rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <>
inline void cblas_asum(rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    *result = cblas_sasum(n, x, incx);
}

template <>
inline void cblas_asum(rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    *result = cblas_dasum(n, x, incx);
}

template <>
inline void
    cblas_asum(rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, float* result)
{
    *result = cblas_scasum(n, x, incx);
}

template <>
inline void
    cblas_asum(rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, double* result)
{
    *result = cblas_dzasum(n, x, incx);
}

// axpy
template <typename T>
void cblas_axpy(rocblas_int n, T alpha, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
inline void
    cblas_axpy(rocblas_int n, float alpha, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void cblas_axpy(
    rocblas_int n, double alpha, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void cblas_axpy(rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
inline void cblas_axpy(rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// copy
template <typename T>
void cblas_copy(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
inline void cblas_copy(rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    cblas_scopy(n, x, incx, y, incy);
}

template <>
inline void cblas_copy(rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
inline void cblas_copy(rocblas_int            n,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_ccopy(n, x, incx, y, incy);
}

template <>
inline void cblas_copy(rocblas_int             n,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zcopy(n, x, incx, y, incy);
}

// dot
template <typename T>
void cblas_dot(
    rocblas_int n, const T* x, rocblas_int incx, const T* y, rocblas_int incy, T* result);

template <>
inline void cblas_dot(rocblas_int  n,
                      const float* x,
                      rocblas_int  incx,
                      const float* y,
                      rocblas_int  incy,
                      float*       result)
{
    *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
inline void cblas_dot(rocblas_int   n,
                      const double* x,
                      rocblas_int   incx,
                      const double* y,
                      rocblas_int   incy,
                      double*       result)
{
    *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
inline void cblas_dot(rocblas_int                  n,
                      const rocblas_float_complex* x,
                      rocblas_int                  incx,
                      const rocblas_float_complex* y,
                      rocblas_int                  incy,
                      rocblas_float_complex*       result)
{
    cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
inline void cblas_dot(rocblas_int                   n,
                      const rocblas_double_complex* x,
                      rocblas_int                   incx,
                      const rocblas_double_complex* y,
                      rocblas_int                   incy,
                      rocblas_double_complex*       result)
{
    cblas_zdotu_sub(n, x, incx, y, incy, result);
}

// dotc
template <typename T>
void cblas_dotc(
    rocblas_int n, const T* x, rocblas_int incx, const T* y, rocblas_int incy, T* result);

template <>
inline void cblas_dotc(rocblas_int                  n,
                       const rocblas_float_complex* x,
                       rocblas_int                  incx,
                       const rocblas_float_complex* y,
                       rocblas_int                  incy,
                       rocblas_float_complex*       result)
{
    cblas_cdotc_sub(n, x, incx, y, incy, result);
}

template <>
inline void cblas_dotc(rocblas_int                   n,
                       const rocblas_double_complex* x,
                       rocblas_int                   incx,
                       const rocblas_double_complex* y,
                       rocblas_int                   incy,
                       rocblas_double_complex*       result)
{
    cblas_zdotc_sub(n, x, incx, y, incy, result);
}

// nrm2
template <typename T1, typename T2>
void cblas_nrm2(rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <>
inline void cblas_nrm2(rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    *result = cblas_snrm2(n, x, incx);
}

template <>
inline void cblas_nrm2(rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    *result = cblas_dnrm2(n, x, incx);
}

template <>
inline void
    cblas_nrm2(rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, float* result)
{
    *result = cblas_scnrm2(n, x, incx);
}

template <>
inline void
    cblas_nrm2(rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, double* result)
{
    *result = cblas_dznrm2(n, x, incx);
}

// scal
template <typename T, typename U>
inline void cblas_scal(rocblas_int n, U alpha, T* x, rocblas_int incx);

template <>
inline void cblas_scal(rocblas_int n, float alpha, float* x, rocblas_int incx)
{
    cblas_sscal(n, alpha, x, incx);
}

template <>
inline void cblas_scal(rocblas_int n, double alpha, double* x, rocblas_int incx)
{
    cblas_dscal(n, alpha, x, incx);
}

template <>
inline void cblas_scal(rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* x,
                       rocblas_int            incx)
{
    cblas_cscal(n, &alpha, x, incx);
}

template <>
inline void cblas_scal(rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* x,
                       rocblas_int             incx)
{
    cblas_zscal(n, &alpha, x, incx);
}

template <>
inline void cblas_scal(rocblas_int n, float alpha, rocblas_float_complex* x, rocblas_int incx)
{
    cblas_csscal(n, alpha, x, incx);
}

template <>
inline void cblas_scal(rocblas_int n, double alpha, rocblas_double_complex* x, rocblas_int incx)
{
    cblas_zdscal(n, alpha, x, incx);
}

// swap
template <typename T>
inline void cblas_swap(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
inline void cblas_swap(rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    cblas_sswap(n, x, incx, y, incy);
}

template <>
inline void cblas_swap(rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    cblas_dswap(n, x, incx, y, incy);
}

template <>
inline void cblas_swap(rocblas_int            n,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_cswap(n, x, incx, y, incy);
}

template <>
inline void cblas_swap(rocblas_int             n,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zswap(n, x, incx, y, incy);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gemv
template <typename T>
void cblas_gemv(rocblas_operation transA,
                rocblas_int       m,
                rocblas_int       n,
                T                 alpha,
                T*                A,
                rocblas_int       lda,
                T*                x,
                rocblas_int       incx,
                T                 beta,
                T*                y,
                rocblas_int       incy);

template <>
inline void cblas_gemv(rocblas_operation transA,
                       rocblas_int       m,
                       rocblas_int       n,
                       float             alpha,
                       float*            A,
                       rocblas_int       lda,
                       float*            x,
                       rocblas_int       incx,
                       float             beta,
                       float*            y,
                       rocblas_int       incy)
{
    cblas_sgemv(
        CblasColMajor, CBLAS_TRANSPOSE(transA), m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void cblas_gemv(rocblas_operation transA,
                       rocblas_int       m,
                       rocblas_int       n,
                       double            alpha,
                       double*           A,
                       rocblas_int       lda,
                       double*           x,
                       rocblas_int       incx,
                       double            beta,
                       double*           y,
                       rocblas_int       incy)
{
    cblas_dgemv(
        CblasColMajor, CBLAS_TRANSPOSE(transA), m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void cblas_gemv(rocblas_operation      transA,
                       rocblas_int            m,
                       rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* A,
                       rocblas_int            lda,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex  beta,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_cgemv(
        CblasColMajor, CBLAS_TRANSPOSE(transA), m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline void cblas_gemv(rocblas_operation       transA,
                       rocblas_int             m,
                       rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* A,
                       rocblas_int             lda,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex  beta,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zgemv(
        CblasColMajor, CBLAS_TRANSPOSE(transA), m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// trsv
template <typename T>
void cblas_trsv(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       m,
                const T*          A,
                rocblas_int       lda,
                T*                x,
                rocblas_int       incx);

template <>
inline void cblas_trsv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       const float*      A,
                       rocblas_int       lda,
                       float*            x,
                       rocblas_int       incx)
{
    cblas_strsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_trsv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       const double*     A,
                       rocblas_int       lda,
                       double*           x,
                       rocblas_int       incx)
{
    cblas_dtrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// trmv
template <typename T>
void cblas_trmv(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       m,
                const T*          A,
                rocblas_int       lda,
                T*                x,
                rocblas_int       incx);

template <>
inline void cblas_trmv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       const float*      A,
                       rocblas_int       lda,
                       float*            x,
                       rocblas_int       incx)
{
    cblas_strmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_trmv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       const double*     A,
                       rocblas_int       lda,
                       double*           x,
                       rocblas_int       incx)
{
    cblas_dtrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// symv
template <typename T>
void cblas_symv(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           A,
                rocblas_int  lda,
                T*           x,
                rocblas_int  incx,
                T            beta,
                T*           y,
                rocblas_int  incy);

template <>
inline void cblas_symv(rocblas_fill uplo,
                       rocblas_int  n,
                       float        alpha,
                       float*       A,
                       rocblas_int  lda,
                       float*       x,
                       rocblas_int  incx,
                       float        beta,
                       float*       y,
                       rocblas_int  incy)
{
    cblas_ssymv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void cblas_symv(rocblas_fill uplo,
                       rocblas_int  n,
                       double       alpha,
                       double*      A,
                       rocblas_int  lda,
                       double*      x,
                       rocblas_int  incx,
                       double       beta,
                       double*      y,
                       rocblas_int  incy)
{
    cblas_dsymv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
}

// ger
template <typename T>
void cblas_ger(rocblas_int m,
               rocblas_int n,
               T           alpha,
               T*          x,
               rocblas_int incx,
               T*          y,
               rocblas_int incy,
               T*          A,
               rocblas_int lda);

template <>
inline void cblas_ger(rocblas_int m,
                      rocblas_int n,
                      float       alpha,
                      float*      x,
                      rocblas_int incx,
                      float*      y,
                      rocblas_int incy,
                      float*      A,
                      rocblas_int lda)
{
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_ger(rocblas_int m,
                      rocblas_int n,
                      double      alpha,
                      double*     x,
                      rocblas_int incx,
                      double*     y,
                      rocblas_int incy,
                      double*     A,
                      rocblas_int lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

// syr
template <typename T>
void cblas_syr(
    rocblas_fill uplo, rocblas_int n, T alpha, T* x, rocblas_int incx, T* A, rocblas_int lda);

template <>
inline void cblas_syr(rocblas_fill uplo,
                      rocblas_int  n,
                      float        alpha,
                      float*       x,
                      rocblas_int  incx,
                      float*       A,
                      rocblas_int  lda)
{
    cblas_ssyr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

template <>
inline void cblas_syr(rocblas_fill uplo,
                      rocblas_int  n,
                      double       alpha,
                      double*      x,
                      rocblas_int  incx,
                      double*      A,
                      rocblas_int  lda)
{
    cblas_dsyr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

// hemv
template <typename T>
void cblas_hemv(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           A,
                rocblas_int  lda,
                T*           x,
                rocblas_int  incx,
                T            beta,
                T*           y,
                rocblas_int  incy);

template <>
inline void cblas_hemv(rocblas_fill           uplo,
                       rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* A,
                       rocblas_int            lda,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex  beta,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_chemv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline void cblas_hemv(rocblas_fill            uplo,
                       rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* A,
                       rocblas_int             lda,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex  beta,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zhemv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, lda, x, incx, &beta, y, incy);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// gemm
template <typename Ti, typename To, typename Tc>
void cblas_gemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                Tc                alpha,
                Ti*               A,
                rocblas_int       lda,
                Ti*               B,
                rocblas_int       ldb,
                Tc                beta,
                To*               C,
                rocblas_int       ldc);

template <>
inline void cblas_gemm(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int       m,
                       rocblas_int       n,
                       rocblas_int       k,
                       float             alpha,
                       float*            A,
                       rocblas_int       lda,
                       float*            B,
                       rocblas_int       ldb,
                       float             beta,
                       float*            C,
                       rocblas_int       ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                CBLAS_TRANSPOSE(transB),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
inline void cblas_gemm(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int       m,
                       rocblas_int       n,
                       rocblas_int       k,
                       double            alpha,
                       float*            A,
                       rocblas_int       lda,
                       float*            B,
                       rocblas_int       ldb,
                       double            beta,
                       float*            C,
                       rocblas_int       ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                CBLAS_TRANSPOSE(transB),
                m,
                n,
                k,
                float(alpha),
                A,
                lda,
                B,
                ldb,
                float(beta),
                C,
                ldc);
}

template <>
inline void cblas_gemm(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int       m,
                       rocblas_int       n,
                       rocblas_int       k,
                       double            alpha,
                       double*           A,
                       rocblas_int       lda,
                       double*           B,
                       rocblas_int       ldb,
                       double            beta,
                       double*           C,
                       rocblas_int       ldc)
{
    cblas_dgemm(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                CBLAS_TRANSPOSE(transB),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
inline void cblas_gemm(rocblas_operation      transA,
                       rocblas_operation      transB,
                       rocblas_int            m,
                       rocblas_int            n,
                       rocblas_int            k,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* A,
                       rocblas_int            lda,
                       rocblas_float_complex* B,
                       rocblas_int            ldb,
                       rocblas_float_complex  beta,
                       rocblas_float_complex* C,
                       rocblas_int            ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_cgemm(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                CBLAS_TRANSPOSE(transB),
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
inline void cblas_gemm(rocblas_operation       transA,
                       rocblas_operation       transB,
                       rocblas_int             m,
                       rocblas_int             n,
                       rocblas_int             k,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* A,
                       rocblas_int             lda,
                       rocblas_double_complex* B,
                       rocblas_int             ldb,
                       rocblas_double_complex  beta,
                       rocblas_double_complex* C,
                       rocblas_int             ldc)
{
    cblas_zgemm(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                CBLAS_TRANSPOSE(transB),
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// trsm
template <typename T>
void cblas_trsm(rocblas_side      side,
                rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       m,
                rocblas_int       n,
                T                 alpha,
                const T*          A,
                rocblas_int       lda,
                T*                B,
                rocblas_int       ldb);

template <>
inline void cblas_trsm(rocblas_side      side,
                       rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       rocblas_int       n,
                       float             alpha,
                       const float*      A,
                       rocblas_int       lda,
                       float*            B,
                       rocblas_int       ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strsm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
inline void cblas_trsm(rocblas_side      side,
                       rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       rocblas_int       n,
                       double            alpha,
                       const double*     A,
                       rocblas_int       lda,
                       double*           B,
                       rocblas_int       ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrsm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
inline void cblas_trsm(rocblas_side                 side,
                       rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  m,
                       rocblas_int                  n,
                       rocblas_float_complex        alpha,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       rocblas_float_complex*       B,
                       rocblas_int                  ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ctrsm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
inline void cblas_trsm(rocblas_side                  side,
                       rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   m,
                       rocblas_int                   n,
                       rocblas_double_complex        alpha,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       rocblas_double_complex*       B,
                       rocblas_int                   ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ztrsm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

// trtri
template <typename T>
rocblas_int cblas_trtri(char uplo, char diag, rocblas_int n, T* A, rocblas_int lda);

template <>
inline rocblas_int cblas_trtri(char uplo, char diag, rocblas_int n, float* A, rocblas_int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    rocblas_int info;
    strtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

template <>
inline rocblas_int cblas_trtri(char uplo, char diag, rocblas_int n, double* A, rocblas_int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    rocblas_int info;
    dtrtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

// trmm
template <typename T>
void cblas_trmm(rocblas_side      side,
                rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       m,
                rocblas_int       n,
                T                 alpha,
                const T*          A,
                rocblas_int       lda,
                T*                B,
                rocblas_int       ldb);

template <>
inline void cblas_trmm(rocblas_side      side,
                       rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       rocblas_int       n,
                       float             alpha,
                       const float*      A,
                       rocblas_int       lda,
                       float*            B,
                       rocblas_int       ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strmm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
inline void cblas_trmm(rocblas_side      side,
                       rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       rocblas_int       n,
                       double            alpha,
                       const double*     A,
                       rocblas_int       lda,
                       double*           B,
                       rocblas_int       ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrmm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
inline void cblas_trmm(rocblas_side                 side,
                       rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  m,
                       rocblas_int                  n,
                       rocblas_float_complex        alpha,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       rocblas_float_complex*       B,
                       rocblas_int                  ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ctrmm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
inline void cblas_trmm(rocblas_side                  side,
                       rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   m,
                       rocblas_int                   n,
                       rocblas_double_complex        alpha,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       rocblas_double_complex*       B,
                       rocblas_int                   ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ztrmm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

// getrf
template <typename T>
rocblas_int cblas_getrf(rocblas_int m, rocblas_int n, T* A, rocblas_int lda, rocblas_int* ipiv);

template <>
inline rocblas_int
    cblas_getrf(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
inline rocblas_int
    cblas_getrf(rocblas_int m, rocblas_int n, double* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
inline rocblas_int cblas_getrf(
    rocblas_int m, rocblas_int n, rocblas_float_complex* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    cgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
inline rocblas_int cblas_getrf(
    rocblas_int m, rocblas_int n, rocblas_double_complex* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    zgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// potrf
template <typename T>
rocblas_int cblas_potrf(char uplo, rocblas_int m, T* A, rocblas_int lda);

template <>
inline rocblas_int cblas_potrf(char uplo, rocblas_int m, float* A, rocblas_int lda)
{
    rocblas_int info;
    spotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
inline rocblas_int cblas_potrf(char uplo, rocblas_int m, double* A, rocblas_int lda)
{
    rocblas_int info;
    dpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
inline rocblas_int cblas_potrf(char uplo, rocblas_int m, rocblas_float_complex* A, rocblas_int lda)
{
    rocblas_int info;
    cpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
inline rocblas_int cblas_potrf(char uplo, rocblas_int m, rocblas_double_complex* A, rocblas_int lda)
{
    rocblas_int info;
    zpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

/* ============================================================================================ */

#endif /* _CBLAS_INTERFACE_ */
