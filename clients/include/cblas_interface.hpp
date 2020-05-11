/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _CBLAS_INTERFACE_
#define _CBLAS_INTERFACE_

#include "cblas.h"
#include "rocblas.h"
#include "rocblas.hpp"
#include <type_traits>

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
template <typename T>
void cblas_asum(rocblas_int n, const T* x, rocblas_int incx, real_t<T>* result);

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
template <typename T>
void cblas_nrm2(rocblas_int n, const T* x, rocblas_int incx, real_t<T>* result);

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

// rot

// LAPACK fortran library functionality
extern "C" {
void crot_(const int*                   n,
           rocblas_float_complex*       cx,
           const int*                   incx,
           rocblas_float_complex*       cy,
           const int*                   incy,
           const float*                 c,
           const rocblas_float_complex* s);
void csrot_(const int*             n,
            rocblas_float_complex* cx,
            const int*             incx,
            rocblas_float_complex* cy,
            const int*             incy,
            const float*           c,
            const float*           s);
void zrot_(const int*                    n,
           rocblas_double_complex*       cx,
           const int*                    incx,
           rocblas_double_complex*       cy,
           const int*                    incy,
           const double*                 c,
           const rocblas_double_complex* s);
void zdrot_(const int*              n,
            rocblas_double_complex* cx,
            const int*              incx,
            rocblas_double_complex* cy,
            const int*              incy,
            const double*           c,
            const double*           s);
}

template <typename T, typename U, typename V>
inline void cblas_rot(
    rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy, const U* c, const V* s);

template <>
inline void cblas_rot(rocblas_int  n,
                      float*       x,
                      rocblas_int  incx,
                      float*       y,
                      rocblas_int  incy,
                      const float* c,
                      const float* s)
{
    cblas_srot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void cblas_rot(rocblas_int   n,
                      double*       x,
                      rocblas_int   incx,
                      double*       y,
                      rocblas_int   incy,
                      const double* c,
                      const double* s)
{
    cblas_drot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void cblas_rot(rocblas_int                  n,
                      rocblas_float_complex*       x,
                      rocblas_int                  incx,
                      rocblas_float_complex*       y,
                      rocblas_int                  incy,
                      const float*                 c,
                      const rocblas_float_complex* s)
{
    crot_(&n, x, &incx, y, &incx, c, s);
}

template <>
inline void cblas_rot(rocblas_int            n,
                      rocblas_float_complex* x,
                      rocblas_int            incx,
                      rocblas_float_complex* y,
                      rocblas_int            incy,
                      const float*           c,
                      const float*           s)
{
    csrot_(&n, x, &incx, y, &incy, c, s);
}

template <>
inline void cblas_rot(rocblas_int                   n,
                      rocblas_double_complex*       x,
                      rocblas_int                   incx,
                      rocblas_double_complex*       y,
                      rocblas_int                   incy,
                      const double*                 c,
                      const rocblas_double_complex* s)
{
    zrot_(&n, x, &incx, y, &incy, c, s);
}

template <>
inline void cblas_rot(rocblas_int             n,
                      rocblas_double_complex* x,
                      rocblas_int             incx,
                      rocblas_double_complex* y,
                      rocblas_int             incy,
                      const double*           c,
                      const double*           s)
{
    zdrot_(&n, x, &incx, y, &incy, c, s);
}

// rotg

// LAPACK fortran library functionality
extern "C" {
void crotg_(rocblas_float_complex* a, rocblas_float_complex* b, float* c, rocblas_float_complex* s);
void zrotg_(rocblas_double_complex* a,
            rocblas_double_complex* b,
            double*                 c,
            rocblas_double_complex* s);
}

template <typename T, typename U>
inline void cblas_rotg(T* a, T* b, U* c, T* s);

template <>
inline void cblas_rotg(float* a, float* b, float* c, float* s)
{
    cblas_srotg(a, b, c, s);
}

template <>
inline void cblas_rotg(double* a, double* b, double* c, double* s)
{
    cblas_drotg(a, b, c, s);
}

template <>
inline void cblas_rotg(rocblas_float_complex* a,
                       rocblas_float_complex* b,
                       float*                 c,
                       rocblas_float_complex* s)
{
    crotg_(a, b, c, s);
}

template <>
inline void cblas_rotg(rocblas_double_complex* a,
                       rocblas_double_complex* b,
                       double*                 c,
                       rocblas_double_complex* s)
{
    zrotg_(a, b, c, s);
}

// rotm

template <typename T>
inline void cblas_rotm(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy, const T* p);

template <>
inline void cblas_rotm(
    rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy, const float* p)
{
    cblas_srotm(n, x, incx, y, incy, p);
}

template <>
inline void cblas_rotm(
    rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy, const double* p)
{
    cblas_drotm(n, x, incx, y, incy, p);
}

// rotmg

template <typename T>
inline void cblas_rotmg(T* d1, T* d2, T* b1, const T* b2, T* p);

template <>
inline void cblas_rotmg(float* d1, float* d2, float* b1, const float* b2, float* p)
{
    cblas_srotmg(d1, d2, b1, *b2, p);
}

template <>
inline void cblas_rotmg(double* d1, double* d2, double* b1, const double* b2, double* p)
{
    cblas_drotmg(d1, d2, b1, *b2, p);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
template <typename T>
void cblas_gbmv(rocblas_operation transA,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       kl,
                rocblas_int       ku,
                T                 alpha,
                T*                A,
                rocblas_int       lda,
                T*                x,
                rocblas_int       incx,
                T                 beta,
                T*                y,
                rocblas_int       incy);

template <>
inline void cblas_gbmv(rocblas_operation transA,
                       rocblas_int       m,
                       rocblas_int       n,
                       rocblas_int       kl,
                       rocblas_int       ku,
                       float             alpha,
                       float*            A,
                       rocblas_int       lda,
                       float*            x,
                       rocblas_int       incx,
                       float             beta,
                       float*            y,
                       rocblas_int       incy)
{
    cblas_sgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
inline void cblas_gbmv(rocblas_operation transA,
                       rocblas_int       m,
                       rocblas_int       n,
                       rocblas_int       kl,
                       rocblas_int       ku,
                       double            alpha,
                       double*           A,
                       rocblas_int       lda,
                       double*           x,
                       rocblas_int       incx,
                       double            beta,
                       double*           y,
                       rocblas_int       incy)
{
    cblas_dgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
inline void cblas_gbmv(rocblas_operation      transA,
                       rocblas_int            m,
                       rocblas_int            n,
                       rocblas_int            kl,
                       rocblas_int            ku,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* A,
                       rocblas_int            lda,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex  beta,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_cgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

template <>
inline void cblas_gbmv(rocblas_operation       transA,
                       rocblas_int             m,
                       rocblas_int             n,
                       rocblas_int             kl,
                       rocblas_int             ku,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* A,
                       rocblas_int             lda,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex  beta,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

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

// tbmv
template <typename T>
void cblas_tbmv(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       m,
                rocblas_int       k,
                T*                A,
                rocblas_int       lda,
                T*                x,
                rocblas_int       incx);

template <>
inline void cblas_tbmv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       rocblas_int       k,
                       float*            A,
                       rocblas_int       lda,
                       float*            x,
                       rocblas_int       incx)
{
    cblas_stbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_tbmv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       rocblas_int       k,
                       double*           A,
                       rocblas_int       lda,
                       double*           x,
                       rocblas_int       incx)
{
    cblas_dtbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_tbmv(rocblas_fill           uplo,
                       rocblas_operation      transA,
                       rocblas_diagonal       diag,
                       rocblas_int            m,
                       rocblas_int            k,
                       rocblas_float_complex* A,
                       rocblas_int            lda,
                       rocblas_float_complex* x,
                       rocblas_int            incx)
{
    cblas_ctbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_tbmv(rocblas_fill            uplo,
                       rocblas_operation       transA,
                       rocblas_diagonal        diag,
                       rocblas_int             m,
                       rocblas_int             k,
                       rocblas_double_complex* A,
                       rocblas_int             lda,
                       rocblas_double_complex* x,
                       rocblas_int             incx)
{
    cblas_ztbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

// tbsv
template <typename T>
void cblas_tbsv(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       n,
                rocblas_int       k,
                const T*          A,
                rocblas_int       lda,
                T*                x,
                rocblas_int       incx);

template <>
inline void cblas_tbsv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       n,
                       rocblas_int       k,
                       const float*      A,
                       rocblas_int       lda,
                       float*            x,
                       rocblas_int       incx)
{
    cblas_stbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_tbsv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       n,
                       rocblas_int       k,
                       const double*     A,
                       rocblas_int       lda,
                       double*           x,
                       rocblas_int       incx)
{
    cblas_dtbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_tbsv(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  n,
                       rocblas_int                  k,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       rocblas_float_complex*       x,
                       rocblas_int                  incx)
{
    cblas_ctbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void cblas_tbsv(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   n,
                       rocblas_int                   k,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       rocblas_double_complex*       x,
                       rocblas_int                   incx)
{
    cblas_ztbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

// tpsv
template <typename T>
void cblas_tpsv(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       n,
                const T*          AP,
                T*                x,
                rocblas_int       incx);

template <>
inline void cblas_tpsv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       n,
                       const float*      AP,
                       float*            x,
                       rocblas_int       incx)
{
    cblas_stpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
inline void cblas_tpsv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       n,
                       const double*     AP,
                       double*           x,
                       rocblas_int       incx)
{
    cblas_dtpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
inline void cblas_tpsv(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  n,
                       const rocblas_float_complex* AP,
                       rocblas_float_complex*       x,
                       rocblas_int                  incx)
{
    cblas_ctpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
inline void cblas_tpsv(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   n,
                       const rocblas_double_complex* AP,
                       rocblas_double_complex*       x,
                       rocblas_int                   incx)
{
    cblas_ztpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
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

template <>
inline void cblas_trsv(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  m,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       rocblas_float_complex*       x,
                       rocblas_int                  incx)
{
    cblas_ctrsv(CblasColMajor,
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
inline void cblas_trsv(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   m,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       rocblas_double_complex*       x,
                       rocblas_int                   incx)
{
    cblas_ztrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// tpmv
template <typename T>
void cblas_tpmv(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_diagonal  diag,
                rocblas_int       m,
                const T*          A,
                T*                x,
                rocblas_int       incx);

template <>
inline void cblas_tpmv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       const float*      A,
                       float*            x,
                       rocblas_int       incx)
{
    cblas_stpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
inline void cblas_tpmv(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_diagonal  diag,
                       rocblas_int       m,
                       const double*     A,
                       double*           x,
                       rocblas_int       incx)
{
    cblas_dtpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
inline void cblas_tpmv(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  m,
                       const rocblas_float_complex* A,
                       rocblas_float_complex*       x,
                       rocblas_int                  incx)
{
    cblas_ctpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
inline void cblas_tpmv(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   m,
                       const rocblas_double_complex* A,
                       rocblas_double_complex*       x,
                       rocblas_int                   incx)
{
    cblas_ztpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
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

template <>
inline void cblas_trmv(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_diagonal             diag,
                       rocblas_int                  m,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       rocblas_float_complex*       x,
                       rocblas_int                  incx)
{
    cblas_ctrmv(CblasColMajor,
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
inline void cblas_trmv(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_diagonal              diag,
                       rocblas_int                   m,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       rocblas_double_complex*       x,
                       rocblas_int                   incx)
{
    cblas_ztrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// sbmv
template <typename T>
void cblas_sbmv(rocblas_fill uplo,
                rocblas_int  n,
                rocblas_int  k,
                T            alpha,
                T*           A,
                rocblas_int  lda,
                T*           x,
                rocblas_int  incx,
                T            beta,
                T*           y,
                rocblas_int  incy);

template <>
inline void cblas_sbmv(rocblas_fill uplo,
                       rocblas_int  n,
                       rocblas_int  k,
                       float        alpha,
                       float*       A,
                       rocblas_int  lda,
                       float*       x,
                       rocblas_int  incx,
                       float        beta,
                       float*       y,
                       rocblas_int  incy)
{
    cblas_ssbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void cblas_sbmv(rocblas_fill uplo,
                       rocblas_int  n,
                       rocblas_int  k,
                       double       alpha,
                       double*      A,
                       rocblas_int  lda,
                       double*      x,
                       rocblas_int  incx,
                       double       beta,
                       double*      y,
                       rocblas_int  incy)
{
    cblas_dsbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// spmv
template <typename T>
void cblas_spmv(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           A,
                T*           x,
                rocblas_int  incx,
                T            beta,
                T*           y,
                rocblas_int  incy);

template <>
inline void cblas_spmv(rocblas_fill uplo,
                       rocblas_int  n,
                       float        alpha,
                       float*       A,
                       float*       x,
                       rocblas_int  incx,
                       float        beta,
                       float*       y,
                       rocblas_int  incy)
{
    cblas_sspmv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, x, incx, beta, y, incy);
}

template <>
inline void cblas_spmv(rocblas_fill uplo,
                       rocblas_int  n,
                       double       alpha,
                       double*      A,
                       double*      x,
                       rocblas_int  incx,
                       double       beta,
                       double*      y,
                       rocblas_int  incy)
{
    cblas_dspmv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, x, incx, beta, y, incy);
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

// blis flame symbols
extern "C" {
void csymv_(char*                  uplo,
            int*                   n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int*                   lda,
            rocblas_float_complex* x,
            int*                   incx,
            rocblas_float_complex* beta,
            rocblas_float_complex* y,
            int*                   yinc);

void zsymv_(char*                   uplo,
            int*                    n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int*                    lda,
            rocblas_double_complex* x,
            int*                    incx,
            rocblas_double_complex* beta,
            rocblas_double_complex* y,
            int*                    yinc);
}

template <>
inline void cblas_symv(rocblas_fill           uplo,
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
    char u = uplo == rocblas_fill_upper ? 'U' : 'L';
    csymv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
inline void cblas_symv(rocblas_fill            uplo,
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
    char u = uplo == rocblas_fill_upper ? 'U' : 'L';
    zsymv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

// spr
extern "C" {
void cspr_(char*                  uplo,
           int*                   n,
           rocblas_float_complex* alpha,
           rocblas_float_complex* x,
           int*                   incx,
           rocblas_float_complex* A);

void zspr_(char*                   uplo,
           int*                    n,
           rocblas_double_complex* alpha,
           rocblas_double_complex* x,
           int*                    incx,
           rocblas_double_complex* A);
}

template <typename T>
void cblas_spr(rocblas_fill uplo, rocblas_int n, T alpha, T* x, rocblas_int incx, T* A);

template <>
inline void
    cblas_spr(rocblas_fill uplo, rocblas_int n, float alpha, float* x, rocblas_int incx, float* A)
{
    cblas_sspr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

template <>
inline void cblas_spr(
    rocblas_fill uplo, rocblas_int n, double alpha, double* x, rocblas_int incx, double* A)
{
    cblas_dspr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

template <>
inline void cblas_spr(rocblas_fill           uplo,
                      rocblas_int            n,
                      rocblas_float_complex  alpha,
                      rocblas_float_complex* x,
                      rocblas_int            incx,
                      rocblas_float_complex* A)
{
    char u = uplo == rocblas_fill_upper ? 'U' : 'L';
    cspr_(&u, &n, &alpha, x, &incx, A);
}

template <>
inline void cblas_spr(rocblas_fill            uplo,
                      rocblas_int             n,
                      rocblas_double_complex  alpha,
                      rocblas_double_complex* x,
                      rocblas_int             incx,
                      rocblas_double_complex* A)
{
    char u = uplo == rocblas_fill_upper ? 'U' : 'L';
    zspr_(&u, &n, &alpha, x, &incx, A);
}

// spr2
template <typename T>
void cblas_spr2(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           x,
                rocblas_int  incx,
                T*           y,
                rocblas_int  incy,
                T*           A);

template <>
inline void cblas_spr2(rocblas_fill uplo,
                       rocblas_int  n,
                       float        alpha,
                       float*       x,
                       rocblas_int  incx,
                       float*       y,
                       rocblas_int  incy,
                       float*       A)
{
    cblas_sspr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A);
}

template <>
inline void cblas_spr2(rocblas_fill uplo,
                       rocblas_int  n,
                       double       alpha,
                       double*      x,
                       rocblas_int  incx,
                       double*      y,
                       rocblas_int  incy,
                       double*      A)
{
    cblas_dspr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A);
}

// ger maps to ger, geru, gerc
template <typename T, bool CONJ>
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
inline void cblas_ger<float, false>(rocblas_int m,
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
inline void cblas_ger<double, false>(rocblas_int m,
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

template <>
inline void cblas_ger<rocblas_float_complex, false>(rocblas_int            m,
                                                    rocblas_int            n,
                                                    rocblas_float_complex  alpha,
                                                    rocblas_float_complex* x,
                                                    rocblas_int            incx,
                                                    rocblas_float_complex* y,
                                                    rocblas_int            incy,
                                                    rocblas_float_complex* A,
                                                    rocblas_int            lda)
{
    cblas_cgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_ger<rocblas_double_complex, false>(rocblas_int             m,
                                                     rocblas_int             n,
                                                     rocblas_double_complex  alpha,
                                                     rocblas_double_complex* x,
                                                     rocblas_int             incx,
                                                     rocblas_double_complex* y,
                                                     rocblas_int             incy,
                                                     rocblas_double_complex* A,
                                                     rocblas_int             lda)
{
    cblas_zgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_ger<rocblas_float_complex, true>(rocblas_int            m,
                                                   rocblas_int            n,
                                                   rocblas_float_complex  alpha,
                                                   rocblas_float_complex* x,
                                                   rocblas_int            incx,
                                                   rocblas_float_complex* y,
                                                   rocblas_int            incy,
                                                   rocblas_float_complex* A,
                                                   rocblas_int            lda)
{
    cblas_cgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_ger<rocblas_double_complex, true>(rocblas_int             m,
                                                    rocblas_int             n,
                                                    rocblas_double_complex  alpha,
                                                    rocblas_double_complex* x,
                                                    rocblas_int             incx,
                                                    rocblas_double_complex* y,
                                                    rocblas_int             incy,
                                                    rocblas_double_complex* A,
                                                    rocblas_int             lda)
{
    cblas_zgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

// syr
template <typename T>
inline void cblas_syr(
    rocblas_fill uplo, rocblas_int n, T alpha, T* xa, rocblas_int incx, T* A, rocblas_int lda);

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

// blis flame symbols
extern "C" {
void csyr_(char*                  uplo,
           int*                   n,
           rocblas_float_complex* alpha,
           rocblas_float_complex* x,
           int*                   incx,
           rocblas_float_complex* a,
           int*                   lda);
void zsyr_(char*                   uplo,
           int*                    n,
           rocblas_double_complex* alpha,
           rocblas_double_complex* x,
           int*                    incx,
           rocblas_double_complex* a,
           int*                    lda);
}

template <>
inline void cblas_syr(rocblas_fill           uplo,
                      rocblas_int            n,
                      rocblas_float_complex  alpha,
                      rocblas_float_complex* xa,
                      rocblas_int            incx,
                      rocblas_float_complex* A,
                      rocblas_int            lda)
{
    char u = uplo == rocblas_fill_upper ? 'U' : 'L';
    csyr_(&u, &n, &alpha, xa, &incx, A, &lda);
}

template <>
inline void cblas_syr(rocblas_fill            uplo,
                      rocblas_int             n,
                      rocblas_double_complex  alpha,
                      rocblas_double_complex* xa,
                      rocblas_int             incx,
                      rocblas_double_complex* A,
                      rocblas_int             lda)
{
    char u = uplo == rocblas_fill_upper ? 'U' : 'L';
    zsyr_(&u, &n, &alpha, xa, &incx, A, &lda);
}

/* working cpu template code in case flame symbols disappear
// cblas_syr doesn't have complex support so implementation below for float/double complex
template <typename T>
void cblas_syr(
    rocblas_fill uplo, rocblas_int n, T alpha, T* xa, rocblas_int incx, T* A, rocblas_int lda)
{
    if(n <= 0)
        return;

    T* x = (incx < 0) ? xa - ptrdiff_t(incx) * (n - 1) : xa;

    if(uplo == rocblas_fill_upper)
    {
        for(int j = 0; j < n; ++j)
        {
            T tmp = alpha * x[j * incx];
            for(int i = 0; i <= j; ++i)
            {
                A[i + j * lda] = A[i + j * lda] + x[i * incx] * tmp;
            }
        }
    }
    else
    {
        for(int j = 0; j < n; ++j)
        {
            T tmp = alpha * x[j * incx];
            for(int i = j; i < n; ++i)
            {
                A[i + j * lda] = A[i + j * lda] + x[i * incx] * tmp;
            }
        }
    }
}
*/

// syr2
template <typename T>
inline void cblas_syr2(rocblas_fill uplo,
                       rocblas_int  n,
                       T            alpha,
                       T*           x,
                       rocblas_int  incx,
                       T*           y,
                       rocblas_int  incy,
                       T*           A,
                       rocblas_int  lda);

template <>
inline void cblas_syr2(rocblas_fill uplo,
                       rocblas_int  n,
                       float        alpha,
                       float*       x,
                       rocblas_int  incx,
                       float*       y,
                       rocblas_int  incy,
                       float*       A,
                       rocblas_int  lda)
{
    cblas_ssyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_syr2(rocblas_fill uplo,
                       rocblas_int  n,
                       double       alpha,
                       double*      x,
                       rocblas_int  incx,
                       double*      y,
                       rocblas_int  incy,
                       double*      A,
                       rocblas_int  lda)
{
    cblas_dsyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

// No complex implementation of syr2, make a local version.
template <typename T>
inline void cblas_syr2_local(rocblas_fill uplo,
                             rocblas_int  n,
                             T            alpha,
                             T*           xa,
                             rocblas_int  incx,
                             T*           ya,
                             rocblas_int  incy,
                             T*           A,
                             rocblas_int  lda)
{
    if(n <= 0)
        return;

    T* x = (incx < 0) ? xa - ptrdiff_t(incx) * (n - 1) : xa;
    T* y = (incy < 0) ? ya - ptrdiff_t(incy) * (n - 1) : ya;

    if(uplo == rocblas_fill_upper)
    {
        for(int j = 0; j < n; ++j)
        {
            T tmpx = alpha * x[j * incx];
            T tmpy = alpha * y[j * incy];
            for(int i = 0; i <= j; ++i)
            {
                A[i + j * lda] = A[i + j * lda] + x[i * incx] * tmpy + y[i * incy] * tmpx;
            }
        }
    }
    else
    {
        for(int j = 0; j < n; ++j)
        {
            T tmpx = alpha * x[j * incx];
            T tmpy = alpha * y[j * incy];
            for(int i = j; i < n; ++i)
            {
                A[i + j * lda] = A[i + j * lda] + x[i * incx] * tmpy + y[i * incy] * tmpx;
            }
        }
    }
}

template <>
inline void cblas_syr2(rocblas_fill           uplo,
                       rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex* y,
                       rocblas_int            incy,
                       rocblas_float_complex* A,
                       rocblas_int            lda)
{
    cblas_syr2_local(uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_syr2(rocblas_fill            uplo,
                       rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex* y,
                       rocblas_int             incy,
                       rocblas_double_complex* A,
                       rocblas_int             lda)
{
    cblas_syr2_local(uplo, n, alpha, x, incx, y, incy, A, lda);
}

// hbmv
template <typename T>
void cblas_hbmv(rocblas_fill uplo,
                rocblas_int  n,
                rocblas_int  k,
                T            alpha,
                T*           A,
                rocblas_int  lda,
                T*           x,
                rocblas_int  incx,
                T            beta,
                T*           y,
                rocblas_int  incy);

template <>
inline void cblas_hbmv(rocblas_fill           uplo,
                       rocblas_int            n,
                       rocblas_int            k,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* A,
                       rocblas_int            lda,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex  beta,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_chbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline void cblas_hbmv(rocblas_fill            uplo,
                       rocblas_int             n,
                       rocblas_int             k,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* A,
                       rocblas_int             lda,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex  beta,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zhbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, &alpha, A, lda, x, incx, &beta, y, incy);
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

// her
template <typename T>
void cblas_her(rocblas_fill uplo,
               rocblas_int  n,
               real_t<T>    alpha,
               T*           x,
               rocblas_int  incx,
               T*           A,
               rocblas_int  lda);

template <>
inline void cblas_her(rocblas_fill           uplo,
                      rocblas_int            n,
                      float                  alpha,
                      rocblas_float_complex* x,
                      rocblas_int            incx,
                      rocblas_float_complex* A,
                      rocblas_int            lda)
{
    cblas_cher(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

template <>
inline void cblas_her(rocblas_fill            uplo,
                      rocblas_int             n,
                      double                  alpha,
                      rocblas_double_complex* x,
                      rocblas_int             incx,
                      rocblas_double_complex* A,
                      rocblas_int             lda)
{
    cblas_zher(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

// her2
template <typename T>
void cblas_her2(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           x,
                rocblas_int  incx,
                T*           y,
                rocblas_int  incy,
                T*           A,
                rocblas_int  lda);

template <>
inline void cblas_her2(rocblas_fill           uplo,
                       rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex* y,
                       rocblas_int            incy,
                       rocblas_float_complex* A,
                       rocblas_int            lda)
{
    cblas_cher2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void cblas_her2(rocblas_fill            uplo,
                       rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex* y,
                       rocblas_int             incy,
                       rocblas_double_complex* A,
                       rocblas_int             lda)
{
    cblas_zher2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A, lda);
}

// hpmv
template <typename T>
void cblas_hpmv(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           A,
                T*           x,
                rocblas_int  incx,
                T            beta,
                T*           y,
                rocblas_int  incy);

template <>
inline void cblas_hpmv(rocblas_fill           uplo,
                       rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* A,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex  beta,
                       rocblas_float_complex* y,
                       rocblas_int            incy)
{
    cblas_chpmv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, x, incx, &beta, y, incy);
}

template <>
inline void cblas_hpmv(rocblas_fill            uplo,
                       rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* A,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex  beta,
                       rocblas_double_complex* y,
                       rocblas_int             incy)
{
    cblas_zhpmv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, x, incx, &beta, y, incy);
}

// hpr
template <typename T>
void cblas_hpr(rocblas_fill uplo, rocblas_int n, real_t<T> alpha, T* x, rocblas_int incx, T* A);

template <>
inline void cblas_hpr(rocblas_fill           uplo,
                      rocblas_int            n,
                      float                  alpha,
                      rocblas_float_complex* x,
                      rocblas_int            incx,
                      rocblas_float_complex* A)
{
    cblas_chpr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

template <>
inline void cblas_hpr(rocblas_fill            uplo,
                      rocblas_int             n,
                      double                  alpha,
                      rocblas_double_complex* x,
                      rocblas_int             incx,
                      rocblas_double_complex* A)
{
    cblas_zhpr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

// hpr2
template <typename T>
void cblas_hpr2(rocblas_fill uplo,
                rocblas_int  n,
                T            alpha,
                T*           x,
                rocblas_int  incx,
                T*           y,
                rocblas_int  incy,
                T*           A);

template <>
inline void cblas_hpr2(rocblas_fill           uplo,
                       rocblas_int            n,
                       rocblas_float_complex  alpha,
                       rocblas_float_complex* x,
                       rocblas_int            incx,
                       rocblas_float_complex* y,
                       rocblas_int            incy,
                       rocblas_float_complex* A)
{
    cblas_chpr2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A);
}

template <>
inline void cblas_hpr2(rocblas_fill            uplo,
                       rocblas_int             n,
                       rocblas_double_complex  alpha,
                       rocblas_double_complex* x,
                       rocblas_int             incx,
                       rocblas_double_complex* y,
                       rocblas_int             incy,
                       rocblas_double_complex* A)
{
    cblas_zhpr2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// geam
template <typename T>
void cblas_geam(rocblas_operation transa,
                rocblas_operation transb,
                rocblas_int       m,
                rocblas_int       n,
                T*                alpha,
                T*                A,
                rocblas_int       lda,
                T*                beta,
                T*                B,
                rocblas_int       ldb,
                T*                C,
                rocblas_int       ldc);

// gemm
template <typename Ti, typename To = Ti, typename Tc>
void cblas_gemm(rocblas_operation      transA,
                rocblas_operation      transB,
                rocblas_int            m,
                rocblas_int            n,
                rocblas_int            k,
                Tc                     alpha,
                Ti*                    A,
                rocblas_int            lda,
                Ti*                    B,
                rocblas_int            ldb,
                Tc                     beta,
                std::add_pointer_t<To> C,
                rocblas_int            ldc);

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

// symm
template <typename T>
void cblas_symm(rocblas_side side,
                rocblas_fill uplo,
                rocblas_int  m,
                rocblas_int  n,
                T            alpha,
                const T*     A,
                rocblas_int  lda,
                const T*     B,
                rocblas_int  ldb,
                T            beta,
                T*           C,
                rocblas_int  ldc);

template <>
inline void cblas_symm(rocblas_side side,
                       rocblas_fill uplo,
                       rocblas_int  m,
                       rocblas_int  n,
                       float        alpha,
                       const float* A,
                       rocblas_int  lda,
                       const float* B,
                       rocblas_int  ldb,
                       float        beta,
                       float*       C,
                       rocblas_int  ldc)
{
    cblas_ssymm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                m,
                n,
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
inline void cblas_symm(rocblas_side  side,
                       rocblas_fill  uplo,
                       rocblas_int   m,
                       rocblas_int   n,
                       double        alpha,
                       const double* A,
                       rocblas_int   lda,
                       const double* B,
                       rocblas_int   ldb,
                       double        beta,
                       double*       C,
                       rocblas_int   ldc)
{
    cblas_dsymm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                m,
                n,
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
inline void cblas_symm(rocblas_side                 side,
                       rocblas_fill                 uplo,
                       rocblas_int                  m,
                       rocblas_int                  n,
                       rocblas_float_complex        alpha,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       const rocblas_float_complex* B,
                       rocblas_int                  ldb,
                       rocblas_float_complex        beta,
                       rocblas_float_complex*       C,
                       rocblas_int                  ldc)
{
    cblas_csymm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                m,
                n,
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
inline void cblas_symm(rocblas_side                  side,
                       rocblas_fill                  uplo,
                       rocblas_int                   m,
                       rocblas_int                   n,
                       rocblas_double_complex        alpha,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       const rocblas_double_complex* B,
                       rocblas_int                   ldb,
                       rocblas_double_complex        beta,
                       rocblas_double_complex*       C,
                       rocblas_int                   ldc)
{
    cblas_zsymm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// syrk
template <typename T>
void cblas_syrk(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_int       n,
                rocblas_int       k,
                T                 alpha,
                const T*          A,
                rocblas_int       lda,
                T                 beta,
                T*                C,
                rocblas_int       ldc);

template <>
inline void cblas_syrk(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_int       n,
                       rocblas_int       k,
                       float             alpha,
                       const float*      A,
                       rocblas_int       lda,
                       float             beta,
                       float*            C,
                       rocblas_int       ldc)
{
    cblas_ssyrk(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
inline void cblas_syrk(rocblas_fill      uplo,
                       rocblas_operation transA,
                       rocblas_int       n,
                       rocblas_int       k,
                       double            alpha,
                       const double*     A,
                       rocblas_int       lda,
                       double            beta,
                       double*           C,
                       rocblas_int       ldc)
{
    cblas_dsyrk(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
inline void cblas_syrk(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_int                  n,
                       rocblas_int                  k,
                       rocblas_float_complex        alpha,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       rocblas_float_complex        beta,
                       rocblas_float_complex*       C,
                       rocblas_int                  ldc)
{
    cblas_csyrk(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                n,
                k,
                &alpha,
                A,
                lda,
                &beta,
                C,
                ldc);
}

template <>
inline void cblas_syrk(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_int                   n,
                       rocblas_int                   k,
                       rocblas_double_complex        alpha,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       rocblas_double_complex        beta,
                       rocblas_double_complex*       C,
                       rocblas_int                   ldc)
{
    cblas_zsyrk(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                n,
                k,
                &alpha,
                A,
                lda,
                &beta,
                C,
                ldc);
}

// syr2k
template <typename T>
void cblas_syr2k(rocblas_fill      uplo,
                 rocblas_operation transA,
                 rocblas_int       n,
                 rocblas_int       k,
                 T                 alpha,
                 const T*          A,
                 rocblas_int       lda,
                 const T*          B,
                 rocblas_int       ldb,
                 T                 beta,
                 T*                C,
                 rocblas_int       ldc);

template <>
inline void cblas_syr2k(rocblas_fill      uplo,
                        rocblas_operation transA,
                        rocblas_int       n,
                        rocblas_int       k,
                        float             alpha,
                        const float*      A,
                        rocblas_int       lda,
                        const float*      B,
                        rocblas_int       ldb,
                        float             beta,
                        float*            C,
                        rocblas_int       ldc)
{
    cblas_ssyr2k(CblasColMajor,
                 CBLAS_UPLO(uplo),
                 CBLAS_TRANSPOSE(transA),
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
inline void cblas_syr2k(rocblas_fill      uplo,
                        rocblas_operation transA,
                        rocblas_int       n,
                        rocblas_int       k,
                        double            alpha,
                        const double*     A,
                        rocblas_int       lda,
                        const double*     B,
                        rocblas_int       ldb,
                        double            beta,
                        double*           C,
                        rocblas_int       ldc)
{
    cblas_dsyr2k(CblasColMajor,
                 CBLAS_UPLO(uplo),
                 CBLAS_TRANSPOSE(transA),
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
inline void cblas_syr2k(rocblas_fill                 uplo,
                        rocblas_operation            transA,
                        rocblas_int                  n,
                        rocblas_int                  k,
                        rocblas_float_complex        alpha,
                        const rocblas_float_complex* A,
                        rocblas_int                  lda,
                        const rocblas_float_complex* B,
                        rocblas_int                  ldb,
                        rocblas_float_complex        beta,
                        rocblas_float_complex*       C,
                        rocblas_int                  ldc)
{
    cblas_csyr2k(CblasColMajor,
                 CBLAS_UPLO(uplo),
                 CBLAS_TRANSPOSE(transA),
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
inline void cblas_syr2k(rocblas_fill                  uplo,
                        rocblas_operation             transA,
                        rocblas_int                   n,
                        rocblas_int                   k,
                        rocblas_double_complex        alpha,
                        const rocblas_double_complex* A,
                        rocblas_int                   lda,
                        const rocblas_double_complex* B,
                        rocblas_int                   ldb,
                        rocblas_double_complex        beta,
                        rocblas_double_complex*       C,
                        rocblas_int                   ldc)
{
    cblas_zsyr2k(CblasColMajor,
                 CBLAS_UPLO(uplo),
                 CBLAS_TRANSPOSE(transA),
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

// hemm
template <typename T>
void cblas_hemm(rocblas_side side,
                rocblas_fill uplo,
                rocblas_int  m,
                rocblas_int  n,
                const T*     alpha,
                const T*     A,
                rocblas_int  lda,
                const T*     B,
                rocblas_int  ldb,
                const T*     beta,
                T*           C,
                rocblas_int  ldc);

template <>
inline void cblas_hemm(rocblas_side                 side,
                       rocblas_fill                 uplo,
                       rocblas_int                  m,
                       rocblas_int                  n,
                       const rocblas_float_complex* alpha,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       const rocblas_float_complex* B,
                       rocblas_int                  ldb,
                       const rocblas_float_complex* beta,
                       rocblas_float_complex*       C,
                       rocblas_int                  ldc)
{
    cblas_chemm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                m,
                n,
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
inline void cblas_hemm(rocblas_side                  side,
                       rocblas_fill                  uplo,
                       rocblas_int                   m,
                       rocblas_int                   n,
                       const rocblas_double_complex* alpha,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       const rocblas_double_complex* B,
                       rocblas_int                   ldb,
                       const rocblas_double_complex* beta,
                       rocblas_double_complex*       C,
                       rocblas_int                   ldc)
{
    cblas_zhemm(CblasColMajor,
                CBLAS_SIDE(side),
                CBLAS_UPLO(uplo),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

// herk
template <typename T, typename U>
void cblas_herk(rocblas_fill      uplo,
                rocblas_operation transA,
                rocblas_int       n,
                rocblas_int       k,
                U                 alpha,
                const T*          A,
                rocblas_int       lda,
                U                 beta,
                T*                C,
                rocblas_int       ldc);

template <>
inline void cblas_herk(rocblas_fill                 uplo,
                       rocblas_operation            transA,
                       rocblas_int                  n,
                       rocblas_int                  k,
                       float                        alpha,
                       const rocblas_float_complex* A,
                       rocblas_int                  lda,
                       float                        beta,
                       rocblas_float_complex*       C,
                       rocblas_int                  ldc)
{
    cblas_cherk(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
inline void cblas_herk(rocblas_fill                  uplo,
                       rocblas_operation             transA,
                       rocblas_int                   n,
                       rocblas_int                   k,
                       double                        alpha,
                       const rocblas_double_complex* A,
                       rocblas_int                   lda,
                       double                        beta,
                       rocblas_double_complex*       C,
                       rocblas_int                   ldc)
{
    cblas_zherk(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

// her2k
template <typename T>
void cblas_her2k(rocblas_fill      uplo,
                 rocblas_operation transA,
                 rocblas_int       n,
                 rocblas_int       k,
                 const T*          alpha,
                 const T*          A,
                 rocblas_int       lda,
                 const T*          B,
                 rocblas_int       ldb,
                 const real_t<T>*  beta,
                 T*                C,
                 rocblas_int       ldc);

template <>
inline void cblas_her2k(rocblas_fill                 uplo,
                        rocblas_operation            transA,
                        rocblas_int                  n,
                        rocblas_int                  k,
                        const rocblas_float_complex* alpha,
                        const rocblas_float_complex* A,
                        rocblas_int                  lda,
                        const rocblas_float_complex* B,
                        rocblas_int                  ldb,
                        const float*                 beta,
                        rocblas_float_complex*       C,
                        rocblas_int                  ldc)
{
    cblas_cher2k(CblasColMajor,
                 CBLAS_UPLO(uplo),
                 CBLAS_TRANSPOSE(transA),
                 n,
                 k,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 *beta,
                 C,
                 ldc);
}

template <>
inline void cblas_her2k(rocblas_fill                  uplo,
                        rocblas_operation             transA,
                        rocblas_int                   n,
                        rocblas_int                   k,
                        const rocblas_double_complex* alpha,
                        const rocblas_double_complex* A,
                        rocblas_int                   lda,
                        const rocblas_double_complex* B,
                        rocblas_int                   ldb,
                        const double*                 beta,
                        rocblas_double_complex*       C,
                        rocblas_int                   ldc)
{
    cblas_zher2k(CblasColMajor,
                 CBLAS_UPLO(uplo),
                 CBLAS_TRANSPOSE(transA),
                 n,
                 k,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 *beta,
                 C,
                 ldc);
}

// cblas_herkx doesn't exist. implementation in cpp
template <typename T, typename U = real_t<T>>
void cblas_herkx(rocblas_fill      uplo,
                 rocblas_operation transA,
                 rocblas_int       n,
                 rocblas_int       k,
                 const T*          alpha,
                 const T*          A,
                 rocblas_int       lda,
                 const T*          B,
                 rocblas_int       ldb,
                 const U*          beta,
                 T*                C,
                 rocblas_int       ldc);

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

template <>
inline rocblas_int
    cblas_trtri(char uplo, char diag, rocblas_int n, rocblas_float_complex* A, rocblas_int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    rocblas_int info;
    ctrtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

template <>
inline rocblas_int
    cblas_trtri(char uplo, char diag, rocblas_int n, rocblas_double_complex* A, rocblas_int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    rocblas_int info;
    ztrtri_(&uplo, &diag, &n, A, &lda, &info);
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
