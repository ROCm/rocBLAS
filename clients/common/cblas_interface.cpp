/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include <typeinfo>
#include <memory>
#include "rocblas.h"
#include "cblas_interface.h"
#include "cblas.h"
#include "utility.h"

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
 */

#ifdef __cplusplus
extern "C" {
#endif

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

void spotf2_(char* uplo, int* n, float* A, int* lda, int* info);
void dpotf2_(char* uplo, int* n, double* A, int* lda, int* info);
void cpotf2_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotf2_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

#ifdef __cplusplus
}
#endif

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */
// scal
template <>
void cblas_scal<float>(rocblas_int n, const float alpha, float* x, rocblas_int incx)
{
    cblas_sscal(n, alpha, x, incx);
}

template <>
void cblas_scal<double>(rocblas_int n, const double alpha, double* x, rocblas_int incx)
{
    cblas_dscal(n, alpha, x, incx);
}

template <>
void cblas_scal<rocblas_float_complex>(rocblas_int n,
                                       const rocblas_float_complex alpha,
                                       rocblas_float_complex* x,
                                       rocblas_int incx)
{
    cblas_cscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<rocblas_double_complex>(rocblas_int n,
                                        const rocblas_double_complex alpha,
                                        rocblas_double_complex* x,
                                        rocblas_int incx)
{
    cblas_zscal(n, &alpha, x, incx);
}

// copy
template <>
void cblas_copy<float>(rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    cblas_scopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<double>(rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<rocblas_float_complex>(rocblas_int n,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex* y,
                                       rocblas_int incy)
{
    cblas_ccopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    cblas_zcopy(n, x, incx, y, incy);
}

// axpy
template <>
void cblas_axpy<rocblas_half>(rocblas_int n,
                              rocblas_half alpha,
                              rocblas_half* x,
                              rocblas_int incx,
                              rocblas_half* y,
                              rocblas_int incy)
{
    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    float alpha_float    = half_to_float(alpha);
    std::unique_ptr<float[]> x_float(new float[n * abs_incx]());
    std::unique_ptr<float[]> y_float(new float[n * abs_incy]());
    for(int i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }

    cblas_saxpy(n, alpha_float, x_float.get(), incx, y_float.get(), incy);

    for(int i = 0; i < n; i++)
    {
        x[i * abs_incx] = float_to_half(x_float[i * abs_incx]);
        y[i * abs_incy] = float_to_half(y_float[i * abs_incy]);
    }
}

template <>
void cblas_axpy<float>(
    rocblas_int n, float alpha, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<double>(
    rocblas_int n, double alpha, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<rocblas_float_complex>(rocblas_int n,
                                       rocblas_float_complex alpha,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex* y,
                                       rocblas_int incy)
{
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex alpha,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// swap
template <>
void cblas_swap<float>(rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    cblas_sswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<double>(rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    cblas_dswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<rocblas_float_complex>(rocblas_int n,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex* y,
                                       rocblas_int incy)
{
    cblas_cswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    cblas_zswap(n, x, incx, y, incy);
}

// dot
template <>
void cblas_dot<float>(rocblas_int n,
                      const float* x,
                      rocblas_int incx,
                      const float* y,
                      rocblas_int incy,
                      float* result)
{
    *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
void cblas_dot<double>(rocblas_int n,
                       const double* x,
                       rocblas_int incx,
                       const double* y,
                       rocblas_int incy,
                       double* result)
{
    *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
void cblas_dot<rocblas_float_complex>(rocblas_int n,
                                      const rocblas_float_complex* x,
                                      rocblas_int incx,
                                      const rocblas_float_complex* y,
                                      rocblas_int incy,
                                      rocblas_float_complex* result)
{
    cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dot<rocblas_double_complex>(rocblas_int n,
                                       const rocblas_double_complex* x,
                                       rocblas_int incx,
                                       const rocblas_double_complex* y,
                                       rocblas_int incy,
                                       rocblas_double_complex* result)
{
    cblas_zdotu_sub(n, x, incx, y, incy, result);
}

// nrm2
template <>
void cblas_nrm2<float, float>(rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    *result = cblas_snrm2(n, x, incx);
}

template <>
void cblas_nrm2<double, double>(rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    *result = cblas_dnrm2(n, x, incx);
}

template <>
void cblas_nrm2<rocblas_float_complex, float>(rocblas_int n,
                                              const rocblas_float_complex* x,
                                              rocblas_int incx,
                                              float* result)
{
    *result = cblas_scnrm2(n, x, incx);
}

template <>
void cblas_nrm2<rocblas_double_complex, double>(rocblas_int n,
                                                const rocblas_double_complex* x,
                                                rocblas_int incx,
                                                double* result)
{
    *result = cblas_dznrm2(n, x, incx);
}

// asum
template <>
void cblas_asum<float, float>(rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    *result = cblas_sasum(n, x, incx);
}

template <>
void cblas_asum<double, double>(rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    *result = cblas_dasum(n, x, incx);
}

template <>
void cblas_asum<rocblas_float_complex, float>(rocblas_int n,
                                              const rocblas_float_complex* x,
                                              rocblas_int incx,
                                              float* result)
{
    *result = cblas_scasum(n, x, incx);
}

template <>
void cblas_asum<rocblas_double_complex, double>(rocblas_int n,
                                                const rocblas_double_complex* x,
                                                rocblas_int incx,
                                                double* result)
{
    *result = cblas_dzasum(n, x, incx);
}

// amax
template <>
void cblas_iamax<float>(rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result)
{
    *result = (rocblas_int)cblas_isamax(n, x, incx);
}

template <>
void cblas_iamax<double>(rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result)
{
    *result = (rocblas_int)cblas_idamax(n, x, incx);
}

template <>
void cblas_iamax<rocblas_float_complex>(rocblas_int n,
                                        const rocblas_float_complex* x,
                                        rocblas_int incx,
                                        rocblas_int* result)
{
    *result = (rocblas_int)cblas_icamax(n, x, incx);
}

template <>
void cblas_iamax<rocblas_double_complex>(rocblas_int n,
                                         const rocblas_double_complex* x,
                                         rocblas_int incx,
                                         rocblas_int* result)
{
    *result = (rocblas_int)cblas_izamax(n, x, incx);
}
/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

template <>
void cblas_gemv<float>(rocblas_operation transA,
                       rocblas_int m,
                       rocblas_int n,
                       float alpha,
                       float* A,
                       rocblas_int lda,
                       float* x,
                       rocblas_int incx,
                       float beta,
                       float* y,
                       rocblas_int incy)
{
    cblas_sgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_gemv<double>(rocblas_operation transA,
                        rocblas_int m,
                        rocblas_int n,
                        double alpha,
                        double* A,
                        rocblas_int lda,
                        double* x,
                        rocblas_int incx,
                        double beta,
                        double* y,
                        rocblas_int incy)
{
    cblas_dgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_gemv<rocblas_float_complex>(rocblas_operation transA,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex beta,
                                       rocblas_float_complex* y,
                                       rocblas_int incy)
{
    cblas_cgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_gemv<rocblas_double_complex>(rocblas_operation transA,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_double_complex alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex beta,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    cblas_zgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_symv<float>(rocblas_fill uplo,
                       rocblas_int n,
                       float alpha,
                       float* A,
                       rocblas_int lda,
                       float* x,
                       rocblas_int incx,
                       float beta,
                       float* y,
                       rocblas_int incy)
{
    cblas_ssymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_symv<double>(rocblas_fill uplo,
                        rocblas_int n,
                        double alpha,
                        double* A,
                        rocblas_int lda,
                        double* x,
                        rocblas_int incx,
                        double beta,
                        double* y,
                        rocblas_int incy)
{
    cblas_dsymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_hemv<rocblas_float_complex>(rocblas_fill uplo,
                                       rocblas_int n,
                                       rocblas_float_complex alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex beta,
                                       rocblas_float_complex* y,
                                       rocblas_int incy)
{
    cblas_chemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_hemv<rocblas_double_complex>(rocblas_fill uplo,
                                        rocblas_int n,
                                        rocblas_double_complex alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex beta,
                                        rocblas_double_complex* y,
                                        rocblas_int incy)
{
    cblas_zhemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_ger<float>(rocblas_int m,
                      rocblas_int n,
                      float alpha,
                      float* x,
                      rocblas_int incx,
                      float* y,
                      rocblas_int incy,
                      float* A,
                      rocblas_int lda)
{
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<double>(rocblas_int m,
                       rocblas_int n,
                       double alpha,
                       double* x,
                       rocblas_int incx,
                       double* y,
                       rocblas_int incy,
                       double* A,
                       rocblas_int lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_syr<float>(rocblas_fill uplo,
                      rocblas_int n,
                      float alpha,
                      float* x,
                      rocblas_int incx,
                      float* A,
                      rocblas_int lda)
{
    cblas_ssyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void cblas_syr<double>(rocblas_fill uplo,
                       rocblas_int n,
                       double alpha,
                       double* x,
                       rocblas_int incx,
                       double* A,
                       rocblas_int lda)
{
    cblas_dsyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */
// gemm
template <>
void cblas_gemm<rocblas_half>(rocblas_operation transA,
                              rocblas_operation transB,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              rocblas_half alpha,
                              rocblas_half* A,
                              rocblas_int lda,
                              rocblas_half* B,
                              rocblas_int ldb,
                              rocblas_half beta,
                              rocblas_half* C,
                              rocblas_int ldc)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing
    float alpha_float = half_to_float(alpha);
    float beta_float  = half_to_float(beta);

    int sizeA = transA == rocblas_operation_none ? k * lda : m * lda;
    int sizeB = transB == rocblas_operation_none ? n * ldb : k * ldb;
    int sizeC = n * ldc;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());
    std::unique_ptr<float[]> C_float(new float[sizeC]());

    for(int i = 0; i < sizeA; i++)
    {
        A_float[i] = half_to_float(A[i]);
    }
    for(int i = 0; i < sizeB; i++)
    {
        B_float[i] = half_to_float(B[i]);
    }
    for(int i = 0; i < sizeC; i++)
    {
        C_float[i] = half_to_float(C[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha_float,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta_float,
                static_cast<float*>(C_float.get()),
                ldc);

    for(int i = 0; i < sizeC; i++)
    {
        C[i] = float_to_half(C_float[i]);
    }
}

template <>
void cblas_gemm<float>(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int m,
                       rocblas_int n,
                       rocblas_int k,
                       float alpha,
                       float* A,
                       rocblas_int lda,
                       float* B,
                       rocblas_int ldb,
                       float beta,
                       float* C,
                       rocblas_int ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
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
void cblas_gemm<double>(rocblas_operation transA,
                        rocblas_operation transB,
                        rocblas_int m,
                        rocblas_int n,
                        rocblas_int k,
                        double alpha,
                        double* A,
                        rocblas_int lda,
                        double* B,
                        rocblas_int ldb,
                        double beta,
                        double* C,
                        rocblas_int ldc)
{
    cblas_dgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
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
void cblas_gemm<rocblas_float_complex>(rocblas_operation transA,
                                       rocblas_operation transB,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_int k,
                                       rocblas_float_complex alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb,
                                       rocblas_float_complex beta,
                                       rocblas_float_complex* C,
                                       rocblas_int ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_cgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
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
void cblas_gemm<rocblas_double_complex>(rocblas_operation transA,
                                        rocblas_operation transB,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_double_complex alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb,
                                        rocblas_double_complex beta,
                                        rocblas_double_complex* C,
                                        rocblas_int ldc)
{
    cblas_zgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
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
template <>
void cblas_trsm<float>(rocblas_side side,
                       rocblas_fill uplo,
                       rocblas_operation transA,
                       rocblas_diagonal diag,
                       rocblas_int m,
                       rocblas_int n,
                       float alpha,
                       const float* A,
                       rocblas_int lda,
                       float* B,
                       rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trsm<double>(rocblas_side side,
                        rocblas_fill uplo,
                        rocblas_operation transA,
                        rocblas_diagonal diag,
                        rocblas_int m,
                        rocblas_int n,
                        double alpha,
                        const double* A,
                        rocblas_int lda,
                        double* B,
                        rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trsm<rocblas_float_complex>(rocblas_side side,
                                       rocblas_fill uplo,
                                       rocblas_operation transA,
                                       rocblas_diagonal diag,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex alpha,
                                       const rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ctrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trsm<rocblas_double_complex>(rocblas_side side,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_double_complex alpha,
                                        const rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

// potf2
template <>
rocblas_int cblas_potf2(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda)
{
    rocblas_int info;
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    spotf2_(&uploC, &n, A, &lda, &info);
    return info;
}

template <>
rocblas_int cblas_potf2(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda)
{
    rocblas_int info;
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    dpotf2_(&uploC, &n, A, &lda, &info);
    return info;
}

// trtri
template <>
rocblas_int cblas_trtri<float>(char uplo, char diag, rocblas_int n, float* A, rocblas_int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    rocblas_int info;
    strtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

template <>
rocblas_int cblas_trtri<double>(char uplo, char diag, rocblas_int n, double* A, rocblas_int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    rocblas_int info;
    dtrtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

// trmm
template <>
void cblas_trmm<float>(rocblas_side side,
                       rocblas_fill uplo,
                       rocblas_operation transA,
                       rocblas_diagonal diag,
                       rocblas_int m,
                       rocblas_int n,
                       float alpha,
                       const float* A,
                       rocblas_int lda,
                       float* B,
                       rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trmm<double>(rocblas_side side,
                        rocblas_fill uplo,
                        rocblas_operation transA,
                        rocblas_diagonal diag,
                        rocblas_int m,
                        rocblas_int n,
                        double alpha,
                        const double* A,
                        rocblas_int lda,
                        double* B,
                        rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trmm<rocblas_float_complex>(rocblas_side side,
                                       rocblas_fill uplo,
                                       rocblas_operation transA,
                                       rocblas_diagonal diag,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex alpha,
                                       const rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ctrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trmm<rocblas_double_complex>(rocblas_side side,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_double_complex alpha,
                                        const rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_ztrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

// getrf
template <>
rocblas_int
cblas_getrf<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
rocblas_int
cblas_getrf<double>(rocblas_int m, rocblas_int n, double* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
rocblas_int cblas_getrf<rocblas_float_complex>(
    rocblas_int m, rocblas_int n, rocblas_float_complex* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    cgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
rocblas_int cblas_getrf<rocblas_double_complex>(
    rocblas_int m, rocblas_int n, rocblas_double_complex* A, rocblas_int lda, rocblas_int* ipiv)
{
    rocblas_int info;
    zgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// potrf
template <>
rocblas_int cblas_potrf<float>(char uplo, rocblas_int m, float* A, rocblas_int lda)
{
    rocblas_int info;
    spotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
rocblas_int cblas_potrf<double>(char uplo, rocblas_int m, double* A, rocblas_int lda)
{
    rocblas_int info;
    dpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
rocblas_int cblas_potrf<rocblas_float_complex>(char uplo,
                                               rocblas_int m,
                                               rocblas_float_complex* A,
                                               rocblas_int lda)
{
    rocblas_int info;
    cpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
rocblas_int cblas_potrf<rocblas_double_complex>(char uplo,
                                                rocblas_int m,
                                                rocblas_double_complex* A,
                                                rocblas_int lda)
{
    rocblas_int info;
    zpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}
