/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/


#include <typeinfo>
#include "rocblas.h"
#include "cblas_interface.h"
#include "cblas.h"

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing not part of the GPU library
*/

#ifdef __cplusplus
extern "C" {
#endif

    void    strtri_(char* uplo, char* diag, int* n, float* A, int* lda, int *info);
    void    dtrtri_(char* uplo, char* diag, int* n, double* A, int* lda, int *info);
    void    ctrtri_(char* uplo, char* diag, int* n, rocblas_float_complex* A,  int* lda, int *info);
    void    ztrtri_(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int *info);

#ifdef __cplusplus
}
#endif

    /*
     * ===========================================================================
     *    level 1 BLAS
     * ===========================================================================
     */
    //scal
    template<>
    void cblas_scal<float>( rocblas_int n,
                            const float alpha,
                            float *x, rocblas_int incx)
    {
        cblas_sscal(n, alpha, x, incx);
    }

    template<>
    void cblas_scal<double>(rocblas_int n,
                            const double alpha,
                            double *x, rocblas_int incx)
    {
        cblas_dscal(n, alpha, x, incx);
    }

    template<>
    void cblas_scal<rocblas_float_complex>( rocblas_int n,
                            const rocblas_float_complex alpha,
                            rocblas_float_complex *x, rocblas_int incx)
    {
        cblas_cscal(n, &alpha, x, incx);
    }

    template<>
    void cblas_scal<rocblas_double_complex>( rocblas_int n,
                            const rocblas_double_complex alpha,
                            rocblas_double_complex *x, rocblas_int incx)
    {
        cblas_zscal(n, &alpha, x, incx);
    }

    //copy
    template<>
    void cblas_copy<float>( rocblas_int n,
                            float *x, rocblas_int incx,
                            float *y, rocblas_int incy)
    {
        cblas_scopy(n, x, incx, y, incy);
    }

    template<>
    void cblas_copy<double>(rocblas_int n,
                            double *x, rocblas_int incx,
                            double *y, rocblas_int incy)
    {
        cblas_dcopy(n, x, incx, y, incy);
    }

    template<>
    void cblas_copy<rocblas_float_complex>( rocblas_int n,
                            rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy)
    {
        cblas_ccopy(n, x, incx, y, incy);
    }

    template<>
    void cblas_copy<rocblas_double_complex>( rocblas_int n,
                            rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy)
    {
        cblas_zcopy(n, x, incx, y, incy);
    }

    //swap
    template<>
    void cblas_swap<float>( rocblas_int n,
                            float *x, rocblas_int incx,
                            float *y, rocblas_int incy)
    {
        cblas_sswap(n, x, incx, y, incy);
    }

    template<>
    void cblas_swap<double>(rocblas_int n,
                            double *x, rocblas_int incx,
                            double *y, rocblas_int incy)
    {
        cblas_dswap(n, x, incx, y, incy);
    }

    template<>
    void cblas_swap<rocblas_float_complex>( rocblas_int n,
                            rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy)
    {
        cblas_cswap(n, x, incx, y, incy);
    }

    template<>
    void cblas_swap<rocblas_double_complex>( rocblas_int n,
                            rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy)
    {
        cblas_zswap(n, x, incx, y, incy);
    }

    //dot
    template<>
    void cblas_dot<float>(  rocblas_int n,
                            const float *x, rocblas_int incx,
                            const float *y, rocblas_int incy,
                            float *result)
    {
        *result = cblas_sdot(n, x, incx, y, incy);
    }

    template<>
    void cblas_dot<double>( rocblas_int n,
                            const double *x, rocblas_int incx,
                            const double *y, rocblas_int incy,
                            double *result)
    {
        *result = cblas_ddot(n, x, incx, y, incy);
    }

    template<>
    void cblas_dot<rocblas_float_complex>( rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            const rocblas_float_complex *y, rocblas_int incy,
                            rocblas_float_complex *result)
    {
        cblas_cdotu_sub(n, x, incx, y, incy, result);
    }

    template<>
    void cblas_dot<rocblas_double_complex>( rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            const rocblas_double_complex *y, rocblas_int incy,
                            rocblas_double_complex *result)
    {
        cblas_zdotu_sub(n, x, incx, y, incy, result);
    }

    //nrm2
    template<>
    void cblas_nrm2<float, float>( rocblas_int n,
                            const float *x, rocblas_int incx,
                            float *result)
    {
        *result = cblas_snrm2(n, x, incx);
    }

    template<>
    void cblas_nrm2<double, double>( rocblas_int n,
                             const double *x, rocblas_int incx,
                             double *result)
    {
        *result = cblas_dnrm2(n, x, incx);
    }

    template<>
    void cblas_nrm2<rocblas_float_complex, float>( rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            float *result)
    {
        *result = cblas_scnrm2(n, x, incx);
    }

    template<>
    void cblas_nrm2<rocblas_double_complex, double>( rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            double *result)
    {
        *result = cblas_dznrm2(n, x, incx);
    }


    //nrm2
    template<>
    void cblas_asum<float, float>( rocblas_int n,
                            const float *x, rocblas_int incx,
                            float *result)
    {
        *result = cblas_sasum(n, x, incx);
    }

    template<>
    void cblas_asum<double, double>( rocblas_int n,
                             const double *x, rocblas_int incx,
                             double *result)
    {
        *result = cblas_dasum(n, x, incx);
    }

    template<>
    void cblas_asum<rocblas_float_complex, float>( rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            float *result)
    {
        *result = cblas_scasum(n, x, incx);
    }

    template<>
    void cblas_asum<rocblas_double_complex, double>( rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            double *result)
    {
        *result = cblas_dzasum(n, x, incx);
    }

    //amax
    template<>
    void cblas_amax<float>( rocblas_int n,
                            const float *x, rocblas_int incx,
                            rocblas_int *result)
    {
        *result = (rocblas_int)cblas_isamax(n, x, incx);
    }

    template<>
    void cblas_amax<double>( rocblas_int n,
                            const double *x, rocblas_int incx,
                            rocblas_int *result)
    {
        *result = (rocblas_int)cblas_idamax(n, x, incx);
    }

    template<>
    void cblas_amax<rocblas_float_complex>( rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            rocblas_int *result)
    {
        *result = (rocblas_int)cblas_icamax(n, x, incx);
    }

    template<>
    void cblas_amax<rocblas_double_complex>( rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            rocblas_int *result)
    {
        *result = (rocblas_int)cblas_izamax(n, x, incx);
    }
    /*
     * ===========================================================================
     *    level 2 BLAS
     * ===========================================================================
     */

    template<>
    void cblas_gemv<float>( rocblas_operation transA, rocblas_int m, rocblas_int n,
                            float alpha,
                            float *A, rocblas_int lda,
                            float *x, rocblas_int incx,
                            float beta, float *y, rocblas_int incy)
    {
        cblas_sgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template<>
    void cblas_gemv<double>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                            double alpha,
                            double *A, rocblas_int lda,
                            double *x, rocblas_int incx,
                            double beta, double *y, rocblas_int incy)
    {
        cblas_dgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template<>
    void cblas_gemv<rocblas_float_complex>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                            rocblas_float_complex alpha,
                            rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex beta, rocblas_float_complex *y, rocblas_int incy)
    {
        cblas_cgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    }

    template<>
    void cblas_gemv<rocblas_double_complex>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                            rocblas_double_complex alpha,
                            rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex beta, rocblas_double_complex *y, rocblas_int incy)
    {
        cblas_zgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    }

    template<>
    void cblas_symv<float>( rocblas_fill uplo, rocblas_int n,
                            float alpha,
                            float *A, rocblas_int lda,
                            float *x, rocblas_int incx,
                            float beta, float *y, rocblas_int incy)
    {
        cblas_ssymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template<>
    void cblas_symv<double>(rocblas_fill uplo, rocblas_int n,
                            double alpha,
                            double *A, rocblas_int lda,
                            double *x, rocblas_int incx,
                            double beta, double *y, rocblas_int incy)
    {
        cblas_dsymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template<>
    void cblas_hemv<rocblas_float_complex>(rocblas_fill uplo, rocblas_int n,
                            rocblas_float_complex alpha,
                            rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex beta, rocblas_float_complex *y, rocblas_int incy)
    {
        cblas_chemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
    }

    template<>
    void cblas_hemv<rocblas_double_complex>(rocblas_fill uplo, rocblas_int n,
                            rocblas_double_complex alpha,
                            rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex beta, rocblas_double_complex *y, rocblas_int incy)
    {
        cblas_zhemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
    }

    template<>
    void cblas_ger<float>( rocblas_int m, rocblas_int n,
                            float alpha,
                            float *x, rocblas_int incx,
                            float *y, rocblas_int incy,
                            float *A, rocblas_int lda)
    {
        cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
    }

    template<>
    void cblas_ger<double>(rocblas_int m, rocblas_int n,
                            double alpha,
                            double *x, rocblas_int incx,
                            double *y, rocblas_int incy,
                            double *A, rocblas_int lda)
    {
        cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
    }

    /*
     * ===========================================================================
     *    level 3 BLAS
     * ===========================================================================
     */

    //gemm

    template<>
    void cblas_gemm<float>( rocblas_operation transA, rocblas_operation transB,
                            rocblas_int m, rocblas_int n, rocblas_int k,
                            float alpha, float *A, rocblas_int lda,
                            float *B, rocblas_int ldb,
                            float beta, float *C, rocblas_int ldc)
    {
        //just directly cast, since transA, transB are integers in the enum
        //printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
        cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template<>
    void cblas_gemm<double>(rocblas_operation transA, rocblas_operation transB,
                            rocblas_int m, rocblas_int n, rocblas_int k,
                            double alpha, double *A, rocblas_int lda,
                            double *B, rocblas_int ldb,
                            double beta, double *C, rocblas_int ldc)
    {
        cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template<>
    void cblas_gemm<rocblas_float_complex>(rocblas_operation transA, rocblas_operation transB,
                            rocblas_int m, rocblas_int n, rocblas_int k,
                            rocblas_float_complex alpha, rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *B, rocblas_int ldb,
                            rocblas_float_complex beta, rocblas_float_complex *C, rocblas_int ldc)
    {
        //just directly cast, since transA, transB are integers in the enum
        cblas_cgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }

    template<>
    void cblas_gemm<rocblas_double_complex>(rocblas_operation transA, rocblas_operation transB,
                            rocblas_int m, rocblas_int n, rocblas_int k,
                            rocblas_double_complex alpha, rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *B, rocblas_int ldb,
                            rocblas_double_complex beta, rocblas_double_complex *C, rocblas_int ldc)
    {
        cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }

    //trsm
    template<>
    void cblas_trsm<float>( rocblas_side side, rocblas_fill uplo,
                            rocblas_operation transA, rocblas_diagonal diag,
                            rocblas_int m, rocblas_int n,
                            float alpha,
                            float *A, rocblas_int lda,
                            float *B, rocblas_int ldb)
    {
        //just directly cast, since transA, transB are integers in the enum
        cblas_strsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B, ldb);
    }

    template<>
    void cblas_trsm<double>(rocblas_side side, rocblas_fill uplo,
                            rocblas_operation transA, rocblas_diagonal diag,
                            rocblas_int m, rocblas_int n,
                            double alpha,
                            double *A, rocblas_int lda,
                            double *B, rocblas_int ldb)
    {
        //just directly cast, since transA, transB are integers in the enum
        cblas_dtrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B, ldb);
    }

    template<>
    void cblas_trsm<rocblas_float_complex>( rocblas_side side, rocblas_fill uplo,
                            rocblas_operation transA, rocblas_diagonal diag,
                            rocblas_int m, rocblas_int n,
                            rocblas_float_complex alpha,
                            rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *B, rocblas_int ldb)
    {
        //just directly cast, since transA, transB are integers in the enum
        cblas_ctrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda, B, ldb);
    }

    template<>
    void cblas_trsm<rocblas_double_complex>( rocblas_side side, rocblas_fill uplo,
                            rocblas_operation transA, rocblas_diagonal diag,
                            rocblas_int m, rocblas_int n,
                            rocblas_double_complex alpha,
                            rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *B, rocblas_int ldb)
    {
        //just directly cast, since transA, transB are integers in the enum
        cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda, B, ldb);
    }

    //trtri
    template<>
    rocblas_int cblas_trtri<float>(char uplo, char diag,
                            rocblas_int n,
                            float *A, rocblas_int lda)
    {
        //just directly cast, since transA, transB are integers in the enum
        //printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
        rocblas_int info;
        strtri_(&uplo, &diag, &n, A, &lda, &info);
        return info;
    }

    template<>
    rocblas_int cblas_trtri<double>(char uplo, char diag,
                            rocblas_int n,
                            double *A, rocblas_int lda)
    {
        //just directly cast, since transA, transB are integers in the enum
        //printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
        rocblas_int info;
        dtrtri_(&uplo, &diag, &n, A, &lda, &info);
        return info;
    }
