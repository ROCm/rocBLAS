/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_HPP_
#define _ROCBLAS_HPP_

/* library headers */
#include "rocblas-export.h"
#include "rocblas-version.h"
#include "rocblas_types.h"
#include "rocblas_auxiliary.h"
#include "rocblas_functions.h"


/*!\file
 * \brief rocblas_template_api.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.
 *  This file exposes C++ templated BLAS interface with only the precision templated.
*/

    /*
     * ===========================================================================
     *   READEME: Please follow the naming convention
     *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
     *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
     * ===========================================================================
     */


     template<typename T>
     rocblas_status
     rocblas_scal(rocblas_handle handle,
         rocblas_int n,
         const T *alpha,
         T *x, rocblas_int incx);

     template<typename T>
     rocblas_status
     rocblas_copy(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         T *y, rocblas_int incy);

     template<typename T>
     rocblas_status
     rocblas_swap(rocblas_handle handle,
         rocblas_int n,
         T *x, rocblas_int incx,
         T *y, rocblas_int incy);

     template<typename T>
     rocblas_status
     rocblas_dot(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         const T *y, rocblas_int incy,
         T *result);

     template<typename T1, typename T2>
     rocblas_status
     rocblas_asum(rocblas_handle handle,
         rocblas_int n,
         const T1 *x, rocblas_int incx,
         T2 *result);

     template<typename T1, typename T2>
     rocblas_status
     rocblas_nrm2(rocblas_handle handle,
         rocblas_int n,
         const T1 *x, rocblas_int incx,
         T2 *result);

     template<typename T>
     rocblas_status
     rocblas_amax(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         rocblas_int *result);

     template<typename T>
     rocblas_status
     rocblas_amin(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         rocblas_int *result);

     template<typename T>
     rocblas_status
     rocblas_axpy(rocblas_handle handle,
         rocblas_int n,
         const T *alpha,
         const T *x, rocblas_int incx,
         T *y, rocblas_int incy);

    template<typename T>
    rocblas_status
    rocblas_gemv(rocblas_handle handle,
             rocblas_operation transA,
             rocblas_int m, rocblas_int n,
             const T *alpha,
             const T *A, rocblas_int lda,
             const T *x, rocblas_int incx,
             const T *beta,
             T *y, rocblas_int incy);

    template<typename T>
    rocblas_status
    rocblas_ger(rocblas_handle handle,
             rocblas_int m, rocblas_int n,
             const T *alpha,
             const T *x, rocblas_int incx,
             const T *y, rocblas_int incy,
                   T *A, rocblas_int lda);

    template<typename T>
    rocblas_status
    rocblas_symv(rocblas_handle handle,
             rocblas_fill uplo,
             rocblas_int n,
             const T *alpha,
             const T *A, rocblas_int lda,
             const T *x, rocblas_int incx,
             const T *beta,
             T *y, rocblas_int incy);

    template<typename T>
    rocblas_status rocblas_gemm(rocblas_handle handle,
        rocblas_operation transA, rocblas_operation transB,
        rocblas_int m, rocblas_int n, rocblas_int k,
        const T *alpha,
        const T *A, rocblas_int lda,
        const T *B, rocblas_int ldb,
        const T *beta,
        T *C, rocblas_int ldc);

    template<typename T>
    rocblas_status rocblas_gemm_batched(
        rocblas_handle handle,
        rocblas_operation transA, rocblas_operation transB,
        rocblas_int m, rocblas_int n, rocblas_int k,
        const T *alpha,
        const T *A, rocblas_int lda, rocblas_int bsa,
        const T *B, rocblas_int ldb, rocblas_int bsb,
        const T *beta,
        T *C, rocblas_int ldc, rocblas_int bsc,
        rocblas_int batch_count);


    template<typename T>
    rocblas_status rocblas_trsm(rocblas_handle handle,
        rocblas_side side, rocblas_fill uplo,
        rocblas_operation transA, rocblas_diagonal diag,
        rocblas_int m, rocblas_int n,
        const T* alpha,
        const T* A, rocblas_int lda,
        T*       B, rocblas_int ldb);

    template<typename T>
    rocblas_status
    rocblas_trtri(rocblas_handle handle,
        rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        T *A, rocblas_int lda,
        T *invA, rocblas_int ldinvA);

    template<typename T>
    rocblas_status
    rocblas_trtri_batched(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda, rocblas_int bsa,
    T *invA, rocblas_int ldinvA, rocblas_int bsinvA,
    rocblas_int batch_count);

    template<typename T, rocblas_int NB>
    rocblas_status
    rocblas_trtri_trsm(rocblas_handle handle,
        rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        T *A, rocblas_int lda,
        T *invA);


    /*
     * ===========================================================================
     *    level 1 BLAS
     * ===========================================================================
     */
    //scal

    template<>
    rocblas_status
    rocblas_scal<float>(rocblas_handle handle,
        rocblas_int n,
        const float *alpha,
        float *x, rocblas_int incx){

        return rocblas_sscal(handle, n, alpha, x, incx);
    }


    template<>
    rocblas_status
    rocblas_scal<double>(rocblas_handle handle,
        rocblas_int n,
        const double *alpha,
        double *x, rocblas_int incx){

        return rocblas_dscal(handle, n, alpha, x, incx);
    }

    template<>
    rocblas_status
    rocblas_scal<rocblas_float_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *alpha,
        rocblas_float_complex *x, rocblas_int incx){

        return rocblas_cscal(handle, n, alpha, x, incx);
    }

    template<>
    rocblas_status
    rocblas_scal<rocblas_double_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *alpha,
        rocblas_double_complex *x, rocblas_int incx){

        return rocblas_zscal(handle, n, alpha, x, incx);
    }

    //swap
    template<>
    rocblas_status
    rocblas_swap<float>(    rocblas_handle handle, rocblas_int n,
                            float *x, rocblas_int incx,
                            float *y, rocblas_int incy)
    {
        return rocblas_swap(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_swap<double>(   rocblas_handle handle, rocblas_int n,
                            double *x, rocblas_int incx,
                            double *y, rocblas_int incy)
    {
        return rocblas_dswap(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_swap<rocblas_float_complex>(    rocblas_handle handle, rocblas_int n,
                            rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy)
    {
        return rocblas_cswap(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_swap<rocblas_double_complex>(    rocblas_handle handle, rocblas_int n,
                            rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy)
    {
        return rocblas_zswap(handle, n, x, incx, y, incy);
    }

    //copy
    template<>
    rocblas_status
    rocblas_copy<float>(   rocblas_handle handle, rocblas_int n,
                            const float *x, rocblas_int incx,
                            float *y, rocblas_int incy)
    {
        return rocblas_scopy(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_copy<double>(   rocblas_handle handle, rocblas_int n,
                            const double *x, rocblas_int incx,
                            double *y, rocblas_int incy)
    {
        return rocblas_dcopy(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_copy<rocblas_float_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy)
    {
        return rocblas_ccopy(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_copy<rocblas_double_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy)
    {
        return rocblas_zcopy(handle, n, x, incx, y, incy);
    }

    //dot
    template<>
    rocblas_status
    rocblas_dot<float>(    rocblas_handle handle, rocblas_int n,
                            const float *x, rocblas_int incx,
                            const float *y, rocblas_int incy,
                            float *result)
    {
        return rocblas_sdot(handle, n, x, incx, y, incy, result);
    }

    template<>
    rocblas_status
    rocblas_dot<double>(    rocblas_handle handle, rocblas_int n,
                            const double *x, rocblas_int incx,
                            const double *y, rocblas_int incy,
                            double *result)
    {
        return rocblas_ddot(handle, n, x, incx, y, incy, result);
    }

    template<>
    rocblas_status
    rocblas_dot<rocblas_float_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            const rocblas_float_complex *y, rocblas_int incy,
                            rocblas_float_complex *result)
    {
        return rocblas_cdotu(handle, n, x, incx, y, incy, result);
    }

    template<>
    rocblas_status
    rocblas_dot<rocblas_double_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            const rocblas_double_complex *y, rocblas_int incy,
                            rocblas_double_complex *result)
    {
        return rocblas_zdotu(handle, n, x, incx, y, incy, result);
    }


    //asum
    template<>
    rocblas_status
    rocblas_asum<float, float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        float *result){

        return rocblas_sasum(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_asum<double, double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        double *result){

        return rocblas_dasum(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_asum<rocblas_float_complex, float>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        float *result){

        return rocblas_scasum(handle, n, x, incx, result);
    }

    //nrm2
    template<>
    rocblas_status
    rocblas_nrm2<float, float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        float *result){

        return rocblas_snrm2(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_nrm2<double, double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        double *result){

        return rocblas_dnrm2(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_nrm2<rocblas_float_complex, float>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        float *result){

        return rocblas_scnrm2(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_nrm2<rocblas_double_complex, double>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *x, rocblas_int incx,
        double *result){

        return rocblas_dznrm2(handle, n, x, incx, result);
    }


    //amin
    template<>
    rocblas_status
    rocblas_amin<float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_samin(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amin<double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_damin(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amin<rocblas_float_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_scamin(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amin<rocblas_double_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_dzamin(handle, n, x, incx, result);
    }

    //amax
    template<>
    rocblas_status
    rocblas_amax<float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_samax(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amax<double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_damax(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amax<rocblas_float_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_scamax(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amax<rocblas_double_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_dzamax(handle, n, x, incx, result);
    }

    /*
     * ===========================================================================
     *    level 2 BLAS
     * ===========================================================================
     */

    template<>
    rocblas_status
    rocblas_gemv<float>(    rocblas_handle handle,
                            rocblas_operation transA, rocblas_int m, rocblas_int n,
                            const float *alpha,
                            const float *A, rocblas_int lda,
                            const float *x, rocblas_int incx,
                            const float *beta, float *y, rocblas_int incy)
    {
        return rocblas_sgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template<>
    rocblas_status
    rocblas_gemv<double>(   rocblas_handle handle,
                            rocblas_operation transA, rocblas_int m, rocblas_int n,
                            const double *alpha,
                            const double *A, rocblas_int lda,
                            const double *x, rocblas_int incx,
                            const double *beta, double *y, rocblas_int incy)
    {
        return rocblas_dgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }



    /*
     * ===========================================================================
     *    level 3 BLAS
     * ===========================================================================
     */

    //



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



    template<>
    rocblas_status
    rocblas_gemm<float>(rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int M, rocblas_int N, rocblas_int K,
        const float *alpha,
        const float *A, rocblas_int lda,
        const float *B, rocblas_int ldb,
        const float *beta,
        float *C, rocblas_int ldc)
    {
        return rocblas_sgemm(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template<>
    rocblas_status
    rocblas_gemm<double>(rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int M, rocblas_int N, rocblas_int K,
        const double *alpha,
        const double *A, rocblas_int lda,
        const double *B, rocblas_int ldb,
        const double *beta,
        double *C, rocblas_int ldc)
    {
        return rocblas_dgemm(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }



    template<>
    rocblas_status
    rocblas_gemm_batched<float>(rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int M, rocblas_int N, rocblas_int K,
        const float *alpha,
        const float *A, rocblas_int lda, rocblas_int bsa,
        const float *B, rocblas_int ldb, rocblas_int bsb,
        const float *beta,
        float *C, rocblas_int ldc, rocblas_int bsc,
        rocblas_int batch_count)
    {
        return rocblas_sgemm_batched(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc, batch_count);
    }

    template<>
    rocblas_status
    rocblas_gemm_batched<double>(rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int M, rocblas_int N, rocblas_int K,
        const double *alpha,
        const double *A, rocblas_int lda, rocblas_int bsa,
        const double *B, rocblas_int ldb, rocblas_int bsb,
        const double *beta,
        double *C, rocblas_int ldc, rocblas_int bsc,
        rocblas_int batch_count)
    {
        return rocblas_dgemm_batched(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc, batch_count);
    }



    template<>
    rocblas_status
    rocblas_trsm<float>(rocblas_handle handle,
        rocblas_side side, rocblas_fill uplo,
        rocblas_operation transA, rocblas_diagonal diag,
        rocblas_int m, rocblas_int n,
        const float* alpha,
        const float* A, rocblas_int lda,
        float*       B, rocblas_int ldb){

        return rocblas_strsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
    }

    template<>
    rocblas_status
    rocblas_trsm<double>(rocblas_handle handle,
        rocblas_side side, rocblas_fill uplo,
        rocblas_operation transA, rocblas_diagonal diag,
        rocblas_int m, rocblas_int n,
        const double* alpha,
        const double* A, rocblas_int lda,
        double*       B, rocblas_int ldb){

        return rocblas_dtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
    }




    template<>
    rocblas_status
    rocblas_trtri<float>(rocblas_handle handle,
        rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        float *A, rocblas_int lda,
        float *invA, rocblas_int ldinvA)
    {
        return rocblas_strtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }

    template<>
    rocblas_status
    rocblas_trtri<double>(rocblas_handle handle,
        rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        double *A, rocblas_int lda,
        double *invA, rocblas_int ldinvA)
    {
        return rocblas_dtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }


    template<>
    rocblas_status
    rocblas_trtri_batched<float>(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_diagonal diag,
    rocblas_int n,
    float *A, rocblas_int lda, rocblas_int bsa,
    float *invA, rocblas_int ldinvA, rocblas_int bsinvA,
    rocblas_int batch_count)
    {
        return rocblas_strtri_batched(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }

    template<>
    rocblas_status
    rocblas_trtri_batched<double>(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_diagonal diag,
    rocblas_int n,
    double *A, rocblas_int lda, rocblas_int bsa,
    double *invA, rocblas_int ldinvA, rocblas_int bsinvA,
    rocblas_int batch_count)
    {
        return rocblas_dtrtri_batched(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }

    //



#endif  // _ROCBLAS_HPP_
