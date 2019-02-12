/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _GEMM_HPP_
#define _GEMM_HPP_

#include <hip/hip_runtime.h>

template <typename T>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int k,
                                     const T* alpha,
                                     const T* A,
                                     rocblas_int lda,
                                     const T* B,
                                     rocblas_int ldb,
                                     const T* beta,
                                     T* C,
                                     rocblas_int ldc);

template <typename T>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int m,
                                                     rocblas_int n,
                                                     rocblas_int k,
                                                     const T* alpha,
                                                     const T* A,
                                                     rocblas_int lda,
                                                     rocblas_int stride_a,
                                                     const T* B,
                                                     rocblas_int ldb,
                                                     rocblas_int stride_b,
                                                     const T* beta,
                                                     T* C,
                                                     rocblas_int ldc,
                                                     rocblas_int stride_c,
                                                     rocblas_int batch_count);

#define COMPLEX 0

/* ============================================================================================ */

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call GEMM C interfaces (see gemm.cpp in the same dir)
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

template <>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int M,
                                     rocblas_int N,
                                     rocblas_int K,
                                     const rocblas_half* alpha,
                                     const rocblas_half* A,
                                     rocblas_int lda,
                                     const rocblas_half* B,
                                     rocblas_int ldb,
                                     const rocblas_half* beta,
                                     rocblas_half* C,
                                     rocblas_int ldc)
{
    return rocblas_hgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int M,
                                     rocblas_int N,
                                     rocblas_int K,
                                     const float* alpha,
                                     const float* A,
                                     rocblas_int lda,
                                     const float* B,
                                     rocblas_int ldb,
                                     const float* beta,
                                     float* C,
                                     rocblas_int ldc)
{
    return rocblas_sgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int M,
                                     rocblas_int N,
                                     rocblas_int K,
                                     const double* alpha,
                                     const double* A,
                                     rocblas_int lda,
                                     const double* B,
                                     rocblas_int ldb,
                                     const double* beta,
                                     double* C,
                                     rocblas_int ldc)
{
    return rocblas_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#if COMPLEX

template <>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int M,
                                     rocblas_int N,
                                     rocblas_int K,
                                     const rocblas_half_complex* alpha,
                                     const rocblas_half_complex* A,
                                     rocblas_int lda,
                                     const rocblas_half_complex* B,
                                     rocblas_int ldb,
                                     const rocblas_half_complex* beta,
                                     rocblas_half_complex* C,
                                     rocblas_int ldc)
{
    return rocblas_qgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int M,
                                     rocblas_int N,
                                     rocblas_int K,
                                     const rocblas_float_complex* alpha,
                                     const rocblas_float_complex* A,
                                     rocblas_int lda,
                                     const rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     const rocblas_float_complex* beta,
                                     rocblas_float_complex* C,
                                     rocblas_int ldc)
{
    return rocblas_cgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblas_gemm_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int M,
                                     rocblas_int N,
                                     rocblas_int K,
                                     const rocblas_double_complex* alpha,
                                     const rocblas_double_complex* A,
                                     rocblas_int lda,
                                     const rocblas_double_complex* B,
                                     rocblas_int ldb,
                                     const rocblas_double_complex* beta,
                                     rocblas_double_complex* C,
                                     rocblas_int ldc)
{
    return rocblas_zgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif
/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details

    This is the batched verion of xGEMM, each matrix perform a xGEMM operation.
    There are number of batch_count matrices in each pointer.

    each xGEMM performs one of the matrix-matrix operations

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
    A         pointer storing matrices of "A" on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of "A".
    @param[in]
    stride_a       rocblas_int
              stride from the start of one "A" matrix to the next
    @param[in]
    B         pointer storing matrices of "B" on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of "B".
    @param[in]
    stride_b       rocblas_int
              stride from the start of one "B" matrix to the next
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing matrices of "C" on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of "C".
    @param[in]
    stride_c       rocblas_int
              stride from the start of one "C" matrix to the next
    @param[in]
    batch_count
              rocblas_int
              number of gemm's in the batch

    ********************************************************************/

template <>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int M,
                                                     rocblas_int N,
                                                     rocblas_int K,
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
                                                     rocblas_int batch_count)
{
    return rocblas_hgemm_strided_batched(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count);
}

template <>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int M,
                                                     rocblas_int N,
                                                     rocblas_int K,
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
                                                     rocblas_int batch_count)
{
    return rocblas_sgemm_strided_batched(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count);
}

template <>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int M,
                                                     rocblas_int N,
                                                     rocblas_int K,
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
                                                     rocblas_int batch_count)
{
    return rocblas_dgemm_strided_batched(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count);
}

#if COMPLEX

template <>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int M,
                                                     rocblas_int N,
                                                     rocblas_int K,
                                                     const rocblas_half_complex* alpha,
                                                     const rocblas_half_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_int stride_a,
                                                     const rocblas_half_complex* B,
                                                     rocblas_int ldb,
                                                     rocblas_int stride_b,
                                                     const rocblas_half_complex* beta,
                                                     rocblas_half_complex* C,
                                                     rocblas_int ldc,
                                                     rocblas_int stride_c,
                                                     rocblas_int batch_count)
{
    return rocblas_qgemm_strided_batched(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count);
}

template <>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int M,
                                                     rocblas_int N,
                                                     rocblas_int K,
                                                     const rocblas_float_complex* alpha,
                                                     const rocblas_float_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_int stride_a,
                                                     const rocblas_float_complex* B,
                                                     rocblas_int ldb,
                                                     rocblas_int stride_b,
                                                     const rocblas_float_complex* beta,
                                                     rocblas_float_complex* C,
                                                     rocblas_int ldc,
                                                     rocblas_int stride_c,
                                                     rocblas_int batch_count)
{
    return rocblas_cgemm_strided_batched(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count);
}

template <>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int M,
                                                     rocblas_int N,
                                                     rocblas_int K,
                                                     const rocblas_double_complex* alpha,
                                                     const rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_int stride_a,
                                                     const rocblas_double_complex* B,
                                                     rocblas_int ldb,
                                                     rocblas_int stride_b,
                                                     const rocblas_double_complex* beta,
                                                     rocblas_double_complex* C,
                                                     rocblas_int ldc,
                                                     rocblas_int stride_c,
                                                     rocblas_int batch_count)
{
    return rocblas_zgemm_strided_batched(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch_count);
}

#endif

#endif // _GEMM_HPP_
