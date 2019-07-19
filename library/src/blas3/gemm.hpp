/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _GEMM_HPP_
#define _GEMM_HPP_
#include "rocblas.h"

template <typename T>
rocblas_status rocblas_gemm_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const T*          alpha,
                                     const T*          A,
                                     rocblas_int       lda,
                                     const T*          B,
                                     rocblas_int       ldb,
                                     const T*          beta,
                                     T*                C,
                                     rocblas_int       ldc);

template <typename T>
rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle    handle,
                                                     rocblas_operation transA,
                                                     rocblas_operation transB,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     rocblas_int       k,
                                                     const T*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_int       stride_a,
                                                     const T*          B,
                                                     rocblas_int       ldb,
                                                     rocblas_int       stride_b,
                                                     const T*          beta,
                                                     T*                C,
                                                     rocblas_int       ldc,
                                                     rocblas_int       stride_c,
                                                     rocblas_int       batch_count);

#undef COMPLEX

/* ============================================================================================ */

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call GEMM C interfaces (see gemm.cpp in the same dir)
 * ===========================================================================
 */

template <>
inline rocblas_status rocblas_gemm_template(rocblas_handle      handle,
                                            rocblas_operation   transA,
                                            rocblas_operation   transB,
                                            rocblas_int         M,
                                            rocblas_int         N,
                                            rocblas_int         K,
                                            const rocblas_half* alpha,
                                            const rocblas_half* A,
                                            rocblas_int         lda,
                                            const rocblas_half* B,
                                            rocblas_int         ldb,
                                            const rocblas_half* beta,
                                            rocblas_half*       C,
                                            rocblas_int         ldc)
{
    return rocblas_hgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline rocblas_status rocblas_gemm_template(rocblas_handle    handle,
                                            rocblas_operation transA,
                                            rocblas_operation transB,
                                            rocblas_int       M,
                                            rocblas_int       N,
                                            rocblas_int       K,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            const float*      B,
                                            rocblas_int       ldb,
                                            const float*      beta,
                                            float*            C,
                                            rocblas_int       ldc)
{
    return rocblas_sgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline rocblas_status rocblas_gemm_template(rocblas_handle    handle,
                                            rocblas_operation transA,
                                            rocblas_operation transB,
                                            rocblas_int       M,
                                            rocblas_int       N,
                                            rocblas_int       K,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            const double*     B,
                                            rocblas_int       ldb,
                                            const double*     beta,
                                            double*           C,
                                            rocblas_int       ldc)
{
    return rocblas_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#if COMPLEX

template <>
inline rocblas_status rocblas_gemm_template(rocblas_handle              handle,
                                            rocblas_operation           transA,
                                            rocblas_operation           transB,
                                            rocblas_int                 M,
                                            rocblas_int                 N,
                                            rocblas_int                 K,
                                            const rocblas_half_complex* alpha,
                                            const rocblas_half_complex* A,
                                            rocblas_int                 lda,
                                            const rocblas_half_complex* B,
                                            rocblas_int                 ldb,
                                            const rocblas_half_complex* beta,
                                            rocblas_half_complex*       C,
                                            rocblas_int                 ldc)
{
    return rocblas_qgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline rocblas_status rocblas_gemm_template(rocblas_handle               handle,
                                            rocblas_operation            transA,
                                            rocblas_operation            transB,
                                            rocblas_int                  M,
                                            rocblas_int                  N,
                                            rocblas_int                  K,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* B,
                                            rocblas_int                  ldb,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc)
{
    return rocblas_cgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline rocblas_status rocblas_gemm_template(rocblas_handle                handle,
                                            rocblas_operation             transA,
                                            rocblas_operation             transB,
                                            rocblas_int                   M,
                                            rocblas_int                   N,
                                            rocblas_int                   K,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* B,
                                            rocblas_int                   ldb,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc)
{
    return rocblas_zgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif
/* ============================================================================================ */

template <>
inline rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle      handle,
                                                            rocblas_operation   transA,
                                                            rocblas_operation   transB,
                                                            rocblas_int         M,
                                                            rocblas_int         N,
                                                            rocblas_int         K,
                                                            const rocblas_half* alpha,
                                                            const rocblas_half* A,
                                                            rocblas_int         lda,
                                                            rocblas_int         stride_a,
                                                            const rocblas_half* B,
                                                            rocblas_int         ldb,
                                                            rocblas_int         stride_b,
                                                            const rocblas_half* beta,
                                                            rocblas_half*       C,
                                                            rocblas_int         ldc,
                                                            rocblas_int         stride_c,
                                                            rocblas_int         batch_count)
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
inline rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       M,
                                                            rocblas_int       N,
                                                            rocblas_int       K,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_int       stride_a,
                                                            const float*      B,
                                                            rocblas_int       ldb,
                                                            rocblas_int       stride_b,
                                                            const float*      beta,
                                                            float*            C,
                                                            rocblas_int       ldc,
                                                            rocblas_int       stride_c,
                                                            rocblas_int       batch_count)
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
inline rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       M,
                                                            rocblas_int       N,
                                                            rocblas_int       K,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_int       stride_a,
                                                            const double*     B,
                                                            rocblas_int       ldb,
                                                            rocblas_int       stride_b,
                                                            const double*     beta,
                                                            double*           C,
                                                            rocblas_int       ldc,
                                                            rocblas_int       stride_c,
                                                            rocblas_int       batch_count)
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
inline rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle              handle,
                                                            rocblas_operation           transA,
                                                            rocblas_operation           transB,
                                                            rocblas_int                 M,
                                                            rocblas_int                 N,
                                                            rocblas_int                 K,
                                                            const rocblas_half_complex* alpha,
                                                            const rocblas_half_complex* A,
                                                            rocblas_int                 lda,
                                                            rocblas_int                 stride_a,
                                                            const rocblas_half_complex* B,
                                                            rocblas_int                 ldb,
                                                            rocblas_int                 stride_b,
                                                            const rocblas_half_complex* beta,
                                                            rocblas_half_complex*       C,
                                                            rocblas_int                 ldc,
                                                            rocblas_int                 stride_c,
                                                            rocblas_int                 batch_count)
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
inline rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle               handle,
                                                            rocblas_operation            transA,
                                                            rocblas_operation            transB,
                                                            rocblas_int                  M,
                                                            rocblas_int                  N,
                                                            rocblas_int                  K,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_int                  stride_a,
                                                            const rocblas_float_complex* B,
                                                            rocblas_int                  ldb,
                                                            rocblas_int                  stride_b,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_int                  stride_c,
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
inline rocblas_status rocblas_gemm_strided_batched_template(rocblas_handle                handle,
                                                            rocblas_operation             transA,
                                                            rocblas_operation             transB,
                                                            rocblas_int                   M,
                                                            rocblas_int                   N,
                                                            rocblas_int                   K,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_int                   stride_a,
                                                            const rocblas_double_complex* B,
                                                            rocblas_int                   ldb,
                                                            rocblas_int                   stride_b,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_int                   stride_c,
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
