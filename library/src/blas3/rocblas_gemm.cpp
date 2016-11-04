/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

 
#include "rocblas.h"
#include "rocblas.hpp"


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
rocblas_gemm<rocblas_float_complex>(rocblas_handle handle,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int M, rocblas_int N, rocblas_int K,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta,
    rocblas_float_complex *C, rocblas_int ldc)
{
    return rocblas_cgemm(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


template<>
rocblas_status
rocblas_gemm<rocblas_double_complex>(rocblas_handle handle,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int M, rocblas_int N, rocblas_int K,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta,
    rocblas_double_complex *C, rocblas_int ldc)
{
    return rocblas_zgemm(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

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
    bsa       rocblas_int
              stride from the start of one "A" matrix to the next
    @param[in]
    B         pointer storing matrices of "B" on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of "B".
    @param[in]
    bsb       rocblas_int
              stride from the start of one "B" matrix to the next
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing matrices of "C" on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of "C".
    @param[in]
    bsc       rocblas_int
              stride from the start of one "C" matrix to the next
    @param[in]
    batch_count
              rocblas_int
              number of gemm's in the batch

    ********************************************************************/

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
rocblas_gemm_batched<rocblas_rocblas_double_complex_complex>(rocblas_handle handle,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int M, rocblas_int N, rocblas_int K,
    const rocblas_rocblas_double_complex_complex *alpha,
    const rocblas_rocblas_double_complex_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_rocblas_double_complex_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_rocblas_double_complex_complex *beta,
    rocblas_rocblas_double_complex_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count)
{
    return rocblas_cgemm_batched(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc, batch_count);
}

template<>
rocblas_status
rocblas_gemm_batched<rocblas_double_complex>(rocblas_handle handle,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int M, rocblas_int N, rocblas_int K,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_double_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_double_complex *beta,
    rocblas_double_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count)
{
    return rocblas_zgemm_batched(handle, rocblas_order_column_major, transA, transB, M, N, K, alpha, A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc, batch_count);
}
