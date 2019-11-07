/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include "cblas_interface.hpp"
#include "rocblas_vector.hpp"
#include "utility.hpp"

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

template <>
void cblas_axpy<rocblas_half>(rocblas_int   n,
                              rocblas_half  alpha,
                              rocblas_half* x,
                              rocblas_int   incx,
                              rocblas_half* y,
                              rocblas_int   incy)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    host_vector<float> x_float(n * abs_incx), y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = x[i * abs_incx];
        y_float[i * abs_incy] = y[i * abs_incy];
    }

    cblas_saxpy(n, alpha, x_float, incx, y_float, incy);

    for(size_t i = 0; i < n; i++)
    {
        x[i * abs_incx] = rocblas_half(x_float[i * abs_incx]);
        y[i * abs_incy] = rocblas_half(y_float[i * abs_incy]);
    }
}

template <>
void cblas_dot<rocblas_half>(rocblas_int         n,
                             const rocblas_half* x,
                             rocblas_int         incx,
                             const rocblas_half* y,
                             rocblas_int         incy,
                             rocblas_half*       result)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    host_vector<float> x_float(n * abs_incx);
    host_vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = x[i * abs_incx];
        y_float[i * abs_incy] = y[i * abs_incy];
    }

    *result = rocblas_half(cblas_sdot(n, x_float, incx, y_float, incy));
}

template <>
void cblas_dot<rocblas_bfloat16>(rocblas_int             n,
                                 const rocblas_bfloat16* x,
                                 rocblas_int             incx,
                                 const rocblas_bfloat16* y,
                                 rocblas_int             incy,
                                 rocblas_bfloat16*       result)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    host_vector<float> x_float(n * abs_incx);
    host_vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = float(x[i * abs_incx]);
        y_float[i * abs_incy] = float(y[i * abs_incy]);
    }

    *result = rocblas_bfloat16(cblas_sdot(n, x_float, incx, y_float, incy));
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

template <>
void cblas_gemm<rocblas_bfloat16, float, float>(rocblas_operation transA,
                                                rocblas_operation transB,
                                                rocblas_int       m,
                                                rocblas_int       n,
                                                rocblas_int       k,
                                                float             alpha,
                                                rocblas_bfloat16* A,
                                                rocblas_int       lda,
                                                rocblas_bfloat16* B,
                                                rocblas_int       ldb,
                                                float             beta,
                                                float*            C,
                                                rocblas_int       ldc)
{
    // cblas does not support rocblas_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA), B_float(sizeB);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<rocblas_bfloat16, rocblas_bfloat16, float>(rocblas_operation transA,
                                                           rocblas_operation transB,
                                                           rocblas_int       m,
                                                           rocblas_int       n,
                                                           rocblas_int       k,
                                                           float             alpha,
                                                           rocblas_bfloat16* A,
                                                           rocblas_int       lda,
                                                           rocblas_bfloat16* B,
                                                           rocblas_int       ldb,
                                                           float             beta,
                                                           rocblas_bfloat16* C,
                                                           rocblas_int       ldc)
{
    // cblas does not support rocblas_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<rocblas_bfloat16>(C_float[i]);
}

template <>
void cblas_gemm<rocblas_half, rocblas_half, float>(rocblas_operation transA,
                                                   rocblas_operation transB,
                                                   rocblas_int       m,
                                                   rocblas_int       n,
                                                   rocblas_int       k,
                                                   float             alpha,
                                                   rocblas_half*     A,
                                                   rocblas_int       lda,
                                                   rocblas_half*     B,
                                                   rocblas_int       ldb,
                                                   float             beta,
                                                   rocblas_half*     C,
                                                   rocblas_int       ldc)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = A[i];
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = B[i];
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = C[i];

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = rocblas_half(C_float[i]);
}

template <>
void cblas_gemm<rocblas_half, rocblas_half, rocblas_half>(rocblas_operation transA,
                                                          rocblas_operation transB,
                                                          rocblas_int       m,
                                                          rocblas_int       n,
                                                          rocblas_int       k,
                                                          rocblas_half      alpha,
                                                          rocblas_half*     A,
                                                          rocblas_int       lda,
                                                          rocblas_half*     B,
                                                          rocblas_int       ldb,
                                                          rocblas_half      beta,
                                                          rocblas_half*     C,
                                                          rocblas_int       ldc)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing
    float alpha_float = alpha;
    float beta_float  = beta;

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = A[i];
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = B[i];
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = C[i];

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha_float,
                A_float,
                lda,
                B_float,
                ldb,
                beta_float,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = rocblas_half(C_float[i]);
}

template <>
void cblas_gemm<int8_t, int32_t, int32_t>(rocblas_operation transA,
                                          rocblas_operation transB,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          int32_t           alpha,
                                          int8_t*           A,
                                          rocblas_int       lda,
                                          int8_t*           B,
                                          rocblas_int       ldb,
                                          int32_t           beta,
                                          int32_t*          C,
                                          rocblas_int       ldc)
{
    // cblas does not support int8_t input / int32_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    size_t const sizeA = ((transA == rocblas_operation_none) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == rocblas_operation_none) ? n : k) * size_t(ldb);
    size_t const sizeC = n * size_t(ldc);

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    cblas_dgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_double,
                lda,
                B_double,
                ldb,
                beta,
                C_double,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<int32_t>(C_double[i]);
}
