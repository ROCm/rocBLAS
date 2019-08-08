/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief gtest unit compare two matrices float/double/complex */

#ifndef _UNIT_H
#define _UNIT_H

#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_test.hpp"

#ifndef GOOGLE_TEST
#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)
#else
// clang-format off
#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)      \
    do                                                                               \
    {                                                                                \
        for(size_t k = 0; k < batch_count; k++)                                      \
            for(size_t j = 0; j < N; j++)                                            \
                for(size_t i = 0; i < M; i++)                                        \
                    if (rocblas_isnan(hCPU[i + j * lda + k * strideA])) {            \
                        ASSERT_TRUE(rocblas_isnan(hGPU[i + j * lda + k * strideA])); \
                    } else {                                                         \
                        UNIT_ASSERT_EQ(hCPU[i + j * lda + k * strideA],              \
                                       hGPU[i + j * lda + k * strideA]);             \
                    }                                                                \
    } while(0)
// clang-format on
#endif

#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(half_to_float(a), half_to_float(b))

#define ASSERT_BFLOAT16_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                  \
    do                                                 \
    {                                                  \
        auto ta = (a), tb = (b);                       \
        ASSERT_FLOAT_EQ(std::real(ta), std::real(tb)); \
        ASSERT_FLOAT_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_DOUBLE_EQ(std::real(ta), std::real(tb)); \
        ASSERT_DOUBLE_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

template <typename T>
void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU);

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_bfloat16* hCPU, rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_half* hCPU, rocblas_half* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void
    unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void
    unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            lda,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             lda,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_int* hCPU, rocblas_int* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_EQ);
}

template <typename T>
void unit_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int batch_count,
                        rocblas_int lda,
                        rocblas_int strideA,
                        T*          hCPU,
                        T*          hGPU);

template <>
inline void unit_check_general(rocblas_int       M,
                               rocblas_int       N,
                               rocblas_int       batch_count,
                               rocblas_int       lda,
                               rocblas_int       strideA,
                               rocblas_bfloat16* hCPU,
                               rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(rocblas_int   M,
                               rocblas_int   N,
                               rocblas_int   batch_count,
                               rocblas_int   lda,
                               rocblas_int   strideA,
                               rocblas_half* hCPU,
                               rocblas_half* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               float*      hCPU,
                               float*      hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               double*     hCPU,
                               double*     hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            batch_count,
                               rocblas_int            lda,
                               rocblas_int            strideA,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             batch_count,
                               rocblas_int             lda,
                               rocblas_int             strideA,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int  M,
                               rocblas_int  N,
                               rocblas_int  batch_count,
                               rocblas_int  lda,
                               rocblas_int  strideA,
                               rocblas_int* hCPU,
                               rocblas_int* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_EQ);
}

template <typename T>
inline void trsm_err_res_check(T max_error, rocblas_int M, T forward_tolerance, T eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

#endif
