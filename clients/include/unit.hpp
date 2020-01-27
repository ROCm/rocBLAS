/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
#include "rocblas_vector.hpp"

#ifndef GOOGLE_TEST
#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, UNIT_ASSERT_EQ)
#else
#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)      \
    do                                                                               \
    {                                                                                \
        for(size_t k = 0; k < batch_count; k++)                                      \
            for(size_t j = 0; j < N; j++)                                            \
                for(size_t i = 0; i < M; i++)                                        \
                    if(rocblas_isnan(hCPU[i + j * lda + k * strideA]))               \
                    {                                                                \
                        ASSERT_TRUE(rocblas_isnan(hGPU[i + j * lda + k * strideA])); \
                    }                                                                \
                    else                                                             \
                    {                                                                \
                        UNIT_ASSERT_EQ(hCPU[i + j * lda + k * strideA],              \
                                       hGPU[i + j * lda + k * strideA]);             \
                    }                                                                \
    } while(0)
#define UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, UNIT_ASSERT_EQ)            \
    do                                                                              \
    {                                                                               \
        for(size_t k = 0; k < batch_count; k++)                                     \
            for(size_t j = 0; j < N; j++)                                           \
                for(size_t i = 0; i < M; i++)                                       \
                    if(rocblas_isnan(hCPU[k][i + j * lda]))                         \
                    {                                                               \
                        ASSERT_TRUE(rocblas_isnan(hGPU[k][i + j * lda]));           \
                    }                                                               \
                    else                                                            \
                    {                                                               \
                        UNIT_ASSERT_EQ(hCPU[k][i + j * lda], hGPU[k][i + j * lda]); \
                    }                                                               \
    } while(0)
#endif

#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

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
void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const T* hCPU, const T* hGPU);

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             lda,
                               const rocblas_bfloat16* hCPU,
                               const rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         lda,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const float* hCPU, const float* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const double* hCPU, const double* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                  M,
                               rocblas_int                  N,
                               rocblas_int                  lda,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   lda,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const rocblas_int* hCPU, const rocblas_int* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_EQ);
}

template <typename T>
void unit_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        rocblas_stride strideA,
                        const T*       hCPU,
                        const T*       hGPU);

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             batch_count,
                               rocblas_int             lda,
                               rocblas_stride          strideA,
                               const rocblas_bfloat16* hCPU,
                               const rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         batch_count,
                               rocblas_int         lda,
                               rocblas_stride      strideA,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               const float*   hCPU,
                               const float*   hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               const double*  hCPU,
                               const double*  hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                  M,
                               rocblas_int                  N,
                               rocblas_int                  batch_count,
                               rocblas_int                  lda,
                               rocblas_stride               strideA,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   batch_count,
                               rocblas_int                   lda,
                               rocblas_stride                strideA,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int        M,
                               rocblas_int        N,
                               rocblas_int        batch_count,
                               rocblas_int        lda,
                               rocblas_stride     strideA,
                               const rocblas_int* hCPU,
                               const rocblas_int* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_EQ);
}

template <typename T>
void unit_check_general(rocblas_int          M,
                        rocblas_int          N,
                        rocblas_int          batch_count,
                        rocblas_int          lda,
                        const host_vector<T> hCPU[],
                        const host_vector<T> hGPU[]);

template <>
inline void unit_check_general(rocblas_int                         M,
                               rocblas_int                         N,
                               rocblas_int                         batch_count,
                               rocblas_int                         lda,
                               const host_vector<rocblas_bfloat16> hCPU[],
                               const host_vector<rocblas_bfloat16> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(rocblas_int                     M,
                               rocblas_int                     N,
                               rocblas_int                     batch_count,
                               rocblas_int                     lda,
                               const host_vector<rocblas_half> hCPU[],
                               const host_vector<rocblas_half> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            batch_count,
                               rocblas_int            lda,
                               const host_vector<int> hCPU[],
                               const host_vector<int> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
inline void unit_check_general(rocblas_int              M,
                               rocblas_int              N,
                               rocblas_int              batch_count,
                               rocblas_int              lda,
                               const host_vector<float> hCPU[],
                               const host_vector<float> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               batch_count,
                               rocblas_int               lda,
                               const host_vector<double> hCPU[],
                               const host_vector<double> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                              M,
                               rocblas_int                              N,
                               rocblas_int                              batch_count,
                               rocblas_int                              lda,
                               const host_vector<rocblas_float_complex> hCPU[],
                               const host_vector<rocblas_float_complex> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                               M,
                               rocblas_int                               N,
                               rocblas_int                               batch_count,
                               rocblas_int                               lda,
                               const host_vector<rocblas_double_complex> hCPU[],
                               const host_vector<rocblas_double_complex> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <typename T>
void unit_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        const T* const hCPU[],
                        const T* const hGPU[]);

template <>
inline void unit_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   batch_count,
                               rocblas_int                   lda,
                               const rocblas_bfloat16* const hCPU[],
                               const rocblas_bfloat16* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               batch_count,
                               rocblas_int               lda,
                               const rocblas_half* const hCPU[],
                               const rocblas_half* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int      M,
                               rocblas_int      N,
                               rocblas_int      batch_count,
                               rocblas_int      lda,
                               const int* const hCPU[],
                               const int* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
inline void unit_check_general(rocblas_int        M,
                               rocblas_int        N,
                               rocblas_int        batch_count,
                               rocblas_int        lda,
                               const float* const hCPU[],
                               const float* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         batch_count,
                               rocblas_int         lda,
                               const double* const hCPU[],
                               const double* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                        M,
                               rocblas_int                        N,
                               rocblas_int                        batch_count,
                               rocblas_int                        lda,
                               const rocblas_float_complex* const hCPU[],
                               const rocblas_float_complex* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                         M,
                               rocblas_int                         N,
                               rocblas_int                         batch_count,
                               rocblas_int                         lda,
                               const rocblas_double_complex* const hCPU[],
                               const rocblas_double_complex* const hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <typename T>
inline void trsm_err_res_check(T max_error, rocblas_int M, T forward_tolerance, T eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

#endif
