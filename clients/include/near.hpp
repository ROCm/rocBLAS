/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/* =====================================================================
    Google Near check: ASSERT_NEAR( elementof(A), elementof(B))
   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Near check.
 */

#ifndef _NEAR_H
#define _NEAR_H

#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_test.hpp"

// sqrt(0.5) factor for complex cutoff calculations
constexpr double sqrthalf = 0.7071067811865475244;

// Sum error tolerance for large sums. Multiplied by the number of items
// in the sum to get an expected absolute error bound.
template <class T>
constexpr double sum_error_tolerance = 0.0;

template <>
constexpr double sum_error_tolerance<rocblas_half> = 1 / 900.0;

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)
#else
// clang-format off
#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT) \
    do                                                            \
    {                                                             \
        _Pragma("unroll")                                         \
        for(size_t k = 0; k < batch_count; k++)                   \
            for(size_t j = 0; j < N; j++)                         \
                for(size_t i = 0; i < M; i++)                     \
                    NEAR_ASSERT(hCPU[i + j * lda + k * strideA],  \
                                hGPU[i + j * lda + k * strideA],  \
                                err);                             \
    } while(0)
// clang-format on
#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(half_to_float(a), half_to_float(b), err)

#define NEAR_ASSERT_FLOAT_COMPLEX(a, b, err) \
    do                                       \
    {                                        \
        auto ta = (a), tb = (b);             \
        ASSERT_NEAR(ta.x, tb.x, err);        \
        ASSERT_NEAR(ta.y, tb.y, err);        \
    } while(0)

#define NEAR_ASSERT_DOUBLE_COMPLEX(a, b, err) \
    do                                        \
    {                                         \
        auto ta = (a), tb = (b);              \
        ASSERT_NEAR(ta.x, tb.x, err);         \
        ASSERT_NEAR(ta.y, tb.y, err);         \
    } while(0)

template <typename T>
void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU, double abs_error);

template <>
inline void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int lda,
                               rocblas_half* hCPU,
                               rocblas_half* hGPU,
                               double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int lda,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU,
                               double abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_FLOAT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int lda,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU,
                               double abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_DOUBLE_COMPLEX);
}

template <typename T>
void near_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int batch_count,
                        rocblas_int lda,
                        rocblas_int strideA,
                        T* hCPU,
                        T* hGPU,
                        double abs_error);

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               float* hCPU,
                               float* hGPU,
                               double abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               double* hCPU,
                               double* hGPU,
                               double abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               rocblas_half* hCPU,
                               rocblas_half* hGPU,
                               double abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU,
                               double abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_FLOAT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU,
                               double abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_DOUBLE_COMPLEX);
}

#endif
