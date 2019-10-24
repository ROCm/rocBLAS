/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
#include "rocblas_vector.hpp"

// sqrt(0.5) factor for complex cutoff calculations
constexpr double sqrthalf = 0.7071067811865475244;

// Sum error tolerance for large sums. Multiplied by the number of items
// in the sum to get an expected absolute error bound.

template <class T>
static constexpr double sum_error_tolerance = 0.0;

template <>
static constexpr double sum_error_tolerance<rocblas_half> = 1 / 900.0;

template <>
static constexpr double sum_error_tolerance<rocblas_float_complex> = 1 / 10000.0;

template <>
static constexpr double sum_error_tolerance<rocblas_double_complex> = 1 / 1000000.0;

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)
#else

#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)               \
    do                                                                                          \
    {                                                                                           \
        for(size_t k = 0; k < batch_count; k++)                                                 \
            for(size_t j = 0; j < N; j++)                                                       \
                for(size_t i = 0; i < M; i++)                                                   \
                    NEAR_ASSERT(                                                                \
                        hCPU[i + j * lda + k * strideA], hGPU[i + j * lda + k * strideA], err); \
    } while(0)

#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)            \
    do                                                                                \
    {                                                                                 \
        for(size_t k = 0; k < batch_count; k++)                                       \
            for(size_t j = 0; j < N; j++)                                             \
                for(size_t i = 0; i < M; i++)                                         \
                    if(rocblas_isnan(hCPU[k][i + j * lda]))                           \
                    {                                                                 \
                        ASSERT_TRUE(rocblas_isnan(hGPU[k][i + j * lda]));             \
                    }                                                                 \
                    else                                                              \
                    {                                                                 \
                        NEAR_ASSERT(hCPU[k][i + j * lda], hGPU[k][i + j * lda], err); \
                    }                                                                 \
    } while(0)

#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(half_to_float(a), half_to_float(b), err)

#define NEAR_ASSERT_COMPLEX(a, b, err)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_NEAR(std::real(ta), std::real(tb), err); \
        ASSERT_NEAR(std::imag(ta), std::imag(tb), err); \
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
inline void near_check_general(rocblas_int   M,
                               rocblas_int   N,
                               rocblas_int   lda,
                               rocblas_half* hCPU,
                               rocblas_half* hGPU,
                               double        abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            lda,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU,
                               double                 abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             lda,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU,
                               double                  abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T>
void near_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        rocblas_stride strideA,
                        T*             hCPU,
                        T*             hGPU,
                        double         abs_error);

template <>
inline void near_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               float*         hCPU,
                               float*         hGPU,
                               double         abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               double*        hCPU,
                               double*        hGPU,
                               double         abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               rocblas_half*  hCPU,
                               rocblas_half*  hGPU,
                               double         abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            batch_count,
                               rocblas_int            lda,
                               rocblas_stride         strideA,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU,
                               double                 abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             batch_count,
                               rocblas_int             lda,
                               rocblas_stride          strideA,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU,
                               double                  abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T>
void near_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        host_vector<T> hCPU[],
                        host_vector<T> hGPU[],
                        double         abs_error);

template <>
inline void near_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               batch_count,
                               rocblas_int               lda,
                               host_vector<rocblas_half> hCPU[],
                               host_vector<rocblas_half> hGPU[],
                               double                    abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general(rocblas_int        M,
                               rocblas_int        N,
                               rocblas_int        batch_count,
                               rocblas_int        lda,
                               host_vector<float> hCPU[],
                               host_vector<float> hGPU[],
                               double             abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         batch_count,
                               rocblas_int         lda,
                               host_vector<double> hCPU[],
                               host_vector<double> hGPU[],
                               double              abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

#endif
