/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
ROCBLAS_CLANG_STATIC constexpr double sum_error_tolerance<rocblas_bfloat16> = 1 / 100.0;

template <>
ROCBLAS_CLANG_STATIC constexpr double sum_error_tolerance<rocblas_half> = 1 / 900.0;

template <>
ROCBLAS_CLANG_STATIC constexpr double sum_error_tolerance<rocblas_float_complex> = 1 / 10000.0;

template <>
ROCBLAS_CLANG_STATIC constexpr double sum_error_tolerance<rocblas_double_complex> = 1 / 1000000.0;

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#else

#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT)               \
    do                                                                                          \
    {                                                                                           \
        for(size_t k = 0; k < batch_count; k++)                                                 \
            for(size_t j = 0; j < N; j++)                                                       \
                for(size_t i = 0; i < M; i++)                                                   \
                    NEAR_ASSERT(                                                                \
                        hCPU[i + j * lda + k * strideA], hGPU[i + j * lda + k * strideA], err); \
    } while(0)

#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)            \
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

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(double(a), double(b), err)

#define NEAR_ASSERT_BF16(a, b, err) ASSERT_NEAR(double(a), double(b), err)

#define NEAR_ASSERT_COMPLEX(a, b, err)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_NEAR(std::real(ta), std::real(tb), err); \
        ASSERT_NEAR(std::imag(ta), std::imag(tb), err); \
    } while(0)

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
inline void near_check_general(rocblas_int                    M,
                               rocblas_int                    N,
                               rocblas_int                    lda,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         lda,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU,
                               double              abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<rocblas_bfloat16, float>(rocblas_int             M,
                                                        rocblas_int             N,
                                                        rocblas_int             lda,
                                                        const float*            hCPU,
                                                        const rocblas_bfloat16* hGPU,
                                                        double                  abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(rocblas_int                  M,
                               rocblas_int                  N,
                               rocblas_int                  lda,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU,
                               double                       abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   lda,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU,
                               double                        abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(rocblas_int                    M,
                               rocblas_int                    N,
                               rocblas_int                    lda,
                               rocblas_stride                 strideA,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               rocblas_int                    batch_count,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         lda,
                               rocblas_stride      strideA,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU,
                               rocblas_int         batch_count,
                               double              abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<rocblas_bfloat16, float>(rocblas_int             M,
                                                        rocblas_int             N,
                                                        rocblas_int             lda,
                                                        rocblas_stride          strideA,
                                                        const float*            hCPU,
                                                        const rocblas_bfloat16* hGPU,
                                                        rocblas_int             batch_count,
                                                        double                  abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(rocblas_int                  M,
                               rocblas_int                  N,
                               rocblas_int                  lda,
                               rocblas_stride               strideA,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU,
                               rocblas_int                  batch_count,
                               double                       abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   lda,
                               rocblas_stride                strideA,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU,
                               rocblas_int                   batch_count,
                               double                        abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T, typename T_hpa = T>
void near_check_general(rocblas_int                                M,
                        rocblas_int                                N,
                        rocblas_int                                lda,
                        const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                        const host_vector<T>                       hGPU[],
                        rocblas_int                                batch_count,
                        double                                     abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int                     M,
                               rocblas_int                     N,
                               rocblas_int                     lda,
                               const host_vector<rocblas_half> hCPU[],
                               const host_vector<rocblas_half> hGPU[],
                               rocblas_int                     batch_count,
                               double                          abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}
template <>
inline void near_check_general<rocblas_bfloat16, float>(rocblas_int                         M,
                                                        rocblas_int                         N,
                                                        rocblas_int                         lda,
                                                        const host_vector<float>            hCPU[],
                                                        const host_vector<rocblas_bfloat16> hGPU[],
                                                        rocblas_int batch_count,
                                                        double      abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(rocblas_int                              M,
                               rocblas_int                              N,
                               rocblas_int                              lda,
                               const host_vector<rocblas_float_complex> hCPU[],
                               const host_vector<rocblas_float_complex> hGPU[],
                               rocblas_int                              batch_count,
                               double                                   abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int                               M,
                               rocblas_int                               N,
                               rocblas_int                               lda,
                               const host_vector<rocblas_double_complex> hCPU[],
                               const host_vector<rocblas_double_complex> hGPU[],
                               rocblas_int                               batch_count,
                               double                                    abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(rocblas_int                          M,
                               rocblas_int                          N,
                               rocblas_int                          lda,
                               const std::remove_cv_t<T_hpa>* const hCPU[],
                               const T* const                       hGPU[],
                               rocblas_int                          batch_count,
                               double                               abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               lda,
                               const rocblas_half* const hCPU[],
                               const rocblas_half* const hGPU[],
                               rocblas_int               batch_count,
                               double                    abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<rocblas_bfloat16, float>(rocblas_int                   M,
                                                        rocblas_int                   N,
                                                        rocblas_int                   lda,
                                                        const float* const            hCPU[],
                                                        const rocblas_bfloat16* const hGPU[],
                                                        rocblas_int                   batch_count,
                                                        double                        abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(rocblas_int                        M,
                               rocblas_int                        N,
                               rocblas_int                        lda,
                               const rocblas_float_complex* const hCPU[],
                               const rocblas_float_complex* const hGPU[],
                               rocblas_int                        batch_count,
                               double                             abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(rocblas_int                         M,
                               rocblas_int                         N,
                               rocblas_int                         lda,
                               const rocblas_double_complex* const hCPU[],
                               const rocblas_double_complex* const hGPU[],
                               rocblas_int                         batch_count,
                               double                              abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

#endif
