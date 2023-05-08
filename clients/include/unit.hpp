/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief gtest unit compare two matrices float/double/complex */

#pragma once

#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"

#ifndef GOOGLE_TEST
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#else

// Also used for vectors with lda used for inc, which may be negative
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ) \
    do                                                                          \
    {                                                                           \
        for(int64_t k = 0; k < batch_count; k++)                                \
            for(int64_t j = 0; j < N; j++)                                      \
            {                                                                   \
                int64_t offset = lda >= 0 ? 0 : int64_t(lda) * (1 - N);         \
                offset += j * int64_t(lda) + k * strideA;                       \
                size_t idx = offset;                                            \
                for(size_t i = 0; i < M; i++)                                   \
                    if(rocblas_isnan(hCPU[i + idx]))                            \
                    {                                                           \
                        ASSERT_TRUE(rocblas_isnan(hGPU[i + idx]));              \
                    }                                                           \
                    else                                                        \
                    {                                                           \
                        UNIT_ASSERT_EQ(hCPU[i + idx], hGPU[i + idx]);           \
                    }                                                           \
            }                                                                   \
    } while(0)

// Also used for vectors with lda used for inc, which may be negative
#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)    \
    do                                                                      \
    {                                                                       \
        for(size_t k = 0; k < batch_count; k++)                             \
            for(int64_t j = 0; j < N; j++)                                  \
            {                                                               \
                int64_t offset = lda >= 0 ? 0 : int64_t(lda) * (1 - N);     \
                offset += j * int64_t(lda);                                 \
                size_t idx = offset;                                        \
                for(size_t i = 0; i < M; i++)                               \
                    if(rocblas_isnan(hCPU[k][i + idx]))                     \
                    {                                                       \
                        ASSERT_TRUE(rocblas_isnan(hGPU[k][i + idx]));       \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        UNIT_ASSERT_EQ(hCPU[k][i + idx], hGPU[k][i + idx]); \
                    }                                                       \
            }                                                               \
    } while(0)

#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
#define ASSERT_BF16_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

// Compare float to rocblas_bfloat16
// Allow the rocblas_bfloat16 to match the rounded or truncated value of float
// Only call ASSERT_FLOAT_EQ with the rounded value if the truncated value does not match
#include <gtest/internal/gtest-internal.h>
#define ASSERT_FLOAT_BF16_EQ(a, b)                                                            \
    do                                                                                        \
    {                                                                                         \
        using testing::internal::FloatingPoint;                                               \
        if(!FloatingPoint<float>(b).AlmostEquals(FloatingPoint<float>(                        \
               rocblas_bfloat16(a, rocblas_bfloat16::rocblas_truncate_t::rocblas_truncate)))) \
            ASSERT_FLOAT_EQ(b, rocblas_bfloat16(a));                                          \
    } while(0)

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

#endif // GOOGLE_TEST

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
void unit_check_general(rocblas_int                    M,
                        rocblas_int                    N,
                        rocblas_int                    lda,
                        const std::remove_cv_t<T_hpa>* hCPU,
                        const T*                       hGPU);

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             lda,
                               const rocblas_bfloat16* hCPU,
                               const rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<rocblas_bfloat16, float>(
    rocblas_int M, rocblas_int N, rocblas_int lda, const float* hCPU, const rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         lda,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const float* hCPU, const float* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const double* hCPU, const double* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                  M,
                               rocblas_int                  N,
                               rocblas_int                  lda,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   lda,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, const rocblas_int* hCPU, const rocblas_int* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(rocblas_int                    M,
                        rocblas_int                    N,
                        rocblas_int                    lda,
                        rocblas_stride                 strideA,
                        const std::remove_cv_t<T_hpa>* hCPU,
                        const T*                       hGPU,
                        rocblas_int                    batch_count);

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             lda,
                               rocblas_stride          strideA,
                               const rocblas_bfloat16* hCPU,
                               const rocblas_bfloat16* hGPU,
                               rocblas_int             batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<rocblas_bfloat16, float>(rocblas_int             M,
                                                        rocblas_int             N,
                                                        rocblas_int             lda,
                                                        rocblas_stride          strideA,
                                                        const float*            hCPU,
                                                        const rocblas_bfloat16* hGPU,
                                                        rocblas_int             batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         lda,
                               rocblas_stride      strideA,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU,
                               rocblas_int         batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               const float*   hCPU,
                               const float*   hGPU,
                               rocblas_int    batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               const double*  hCPU,
                               const double*  hGPU,
                               rocblas_int    batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                  M,
                               rocblas_int                  N,
                               rocblas_int                  lda,
                               rocblas_stride               strideA,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU,
                               rocblas_int                  batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   lda,
                               rocblas_stride                strideA,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU,
                               rocblas_int                   batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int        M,
                               rocblas_int        N,
                               rocblas_int        lda,
                               rocblas_stride     strideA,
                               const rocblas_int* hCPU,
                               const rocblas_int* hGPU,
                               rocblas_int        batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(rocblas_int                                M,
                        rocblas_int                                N,
                        rocblas_int                                lda,
                        const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                        const host_vector<T>                       hGPU[],
                        rocblas_int                                batch_count);

template <>
inline void unit_check_general(rocblas_int                         M,
                               rocblas_int                         N,
                               rocblas_int                         lda,
                               const host_vector<rocblas_bfloat16> hCPU[],
                               const host_vector<rocblas_bfloat16> hGPU[],
                               rocblas_int                         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<rocblas_bfloat16, float>(rocblas_int                         M,
                                                        rocblas_int                         N,
                                                        rocblas_int                         lda,
                                                        const host_vector<float>            hCPU[],
                                                        const host_vector<rocblas_bfloat16> hGPU[],
                                                        rocblas_int batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(rocblas_int                     M,
                               rocblas_int                     N,
                               rocblas_int                     lda,
                               const host_vector<rocblas_half> hCPU[],
                               const host_vector<rocblas_half> hGPU[],
                               rocblas_int                     batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            lda,
                               const host_vector<int> hCPU[],
                               const host_vector<int> hGPU[],
                               rocblas_int            batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(rocblas_int              M,
                               rocblas_int              N,
                               rocblas_int              lda,
                               const host_vector<float> hCPU[],
                               const host_vector<float> hGPU[],
                               rocblas_int              batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               lda,
                               const host_vector<double> hCPU[],
                               const host_vector<double> hGPU[],
                               rocblas_int               batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                              M,
                               rocblas_int                              N,
                               rocblas_int                              lda,
                               const host_vector<rocblas_float_complex> hCPU[],
                               const host_vector<rocblas_float_complex> hGPU[],
                               rocblas_int                              batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                               M,
                               rocblas_int                               N,
                               rocblas_int                               lda,
                               const host_vector<rocblas_double_complex> hCPU[],
                               const host_vector<rocblas_double_complex> hGPU[],
                               rocblas_int                               batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(rocblas_int                          M,
                        rocblas_int                          N,
                        rocblas_int                          lda,
                        const std::remove_cv_t<T_hpa>* const hCPU[],
                        const T* const                       hGPU[],
                        rocblas_int                          batch_count);

template <>
inline void unit_check_general(rocblas_int                   M,
                               rocblas_int                   N,
                               rocblas_int                   lda,
                               const rocblas_bfloat16* const hCPU[],
                               const rocblas_bfloat16* const hGPU[],
                               rocblas_int                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<rocblas_bfloat16, float>(rocblas_int                   M,
                                                        rocblas_int                   N,
                                                        rocblas_int                   lda,
                                                        const float* const            hCPU[],
                                                        const rocblas_bfloat16* const hGPU[],
                                                        rocblas_int                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               lda,
                               const rocblas_half* const hCPU[],
                               const rocblas_half* const hGPU[],
                               rocblas_int               batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int      M,
                               rocblas_int      N,
                               rocblas_int      lda,
                               const int* const hCPU[],
                               const int* const hGPU[],
                               rocblas_int      batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(rocblas_int        M,
                               rocblas_int        N,
                               rocblas_int        lda,
                               const float* const hCPU[],
                               const float* const hGPU[],
                               rocblas_int        batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         lda,
                               const double* const hCPU[],
                               const double* const hGPU[],
                               rocblas_int         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int                        M,
                               rocblas_int                        N,
                               rocblas_int                        lda,
                               const rocblas_float_complex* const hCPU[],
                               const rocblas_float_complex* const hGPU[],
                               rocblas_int                        batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int                         M,
                               rocblas_int                         N,
                               rocblas_int                         lda,
                               const rocblas_double_complex* const hCPU[],
                               const rocblas_double_complex* const hGPU[],
                               rocblas_int                         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
inline void trsm_err_res_check(T max_error, rocblas_int M, T forward_tolerance, T eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

template <typename T, std::enable_if_t<+rocblas_is_complex<T>, int> = 0>
inline void trsm_err_res_check(T max_error, rocblas_int M, T forward_tolerance, T eps)
{
    trsm_err_res_check(std::abs(max_error), M, std::abs(forward_tolerance), std::abs(eps));
}

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
constexpr double get_epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T, std::enable_if_t<+rocblas_is_complex<T>, int> = 0>
constexpr auto get_epsilon()
{
    return get_epsilon<decltype(std::real(T{}))>();
}

template <typename T,
          std::enable_if_t<!std::is_same_v<T, double> && !std::is_same_v<T, rocblas_double_complex>,
                           int> = 0>
inline double trtri_tolerance(rocblas_int N)
{
    // Algorithm propagates results so use N when large
    return (get_epsilon<T>() * std::max(1000.0, double(N)));
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, rocblas_double_complex>,
                           int> = 0>
inline double trtri_tolerance(rocblas_int N)
{
    return (get_epsilon<T>()
            * std::max(10000.0, double(N))); // allow one more decimal place for double
}
