/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../../library/src/include/rocblas_ostream.hpp"
#include "host_vector.hpp"
#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include <cinttypes>
#include <iostream>
#include <omp.h>
#include <vector>

//!
//! @brief enum to check the type of matrix
//!
typedef enum rocblas_check_matrix_type_
{
    // General matrix
    rocblas_client_general_matrix,

    // Hermitian matrix
    rocblas_client_hermitian_matrix,

    // Symmetric matrix
    rocblas_client_symmetric_matrix,

    // Triangular matrix
    rocblas_client_triangular_matrix,

} rocblas_check_matrix_type;

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.

template <typename T>
void rocblas_init_matrix_alternating_sign(rocblas_check_matrix_type matrix_type,
                                          const char                uplo,
                                          T                         rand_gen(),
                                          host_vector<T>&           A,
                                          size_t                    M,
                                          size_t                    N,
                                          size_t                    lda,
                                          rocblas_stride            stride      = 0,
                                          rocblas_int               batch_count = 1)
{
    if(matrix_type == rocblas_client_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value                  = rand_gen();
                    A[i + j * lda + b * stride] = (i ^ j) & 1 ? value : negate(value);
                }
    }
    else if(matrix_type == rocblas_client_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : 0) : (j <= i ? rand_gen() : 0);
                    A[i + j * lda + b * stride] = (i ^ j) & 1 ? T(value) : T(negate(value));
                }
    }
}

template <typename U, typename T>
void rocblas_init_matrix_alternating_sign(rocblas_check_matrix_type matrix_type,
                                          const char                uplo,
                                          T                         rand_gen(),
                                          U&                        hA)
{
    for(rocblas_int batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == rocblas_client_general_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value     = rand_gen();
                    A[i + j * lda] = (i ^ j) & 1 ? value : negate(value);
                }
        }
        else if(matrix_type == rocblas_client_triangular_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : 0) : (j <= i ? rand_gen() : 0);
                    A[i + j * lda] = (i ^ j) & 1 ? value : negate(value);
                }
        }
    }
}

// Initialize vector so adjacent entries have alternating sign.
template <typename T>
void rocblas_init_vector_alternating_sign(T               rand_gen(),
                                          host_vector<T>& x,
                                          size_t          N,
                                          size_t          incx,
                                          rocblas_stride  stride      = 0,
                                          rocblas_int     batch_count = 1)
{
    for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            auto value               = rand_gen();
            x[j * incx + b * stride] = j & 1 ? value : negate(value);
        }
}

/* ============================================================================================ */
/*! \brief  matrix initialization: */
// Initialize matrix with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_matrix(rocblas_check_matrix_type matrix_type,
                         const char                uplo,
                         T                         rand_gen(),
                         host_vector<T>&           A,
                         size_t                    M,
                         size_t                    N,
                         size_t                    lda,
                         rocblas_stride            stride      = 0,
                         rocblas_int               batch_count = 1)
{
    if(matrix_type == rocblas_client_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda + b * stride] = rand_gen();
    }
    else if(matrix_type == rocblas_client_hermitian_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[b * stride + j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = conjugate(value);
                    }
                }
    }
    else if(matrix_type == rocblas_client_symmetric_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[b * stride + j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = value;
                    }
                }
    }
    else if(matrix_type == rocblas_client_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda + b * stride] = value;
                }
    }
}

template <typename U, typename T>
void rocblas_init_matrix(rocblas_check_matrix_type matrix_type,
                         const char                uplo,
                         T                         rand_gen(),
                         U&                        hA)
{
    for(rocblas_int batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();
        if(matrix_type == rocblas_client_general_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda] = rand_gen();
        }
        else if(matrix_type == rocblas_client_hermitian_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = 0;
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = 0;
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = conjugate(value);
                    }
                }
        }
        else if(matrix_type == rocblas_client_symmetric_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = 0;
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = 0;
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == rocblas_client_triangular_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : 0) : (j <= i ? rand_gen() : 0);
                    A[i + j * lda] = value;
                }
        }
    }
}

/*! \brief  vector initialization: */
// Initialize vectors with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_vector(T               rand_gen(),
                         host_vector<T>& x,
                         size_t          N,
                         size_t          incx,
                         rocblas_stride  stride      = 0,
                         rocblas_int     batch_count = 1)
{
    for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
            x[j * incx + b * stride] = rand_gen();
}

/* ============================================================================================ */
/*! \brief  Trigonometric matrix initialization: */
// Initialize matrix with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_matrix_trig(rocblas_check_matrix_type matrix_type,
                              const char                uplo,
                              host_vector<T>&           A,
                              size_t                    M,
                              size_t                    N,
                              size_t                    lda,
                              rocblas_stride            stride      = 0,
                              rocblas_int               batch_count = 1,
                              bool                      seedReset   = false)
{
    if(matrix_type == rocblas_client_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda + b * stride] = T(seedReset ? cos(i + j * lda + b * stride)
                                                              : sin(i + j * lda + b * stride));
    }
    else if(matrix_type == rocblas_client_hermitian_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * lda + b * stride)
                                             : sin(i + j * lda + b * stride));

                    if(i == j)
                        A[b * stride + j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = conjugate(value);
                    }
                }
    }
    else if(matrix_type == rocblas_client_symmetric_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * lda + b * stride)
                                             : sin(i + j * lda + b * stride));
                    if(i == j)
                        A[b * stride + j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = value;
                    }
                }
    }
    else if(matrix_type == rocblas_client_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value                  = uplo == 'U'
                                                      ? (j >= i ? T(seedReset ? cos(i + j * lda + b * stride)
                                                                              : sin(i + j * lda + b * stride))
                                                                : T(0))
                                                      : (j <= i ? T(seedReset ? cos(i + j * lda + b * stride)
                                                                              : sin(i + j * lda + b * stride))
                                                                : T(0));
                    A[i + j * lda + b * stride] = value;
                }
    }
}

template <typename T, typename U>
void rocblas_init_matrix_trig(rocblas_check_matrix_type matrix_type,
                              const char                uplo,
                              U&                        hA,
                              bool                      seedReset = false)
{
    for(rocblas_int batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == rocblas_client_general_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda] = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));
        }
        else if(matrix_type == rocblas_client_hermitian_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));

                    if(i == j)
                        A[j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = 0;
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = 0;
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = conjugate(value);
                    }
                }
        }
        else if(matrix_type == rocblas_client_symmetric_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = 0;
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = 0;
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == rocblas_client_triangular_matrix)
        {
#pragma omp parallel for
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U'
                              ? (j >= i ? T(seedReset ? cos(i + j * lda) : sin(i + j * lda)) : 0)
                              : (j <= i ? T(seedReset ? cos(i + j * lda) : sin(i + j * lda)) : 0);
                    A[i + j * lda] = value;
                }
        }
    }
}

/*! \brief  Trigonometric vector initialization: */
// Initialize vector with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_vector_trig(host_vector<T>& x,
                              size_t          N,
                              size_t          incx,
                              rocblas_stride  stride      = 0,
                              rocblas_int     batch_count = 1,
                              bool            seedReset   = false)
{
    for(size_t b = 0; b < batch_count; b++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
            x[j * incx + b * stride]
                = T(seedReset ? cos(j * incx + b * stride) : sin(j * incx + b * stride));
}

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize matrices with random values
template <typename T>
void rocblas_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t b_idx = i_batch * stride;
        for(size_t j = 0; j < N; ++j)
        {
            size_t col_idx = b_idx + j * lda;
            if(M > 4)
                random_run_generator<T>(A + col_idx, M);
            else
            {
                for(size_t i = 0; i < M; ++i)
                    A[col_idx + i] = random_generator<T>();
            }
        }
    }
}

// Initialize matrices with random values
template <typename T>
inline void rocblas_init(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init(A.data(), M, N, lda, stride, batch_count);
}

template <typename T>
void rocblas_init_sin(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = T(sin(i + offset));
        }
}

template <typename T>
inline void rocblas_init_sin(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_sin(A.data(), M, N, lda, stride, batch_count);
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocblas_init_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : negate(value);
            }
        }
}

template <typename T>
void rocblas_init_alternating_sign(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_alternating_sign(A.data(), M, N, lda, stride, batch_count);
}

// Initialize matrix so adjacent entries have alternating sign.
template <typename T>
void rocblas_init_hpl_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_hpl_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : negate(value);
            }
        }
}

template <typename T>
void rocblas_init_hpl_alternating_sign(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_hpl_alternating_sign(A.data(), M, N, lda, stride, batch_count);
}

template <typename T>
void rocblas_init_cos(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = T(cos(i + offset));
        }
}

template <typename T>
inline void rocblas_init_cos(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_cos(A.data(), M, N, lda, stride, batch_count);
}

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
void rocblas_init_symmetric(host_vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value = random_generator<T>();
            // Warning: It's undefined behavior to assign to the
            // same array element twice in same sequence point (i==j)
            A[j + i * lda] = value;
            A[i + j * lda] = value;
        }
}

/*! \brief  symmetric matrix initialization: */
template <typename T>
void rocblas_init_symmetric(T* A, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t b = 0; b < batch_count; ++b)
    {
        for(size_t i = 0; i < N; ++i)
            for(size_t j = 0; j <= i; ++j)
            {
                auto value = random_generator<T>();
                // Warning: It's undefined behavior to assign to the
                // same array element twice in same sequence point (i==j)
                A[b * stride + j + i * lda] = value;
                A[b * stride + i + j * lda] = value;
            }
    }
}

/*! \brief  symmetric matrix clear: */
template <typename T>
void rocblas_clear_symmetric(
    rocblas_fill uplo, T* A, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t b = 0; b < batch_count; ++b)
    {
        for(size_t i = 0; i < N; ++i)
            for(size_t j = i + 1; j < N; ++j)
            {
                if(uplo == rocblas_fill_upper)
                    A[b * stride + j + i * lda] = 0; // clear lower
                else
                    A[b * stride + i + j * lda] = 0; // clear upper
            }
    }
}

/*! \brief  Hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
void rocblas_init_hermitian(host_vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value     = random_generator<T>();
            A[j + i * lda] = value;
            value.y        = (i == j) ? 0 : negate(value.y);
            A[i + j * lda] = value;
        }
}

// Initialize vector with HPL-like random values
template <typename T>
void rocblas_init_hpl(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

template <typename T>
void rocblas_init_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan_tri(
    bool upper, T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                T val                             = upper ? (j >= i ? T(rocblas_nan_rng()) : 0)
                                                          : (j <= i ? T(rocblas_nan_rng()) : 0);
                A[i + j * lda + i_batch * stride] = val;
            }
}

template <typename T>
void rocblas_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_nan(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with Inf where appropriate */

template <typename T>
void rocblas_init_inf(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_inf_rng());
}

template <typename T>
void rocblas_init_inf(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_inf_rng());
}

template <typename T>
void rocblas_init_inf(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_inf_rng());
}

template <typename T>
void rocblas_init_inf(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_inf(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with zero */

template <typename T>
void rocblas_init_zero(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_zero_rng());
}

template <typename T>
void rocblas_init_zero(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_zero_rng());
}

/* ============================================================================================ */
/*! \brief  Initialize an array with denorm values*/

template <typename T>
void rocblas_init_denorm(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_denorm_rng());
}

template <typename T>
void rocblas_init_denorm(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_denorm_rng());
}

template <typename T>
void rocblas_init_alt_impl_big(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocblas_half ieee_half_max(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_max);
}

template <typename T>
inline void rocblas_init_alt_impl_big(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocblas_half ieee_half_max(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_max);
}

template <typename T>
void rocblas_init_alt_impl_small(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocblas_half ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

template <typename T>
void rocblas_init_alt_impl_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocblas_half ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

/* ============================================================================================ */
/*! \brief  Packs strided_batched matricies into groups of 4 in N */

template <typename T>
void rocblas_packInt8(T* A, const T* temp, size_t M, size_t N, size_t lda)
{
    if(N % 4 != 0)
        rocblas_cerr << "ERROR: dimension must be a multiple of 4 in order to pack" << std::endl;

    for(size_t colBase = 0; colBase < N; colBase += 4)
        for(size_t row = 0; row < lda; row++)
            for(size_t colOffset = 0; colOffset < 4; colOffset++)
                A[(colBase * lda + 4 * row) + colOffset] = temp[(colBase + colOffset) * lda + row];
}

template <typename T>
void rocblas_packInt8(
    host_vector<T>& A, size_t M, size_t N, size_t batch_count, size_t lda, size_t stride_a)
{
    if(N % 4 != 0)
        rocblas_cerr << "ERROR: dimension must be a multiple of 4 in order to pack" << std::endl;

    host_vector<T> temp(A);
    for(size_t count = 0; count < batch_count; count++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * count)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * count)];
}

/* ============================================================================================ */
/*! \brief  Packs matricies into groups of 4 in N */
template <typename T>
void rocblas_packInt8(host_vector<T>& A, size_t M, size_t N, size_t lda)
{
    /* Assumes original matrix provided in column major order, where N is a multiple of 4

        ---------- N ----------
   |  | 00 05 10 15 20 25 30 35      |00 05 10 15|20 25 30 35|
   |  | 01 06 11 16 21 26 31 36      |01 06 11 16|21 26 31 36|
   l  M 02 07 12 17 22 27 32 37  --> |02 07 12 17|22 27 32 37|
   d  | 03 08 13 18 23 28 33 38      |03 08 13 18|23 28 33 38|
   a  | 04 09 14 19 24 29 34 39      |04 09 14 19|24 29 34 39|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|

     Input :  00 01 02 03 04 ** ** 05   ...  38 39 ** **
     Output:  00 05 10 15 01 06 11 16   ...  ** ** ** **

   */

    //  call general code with batch_count = 1 and stride_a = 0
    rocblas_packInt8(A, M, N, 1, lda, 0);
}

/* ============================================================================================ */
/*! \brief  matrix matrix initialization: copies from A into same position in B */
template <typename T>
void rocblas_copy_matrix(const T* A,
                         T*       B,
                         size_t   M,
                         size_t   N,
                         size_t   lda,
                         size_t   ldb,
                         size_t   stridea     = 0,
                         size_t   strideb     = 0,
                         size_t   batch_count = 1)
{

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t stride_offset_a = i_batch * stridea;
        size_t stride_offset_b = i_batch * strideb;
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset_a = stride_offset_a + j * lda;
            size_t offset_b = stride_offset_b + j * ldb;
            memcpy(B + offset_b, A + offset_a, M * sizeof(T));
        }
    }
}

/* ============================================================================================ */
/*! \brief  matrix matrix initialization: copies from A into same position in B */
template <typename T>
void rocblas_copy_matrix(
    const T* const* A, T** B, size_t M, size_t N, size_t lda, size_t ldb, size_t batch_count = 1)
{

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset_a = j * lda;
            size_t offset_b = j * ldb;
            memcpy(B[i_batch] + offset_b, A[i_batch] + offset_a, M * sizeof(T));
        }
    }
}
