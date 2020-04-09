/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_INIT_H_
#define ROCBLAS_INIT_H_

#include "../../library/src/include/rocblas_ostream.hpp"
#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include <cinttypes>
#include <iostream>
#include <vector>

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
void rocblas_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

// Initialize vector with random values
template <typename T>
inline void
    rocblas_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

template <typename T>
void rocblas_init_sin(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = sin(i + j * lda + i_batch * stride);
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
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

template <typename T>
void rocblas_init_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

template <typename T>
void rocblas_init_cos(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = cos(i + j * lda + i_batch * stride);
}

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
void rocblas_init_symmetric(std::vector<T>& A, size_t N, size_t lda)
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

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
void rocblas_init_hermitian(std::vector<T>& A, size_t N, size_t lda)
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
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
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
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_nan(A.data(), M, N, lda, stride, batch_count);
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
    std::vector<T>& A, size_t M, size_t N, size_t batch_count, size_t lda, size_t stride_a)
{
    if(N % 4 != 0)
        rocblas_cerr << "ERROR: dimension must be a multiple of 4 in order to pack" << std::endl;

    std::vector<T> temp(A);
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
void rocblas_packInt8(std::vector<T>& A, size_t M, size_t N, size_t lda)
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
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                B[i + j * ldb + i_batch * strideb] = A[i + j * lda + i_batch * stridea];
}

#endif
