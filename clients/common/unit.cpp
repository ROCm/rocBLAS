/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <iostream>
#include "rocblas.h"
#include "unit.h"

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_half* hCPU, rocblas_half* hGPU)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            float cpu_float = static_cast<float>(hCPU[i + j * lda]);
            float gpu_float = static_cast<float>(hGPU[i + j * lda]);
            ASSERT_FLOAT_EQ(cpu_float, gpu_float);
#endif
        }
    }
}

template <>
void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int lda,
                        rocblas_float_complex* hCPU,
                        rocblas_float_complex* hGPU)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].x, hGPU[i + j * lda].x);
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].y, hGPU[i + j * lda].y);
#endif
        }
    }
}

template <>
void unit_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int lda,
                        rocblas_double_complex* hCPU,
                        rocblas_double_complex* hGPU)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].x, hGPU[i + j * lda].x);
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].y, hGPU[i + j * lda].y);
#endif
        }
    }
}

template <>
void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_int* hCPU, rocblas_int* hGPU)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#endif
        }
    }
}

/* ========================================Gtest Unit Check TRSM
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void trsm_err_res_check(float max_error, rocblas_int M, float forward_tolerance, float eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

template <>
void trsm_err_res_check(double max_error, rocblas_int M, double forward_tolerance, double eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

template <>
void potf2_err_res_check(float max_error, rocblas_int N, float forward_tolerance, float eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * N);
#endif
}

template <>
void potf2_err_res_check(double max_error, rocblas_int N, double forward_tolerance, double eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * N);
#endif
}