/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <iostream>
#include "rocblas.h"
#include "near.h"

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

/* ========================================Gtest Near Check
 * ==================================================== */

/*! \brief Template: gtest near compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU, float abs_error)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            //              ASSERT_FLOAT_EQ(hCPU[i+j*lda], hGPU[i+j*lda]);
            ASSERT_NEAR(hCPU[i + j * lda], hGPU[i + j * lda], abs_error);
#endif
        }
    }
}

template <>
void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU, double abs_error)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda], hGPU[i + j * lda], abs_error);
#endif
        }
    }
}

template <>
void near_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int lda,
                        rocblas_float_complex* hCPU,
                        rocblas_float_complex* hGPU,
                        float abs_error)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda].x, hGPU[i + j * lda].x, abs_error);
            ASSERT_NEAR(hCPU[i + j * lda].y, hGPU[i + j * lda].y, abs_error);
#endif
        }
    }
}

template <>
void near_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int lda,
                        rocblas_double_complex* hCPU,
                        rocblas_double_complex* hGPU,
                        double abs_error)
{
#pragma unroll
    for(rocblas_int j = 0; j < N; j++)
    {
#pragma unroll
        for(rocblas_int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda].x, hGPU[i + j * lda].x, abs_error);
            ASSERT_NEAR(hCPU[i + j * lda].y, hGPU[i + j * lda].y, abs_error);
#endif
        }
    }
}
