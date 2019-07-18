/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _NORM_H
#define _NORM_H

#include "rocblas.h"

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

/*! \brief  Template: norm check for general Matrix: half/float/doubel/complex  */

// see check_norm.cpp for template speciliazation
// use auto as the return type is only allowed in c++14
// convert float/float to double
template <typename T>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU);

/*! \brief  Template: norm check for strided_batched Matrix: half/float/double/complex */
template <typename T>
double norm_check_general(char        norm_type,
                          rocblas_int M,
                          rocblas_int N,
                          rocblas_int lda,
                          rocblas_int stride_a,
                          rocblas_int batch_count,
                          T*          hCPU,
                          T*          hGPU);

/*! \brief  Template: norm check for hermitian/symmetric Matrix: half/float/double/complex */

template <typename T>
double norm_check_symmetric(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU);

#endif
