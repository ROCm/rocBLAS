/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "rocblas/rocblas.h"
#include "utility.hpp"

/*!\file
 * \brief provide common solving utilities
 */

/* ============================================================================================= */
/*! \brief For testing purposes, prepares matrix hA for a triangular solve.                      *
 *         Makes hA strictly diagonal dominant (SPD), then calculates Cholesky factorization     *
 *         of hA.                                                                                */
template <typename T>
void prepare_triangular_solve(T* hA, rocblas_int lda, T* AAT, rocblas_int N, char char_uplo)
{
    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T, T>(rocblas_operation_none,
                     rocblas_operation_conjugate_transpose,
                     N,
                     N,
                     N,
                     T(1.0),
                     hA,
                     lda,
                     hA,
                     lda,
                     T(0.0),
                     AAT,
                     lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += rocblas_abs(AAT[i + j * lda]);
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, N, hA, lda);
}
