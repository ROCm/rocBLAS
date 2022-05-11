/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#pragma once

#include "../../library/src/include/utility.hpp"
#include "cblas_interface.hpp"
#include "rocblas.h"

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
