/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potf2.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status roclapack_spotf2(rocblas_handle handle,
                                           rocblas_fill uplo,
                                           rocblas_int n,
                                           float* A,
                                           rocblas_int lda)
{
    return rocblas_potf2_template<float>(
        handle, uplo, n, A, lda);
}

extern "C" rocblas_status roclapack_dpotf2(rocblas_handle handle,
                                           rocblas_fill uplo,
                                           rocblas_int n,
                                           double* A,
                                           rocblas_int lda)
{
    return rocblas_potf2_template<double>(
        handle, uplo, n, A, lda);
}
