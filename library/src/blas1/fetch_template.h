
/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

#pragma once
#ifndef _FETCH_TEMPLATE_
#define _FETCH_TEMPLATE_

#include "rocblas.h"
#include <cmath>

template <typename T1, typename T2>
__device__ __host__ inline T2 fetch_real(T1 A)
{
    return A;
}

template <typename T1, typename T2>
__device__ __host__ inline T2 fetch_imag(T1 A)
{
    return 0;
}

template <>
__device__ __host__ inline float fetch_real<rocblas_float_complex, float>(rocblas_float_complex A)
{
    return A.x;
}

template <>
__device__ __host__ inline float fetch_imag<rocblas_float_complex, float>(rocblas_float_complex A)
{
    return A.y;
}

template <>
__device__ __host__ inline double
fetch_real<rocblas_double_complex, double>(rocblas_double_complex A)
{
    return A.x;
}

template <>
__device__ __host__ inline double
fetch_imag<rocblas_double_complex, double>(rocblas_double_complex A)
{
    return A.y;
}

template <typename T1, typename T2>
__device__ __host__ inline T2 fetch_asum(T1 A)
{
    return fabs(A);
}

template <>
__device__ __host__ inline float fetch_asum<rocblas_float_complex, float>(rocblas_float_complex A)
{
    return fabs(A.x) + fabs(A.y);
}

template <>
__device__ __host__ inline double fetch_asum<rocblas_double_complex, double>(rocblas_double_complex A)
{
    return fabs(A.x) + fabs(A.y);
}

template <typename T1, typename T2>
__device__ __host__ inline T2 fetch_abs2(T1 A)
{
    return A * A;
}

template <>
__device__ __host__ inline float fetch_abs2<rocblas_float_complex, float>(rocblas_float_complex A)
{
    return A.x * A.x + A.y * A.y;
}

template <>
__device__ __host__ inline double fetch_abs2<rocblas_double_complex, double>(rocblas_double_complex A)
{
    return A.x * A.x + A.y * A.y;
}

#endif
