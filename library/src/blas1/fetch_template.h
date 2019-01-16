
/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

#ifndef _FETCH_TEMPLATE_
#define _FETCH_TEMPLATE_

#include "rocblas.h"
#include <cmath>

template <typename T>
__device__ __host__ inline T fetch_asum(T A)
{
    return std::abs(A);
}

__device__ __host__ inline float fetch_asum(rocblas_float_complex A)
{
    return std::abs(A.x) + std::abs(A.y);
}

__device__ __host__ inline double fetch_asum(rocblas_double_complex A)
{
    return std::abs(A.x) + std::abs(A.y);
}

template <typename T>
__device__ __host__ inline T fetch_abs2(T A)
{
    return A * A;
}

__device__ __host__ inline float fetch_abs2(rocblas_float_complex A)
{
    return A.x * A.x + A.y * A.y;
}

__device__ __host__ inline double fetch_abs2(rocblas_double_complex A)
{
    return A.x * A.x + A.y * A.y;
}

#endif
