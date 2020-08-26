/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

#ifndef _FETCH_TEMPLATE_
#define _FETCH_TEMPLATE_

#include "rocblas.h"

template <typename T>
__device__ __host__ inline auto fetch_asum(T A)
{
    return A < 0 ? -A : A;
}

__device__ __host__ inline auto fetch_asum(rocblas_float_complex A)
{
    return asum(A);
}

__device__ __host__ inline auto fetch_asum(rocblas_double_complex A)
{
    return asum(A);
}

template <typename T>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return std::norm(A);
}

#endif
