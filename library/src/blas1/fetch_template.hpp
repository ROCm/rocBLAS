/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

#pragma once

#include "rocblas/rocblas.h"

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

template <typename T, std::enable_if_t<!std::is_same<T, rocblas_half>{}, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return std::norm(A);
}

template <typename T, std::enable_if_t<std::is_same<T, rocblas_half>{}, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return A * A;
}
