/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_MATH_H_
#define ROCBLAS_MATH_H_

#include <cmath>
#include <immintrin.h>
#include "rocblas.h"

/* ============================================================================================ */
// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline rocblas_half float_to_half(float val) { return _cvtss_sh(val, 0); }

// Helper routine to convert halfs into their floats equivalent; uses F16C instructions
inline float half_to_float(rocblas_half val) { return _cvtsh_ss(val); }

/* ============================================================================================ */
/*! \brief  returns true if value is NaN */

template <typename T>
inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <>
inline bool rocblas_isnan(rocblas_half arg)
{
    return (~arg & 0x7c00) == 0 && (arg & 0x3ff) != 0;
}

/* ============================================================================================ */
/*! \brief is_complex<T> returns true iff T is complex */

template <typename>
constexpr bool is_complex = false;

template <>
constexpr bool is_complex<rocblas_double_complex> = true;

template <>
constexpr bool is_complex<rocblas_float_complex> = true;

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half x)
{
    return x ^ 0x8000;
}

#endif
