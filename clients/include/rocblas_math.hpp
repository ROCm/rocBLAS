/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_MATH_H_
#define ROCBLAS_MATH_H_

#include "rocblas.h"
#include <cmath>
#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <type_traits>

/* ============================================================================================ */
// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline __host__ rocblas_half float_to_half(float val)
{
    return _cvtss_sh(val, 0);
}

// Helper routine to convert halfs into their floats equivalent; uses F16C instructions
inline __host__ float half_to_float(rocblas_half val)
{
    return _cvtsh_ss(val);
}

/* ============================================================================================ */
/*! \brief  returns true if value is NaN */

template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(T)
{
    return false;
}
inline bool rocblas_isnan(double arg)
{
    return std::isnan(arg);
}
inline bool rocblas_isnan(float arg)
{
    return std::isnan(arg);
}
inline bool rocblas_isnan(rocblas_half arg)
{
    return (~arg & 0x7c00) == 0 && (arg & 0x3ff) != 0;
}
template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(const T& arg)
{
    return rocblas_isnan(std::real(arg)) || rocblas_isnan(std::imag(arg));
}

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

template <>
inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}
#endif
