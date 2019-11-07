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
/*! \brief  returns true if value is NaN */

template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
inline bool rocblas_isnan(T)
{
    return false;
}

template <typename T,
          typename std::enable_if<!std::is_integral<T>{} && !is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(const T& arg)
{
    return rocblas_isnan(std::real(arg)) || rocblas_isnan(std::imag(arg));
}

inline bool rocblas_isnan(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}
#endif
