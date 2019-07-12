/**
 * MIT License
 *
 * Copyright 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*!\file
 * \brief rocblas_complex.h provides struct for rocblas_complex typedef
 */

#pragma once
#ifndef _ROCBLAS_COMPLEX_H_
#define _ROCBLAS_COMPLEX_H_

#ifndef _ROCBLAS_INTERNAL_COMPLEX_
// complex types
typedef struct
{
    float x;
    float y;
} rocblas_float_complex;

typedef struct
{
    double x;
    double y;
} rocblas_double_complex;

#else // _ROCBLAS_INTERNAL_COMPLEX_

#include <hip/hip_runtime.h>
#include <iostream>

// complex types
typedef struct
{
    float x;
    float y;
} rocblas_float_complex_public;

typedef struct
{
    double x;
    double y;
} rocblas_double_complex_public;

template <typename T>
struct rocblas_complex
{
    T x;
    T y;

    __host__ __device__ rocblas_complex() = default;

    constexpr __host__ __device__ rocblas_complex(T real, T imag = 0)
        : x{real}
        , y{imag}
    {
    }

    template <typename U, typename std::enable_if<std::is_arithmetic<U>{}>::type* = nullptr>
    rocblas_complex<T>& operator=(const U a)
    {
        x = a;
        y = 0;
        return (*this);
    }
};

using rocblas_float_complex = rocblas_complex<float>;

static_assert(std::is_standard_layout<rocblas_float_complex>{},
              "rocblas_float_complex is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<rocblas_float_complex>{},
              "rocblas_float_complex is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(rocblas_float_complex) == sizeof(rocblas_float_complex_public)
                  && offsetof(rocblas_float_complex, x) == offsetof(rocblas_float_complex_public, x)
                  && offsetof(rocblas_float_complex, y)
                         == offsetof(rocblas_float_complex_public, y),
              "internal rocblas_float_complex does not match public rocblas_float_complex");

using rocblas_double_complex = rocblas_complex<double>;

static_assert(std::is_standard_layout<rocblas_double_complex>{},
              "rocblas_float_complex is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<rocblas_double_complex>{},
              "rocblas_float_complex is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(rocblas_double_complex) == sizeof(rocblas_double_complex_public)
                  && offsetof(rocblas_double_complex, x)
                         == offsetof(rocblas_double_complex_public, x)
                  && offsetof(rocblas_double_complex, y)
                         == offsetof(rocblas_double_complex_public, y),
              "internal rocblas_float_complex does not match public rocblas_double_complex");

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const rocblas_complex<T>& data)
{
    if(data.y >= 0)
    {
        os << data.x << "+" << data.y << "i";
    }
    else
    {
        os << data.x << data.y << "i";
    }

    return os;
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T> operator+(rocblas_complex<T> data)
{
    return data;
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T> operator-(rocblas_complex<T> data)
{
    return rocblas_complex<T>{-data.x, -data.y};
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T> operator+(rocblas_complex<T> a,
                                                           rocblas_complex<T> b)
{
    return rocblas_complex<T>{a.x + b.x, a.y + b.y};
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T> operator-(rocblas_complex<T> a,
                                                           rocblas_complex<T> b)
{
    return rocblas_complex<T>{a.x - b.x, a.y - b.y};
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T> operator*(rocblas_complex<T> a,
                                                           rocblas_complex<T> b)
{
    return rocblas_complex<T>{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T> operator/(rocblas_complex<T> a,
                                                           rocblas_complex<T> b)
{
    return rocblas_complex<T>{(a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y),
                              (a.y * b.x - a.x * b.y) / (b.x * b.x + b.y * b.y)};
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T>& operator+=(rocblas_complex<T>& a,
                                                             rocblas_complex<T>  b)
{
    return (a = a + b);
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T>& operator-=(rocblas_complex<T>& a,
                                                             rocblas_complex<T>  b)
{
    return (a = a - b);
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T>& operator*=(rocblas_complex<T>& a,
                                                             rocblas_complex<T>  b)
{
    return (a = a * b);
}

template <typename T>
constexpr __host__ __device__ rocblas_complex<T>& operator/=(rocblas_complex<T>& a,
                                                             rocblas_complex<T>  b)
{
    return (a = a / b);
}

#endif // _ROCBLAS_INTERNAL_COMPLEX_

#endif // _ROCBLAS_COMPLEX_H_
