/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 * \brief rocblas-complex-types.h defines complex data types used by rocblas
 */

#ifndef _ROCBLAS_COMPLEX_TYPES_H_
#define _ROCBLAS_COMPLEX_TYPES_H_

#if __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__))

// If this is a C compiler, C++ compiler below C++14, or a host-only compiler, we only
// include minimal definitions of rocblas_float_complex and rocblas_double_complex

/*! \brief Struct to represent a complex number with single precision real and imaginary parts. */
typedef struct
{
    float x, y;
} rocblas_float_complex;

/*! \brief Struct to represent a complex number with double precision real and imaginary parts. */
typedef struct
{
    double x, y;
} rocblas_double_complex;

#else // __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__))

// If this a full internal build, we need full support of complex arithmetic
// and classes. We need __host__ and __device__ so we use <hip/hip_runtime.h>.

#include <complex>
#include <hip/hip_runtime.h>
#include <math.h>
#include <ostream>
#include <type_traits>

/*! \brief rocblas_complex_num is a structure which represents a complex number
 *         with precision T.
 */
template <typename T>
class rocblas_complex_num
{
    T x; // The real part of the number.
    T y; // The imaginary part of the number.

    // Internal real absolute function, to be sure we're on both device and host
    static __forceinline__ __device__ __host__ T abs(T x)
    {
        return x < 0 ? -x : x;
    }

    static __forceinline__ __device__ __host__ float sqrt(float x)
    {
        return ::sqrtf(x);
    }

    static __forceinline__ __device__ __host__ double sqrt(double x)
    {
        return ::sqrt(x);
    }

public:
    // We do not initialize the members x or y by default, to ensure that it can
    // be used in __shared__ and that it is a trivial class compatible with C.
    __device__ __host__ rocblas_complex_num()                           = default;
    __device__ __host__ rocblas_complex_num(const rocblas_complex_num&) = default;
    __device__ __host__ rocblas_complex_num(rocblas_complex_num&&)      = default;
    __device__ __host__ rocblas_complex_num& operator=(const rocblas_complex_num& rhs) = default;
    __device__ __host__ rocblas_complex_num& operator=(rocblas_complex_num&& rhs) = default;
    __device__                               __host__ ~rocblas_complex_num()      = default;
    using value_type                                                              = T;

    // Constructor
    __device__ __host__ rocblas_complex_num(T r, T i)
        : x{r}
        , y{i}
    {
    }

    // Conversion from real
    __device__ __host__ rocblas_complex_num(T r)
        : x{r}
        , y{0}
    {
    }

    // Conversion from std::complex<T>
    __device__ __host__ rocblas_complex_num(const std::complex<T>& z)
        : x{reinterpret_cast<const T (&)[2]>(z)[0]}
        , y{reinterpret_cast<const T (&)[2]>(z)[1]}
    {
    }

    // Conversion to std::complex<T>
    __device__ __host__ operator std::complex<T>() const
    {
        return {x, y};
    }

    // Conversion from different complex (explicit)
    template <typename U, typename std::enable_if<std::is_constructible<T, U>{}, int>::type = 0>
    __device__ __host__ explicit rocblas_complex_num(const rocblas_complex_num<U>& z)
        : x(z.x)
        , y(z.y)
    {
    }

    // Conversion to bool
    __device__ __host__ explicit operator bool() const
    {
        return x || y;
    }

    // Accessors
    friend __device__ __host__ T std::real(const rocblas_complex_num& z);
    friend __device__ __host__ T std::imag(const rocblas_complex_num& z);

    // stream output
    friend auto& operator<<(std::ostream& out, const rocblas_complex_num& z)
    {
        return out << '(' << z.x << ',' << z.y << ')';
    }

    // complex-real operations
    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    __device__ __host__ auto& operator+=(const U& rhs)
    {
        return (x += T(rhs)), *this;
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    __device__ __host__ auto& operator-=(const U& rhs)
    {
        return (x -= T(rhs)), *this;
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    __device__ __host__ auto& operator*=(const U& rhs)
    {
        return (x *= rhs), (y *= T(rhs)), *this;
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    __device__ __host__ auto& operator/=(const U& rhs)
    {
        return (x /= T(rhs)), (y /= T(rhs)), *this;
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    __device__ __host__ bool operator==(const U& rhs) const
    {
        return x == T(rhs) && y == 0;
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    __device__ __host__ bool operator!=(const U& rhs) const
    {
        return !(*this == rhs);
    }

    // Increment and decrement
    __device__ __host__ auto& operator++()
    {
        return ++x, *this;
    }

    __device__ __host__ rocblas_complex_num operator++(int)
    {
        return {x++, y};
    }

    __device__ __host__ auto& operator--()
    {
        return --x, *this;
    }

    __device__ __host__ rocblas_complex_num operator--(int)
    {
        return {x--, y};
    }

    // Unary operations
    __device__ __host__ rocblas_complex_num operator-() const
    {
        return {-x, -y};
    }

    __device__ __host__ rocblas_complex_num operator+() const
    {
        return *this;
    }

    friend __device__ __host__ T asum(const rocblas_complex_num& z)
    {
        return abs(z.x) + abs(z.y);
    }

    friend __device__ __host__ rocblas_complex_num std::conj(const rocblas_complex_num& z);
    friend __device__ __host__ T                   std::norm(const rocblas_complex_num& z);
    friend __device__ __host__ T                   std::abs(const rocblas_complex_num<T>& z);

    // in-place complex-complex operations
    __device__ __host__ auto& operator*=(const rocblas_complex_num& rhs)
    {
        return *this = {x * rhs.x - y * rhs.y, y * rhs.x + x * rhs.y};
    }

    __device__ __host__ auto& operator+=(const rocblas_complex_num& rhs)
    {
        return *this = {x + rhs.x, y + rhs.y};
    }

    __device__ __host__ auto& operator-=(const rocblas_complex_num& rhs)
    {
        return *this = {x - rhs.x, y - rhs.y};
    }

    __device__ __host__ auto& operator/=(const rocblas_complex_num& rhs)
    {
        if(abs(rhs.x) > abs(rhs.y))
        {
            T ratio = rhs.y / rhs.x;
            T scale = 1 / (rhs.x + rhs.y * ratio);
            *this   = {(x + y * ratio) * scale, (y - x * ratio) * scale};
        }
        else
        {
            T ratio = rhs.x / rhs.y;
            T scale = 1 / (rhs.x * ratio + rhs.y);
            *this   = {(y + x * ratio) * scale, (y * ratio - x) * scale};
        }
        return *this;
    }

    // out-of-place complex-complex operations
    __device__ __host__ auto operator+(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ auto operator-(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs -= rhs;
    }

    __device__ __host__ auto operator*(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs *= rhs;
    }

    __device__ __host__ auto operator/(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs /= rhs;
    }

    __device__ __host__ bool operator==(const rocblas_complex_num& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ bool operator!=(const rocblas_complex_num& rhs) const
    {
        return !(*this == rhs);
    }

    // real-complex operations (complex-real is handled above)
    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    friend __device__ __host__ rocblas_complex_num operator+(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        return {T(lhs) + rhs.x, rhs.y};
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    friend __device__ __host__ rocblas_complex_num operator-(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        return {T(lhs) - rhs.x, -rhs.y};
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    friend __device__ __host__ rocblas_complex_num operator*(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        return {T(lhs) * rhs.x, T(lhs) * rhs.y};
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    friend __device__ __host__ rocblas_complex_num operator/(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        if(abs(rhs.x) > abs(rhs.y))
        {
            T ratio = rhs.y / rhs.x;
            T scale = T(lhs) / (rhs.x + rhs.y * ratio);
            return {scale, -scale * ratio};
        }
        else
        {
            T ratio = rhs.x / rhs.y;
            T scale = T(lhs) / (rhs.x * ratio + rhs.y);
            return {ratio * scale, -scale};
        }
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    friend __device__ __host__ bool operator==(const U& lhs, const rocblas_complex_num& rhs)
    {
        return T(lhs) == rhs.x && 0 == rhs.y;
    }

    template <typename U, typename std::enable_if<std::is_convertible<U, T>{}, int>::type = 0>
    friend __device__ __host__ bool operator!=(const U& lhs, const rocblas_complex_num& rhs)
    {
        return !(lhs == rhs);
    }
};

// Inject standard functions into namespace std
namespace std
{
    template <typename T>
    __device__ __host__ inline T real(const rocblas_complex_num<T>& z)
    {
        return z.x;
    }

    template <typename T>
    __device__ __host__ inline T imag(const rocblas_complex_num<T>& z)
    {
        return z.y;
    }

    template <typename T>
    __device__ __host__ inline rocblas_complex_num<T> conj(const rocblas_complex_num<T>& z)
    {
        return {z.x, -z.y};
    }

    template <typename T>
    __device__ __host__ inline T norm(const rocblas_complex_num<T>& z)
    {
        return (z.x * z.x) + (z.y * z.y);
    }

    template <typename T>
    __device__ __host__ inline T abs(const rocblas_complex_num<T>& z)
    {
        T tr = rocblas_complex_num<T>::abs(z.x), ti = rocblas_complex_num<T>::abs(z.y);
        return tr > ti ? (ti /= tr, tr * rocblas_complex_num<T>::sqrt(ti * ti + 1))
                       : ti ? (tr /= ti, ti * rocblas_complex_num<T>::sqrt(tr * tr + 1)) : 0;
    }
}

// Test for C compatibility
template <typename T>
class rocblas_complex_num_check
{
    static_assert(
        std::is_standard_layout<rocblas_complex_num<T>>{},
        "rocblas_complex_num<T> is not a standard layout type, and thus is incompatible with C.");

    static_assert(std::is_trivial<rocblas_complex_num<T>>{},
                  "rocblas_complex_num<T> is not a trivial type, and thus is incompatible with C.");

    static_assert(
        sizeof(rocblas_complex_num<T>) == 2 * sizeof(T),
        "rocblas_complex_num<T> is not the correct size, and thus is incompatible with C.");
};

template class rocblas_complex_num_check<float>;
template class rocblas_complex_num_check<double>;

// rocBLAS complex data types
using rocblas_float_complex  = rocblas_complex_num<float>;
using rocblas_double_complex = rocblas_complex_num<double>;

/*! \brief is_complex<T> returns true iff T is complex */
template <typename T>
static constexpr bool is_complex = false;

template <>
static constexpr bool is_complex<rocblas_float_complex> = true;

template <>
static constexpr bool is_complex<rocblas_double_complex> = true;

#endif // __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif
