/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_RANDOM_H_
#define ROCBLAS_RANDOM_H_

#include <cinttypes>
#include <type_traits>
#include <random>
#include "rocblas.h"
#include "rocblas_math.hpp"

/* ============================================================================================ */
// Random number generator
using rocblas_rng_t = std::mt19937;
extern rocblas_rng_t rocblas_rng, rocblas_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand() { rocblas_rng = rocblas_seed; }

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */
class rocblas_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union
        {
            UINT_T u;
            T fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG;   // Exponent = all 1's
        return x.fp;                              // NaN with random bits
    }

    public:
    // Random integer
    template <typename T, typename = typename std::enable_if<std::is_integral<T>{}>::type>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(rocblas_rng);
    }

    // Random NaN double
    explicit operator double() { return random_nan_data<double, uint64_t, 52, 11>(); }

    // Random NaN float
    explicit operator float() { return random_nan_data<float, uint32_t, 23, 8>(); }

    // Random NaN half (non-template rocblas_half takes precedence over integer template above)
    explicit operator rocblas_half() { return random_nan_data<rocblas_half, uint16_t, 10, 5>(); }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return std::uniform_int_distribution<int>(1, 10)(rocblas_rng);
}

// for rocblas_half, generate float, and convert to rocblas_half
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return float_to_half(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return std::uniform_int_distribution<int8_t>(1, 3)(rocblas_rng);
};

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng);
}

#endif
