/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _ROCBLAS_BYTES_H_
#define _ROCBLAS_BYTES_H_

#include "rocblas.h"

/*!\file
 * \brief provides bandwidth measure as byte counts Basic Linear Algebra Subprograms (BLAS) of
 * Level 1, 2, 3. Where possible we are using the values of NOP from the legacy BLAS files
 * [sdcz]blas[23]time.f for byte counts.
 */

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

inline size_t tri_count(rocblas_int n)
{
    return size_t(n) * (1 + n) / 2;
}

/* \brief byte counts of SYMV */
template <typename T>
constexpr double symv_gbyte_count(rocblas_int n)
{
    return (sizeof(T) * n * (n + 1)) / 1e9;
}

/* \brief byte counts of SPMV */
template <typename T>
constexpr double spmv_gbyte_count(rocblas_int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of SBMV */
template <typename T>
constexpr double sbmv_gbyte_count(rocblas_int n, rocblas_int k)
{
    rocblas_int k1 = k < n ? k : n - 1;
    return (sizeof(T) * (tri_count(n) - tri_count(n - (k1 + 1)) + n)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

#endif /* _ROCBLAS_BYTES_H_ */
