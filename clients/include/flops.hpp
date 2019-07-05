/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _ROCBLAS_FLOPS_H_
#define _ROCBLAS_FLOPS_H_

#include "rocblas.h"

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 1, 2,
 * 3.
 */

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */
template <typename T>
constexpr double axpy_gflop_count(rocblas_int n)
{
    return (2.0 * n) / 1e9;
}
template <typename T>
constexpr double dot_gflop_count(rocblas_int n)
{
    return (2.0 * n) / 1e9;
}
template <typename T>
constexpr double scal_gflop_count(rocblas_int n)
{
    return (1.0 * n) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMV */
template <typename T>
constexpr double gemv_gflop_count(rocblas_int m, rocblas_int n)
{
    return (2.0 * m * n) / 1e9;
}
template <>
constexpr double gemv_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n)
{
    return (8.0 * m * n) / 1e9;
}

template <>
constexpr double gemv_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n)
{
    return (8.0 * m * n) / 1e9;
}

/* \brief floating point counts of TRSV */
template <typename T>
constexpr double trsv_gflop_count(rocblas_int m)
{
    return (m * (m + 1.0)) / 1e9;
}

/* \brief floating point counts of SY(HE)MV */
template <typename T>
constexpr double symv_gflop_count(rocblas_int n)
{
    return (2.0 * n * n) / 1e9;
}

/* \brief floating point counts of GER */
template <typename T>
constexpr double ger_gflop_count(rocblas_int m, rocblas_int n)
{
    return (2.0 * m * n) / 1e9;
}

/* \brief floating point counts of SYR */
template <typename T>
constexpr double syr_gflop_count(rocblas_int n)
{
    return ((2.0 * n * (n + 1)) / 2) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (2.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    gemm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (8.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    gemm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (8.0 * m * n * k) / 1e9;
}

/* \brief floating point counts of GEAM */
template <typename T>
constexpr double geam_gflop_count(rocblas_int m, rocblas_int n)
{
    return (3.0 * m * n) / 1e9;
}

/* \brief floating point counts of TRSM */
template <typename T>
constexpr double trsm_gflop_count(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (1.0 * m * n * (k + 1)) / 1e9;
}

template <>
constexpr double
    trsm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (4.0 * m * n * (k + 1)) / 1e9;
}

template <>
constexpr double
    trsm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (4.0 * m * n * (k + 1)) / 1e9;
}

/* \brief floating point counts of TRTRI */
template <typename T>
constexpr double trtri_gflop_count(rocblas_int n)
{
    return (1.0 * n * n * n) / 3e9;
}

template <>
constexpr double trtri_gflop_count<rocblas_float_complex>(rocblas_int n)
{
    return (8.0 * n * n * n) / 3e9;
}

template <>
constexpr double trtri_gflop_count<rocblas_double_complex>(rocblas_int n)
{
    return (8.0 * n * n * n) / 3e9;
}

#endif /* _ROCBLAS_FLOPS_H_ */
