/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _ROCBLAS_FLOPS_H_
#define _ROCBLAS_FLOPS_H_

#include "rocblas.h"

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 1, 2,
 * 3. Where possible we are using the values of NOP from the legacy BLAS files [sdcz]blas[23]time.f
 * for flop count.
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
template <>
constexpr double axpy_gflop_count<rocblas_float_complex>(rocblas_int n)
{
    return (8.0 * n) / 1e9; // 6 for complex-complex multiply, 2 for c-c add
}
template <>
constexpr double axpy_gflop_count<rocblas_double_complex>(rocblas_int n)
{
    return (8.0 * n) / 1e9;
}

template <bool CONJ, typename T>
constexpr double dot_gflop_count(rocblas_int n)
{
    return (2.0 * n) / 1e9;
}
template <>
constexpr double dot_gflop_count<false, rocblas_float_complex>(rocblas_int n)
{
    return (8.0 * n) / 1e9; // 6 for each c-c multiply, 2 for each c-c add
}
template <>
constexpr double dot_gflop_count<false, rocblas_double_complex>(rocblas_int n)
{
    return (8.0 * n) / 1e9;
}
template <>
constexpr double dot_gflop_count<true, rocblas_float_complex>(rocblas_int n)
{
    return (9.0 * n) / 1e9; // regular dot (8n) + 1n for complex conjugate
}
template <>
constexpr double dot_gflop_count<true, rocblas_double_complex>(rocblas_int n)
{
    return (9.0 * n) / 1e9;
}

template <typename T, typename U>
constexpr double scal_gflop_count(rocblas_int n)
{
    return (1.0 * n) / 1e9;
}
template <>
constexpr double scal_gflop_count<rocblas_float_complex, rocblas_float_complex>(rocblas_int n)
{
    return (6.0 * n) / 1e9; // 6 for c-c multiply
}
template <>
constexpr double scal_gflop_count<rocblas_double_complex, rocblas_double_complex>(rocblas_int n)
{
    return (6.0 * n) / 1e9;
}
template <>
constexpr double scal_gflop_count<rocblas_float_complex, float>(rocblas_int n)
{
    return (2.0 * n) / 1e9; // 2 for real-complex multiply
}
template <>
constexpr double scal_gflop_count<rocblas_double_complex, double>(rocblas_int n)
{
    return (2.0 * n) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMV */
template <typename T>
constexpr double gemv_gflop_count(rocblas_operation transA, rocblas_int m, rocblas_int n)
{
    return (2.0 * m * n + 2.0 * (transA == rocblas_operation_none ? m : n)) / 1e9;
}
template <>
constexpr double
    gemv_gflop_count<rocblas_float_complex>(rocblas_operation transA, rocblas_int m, rocblas_int n)
{
    return (8.0 * m * n + 6.0 * (transA == rocblas_operation_none ? m : n)) / 1e9;
}

template <>
constexpr double
    gemv_gflop_count<rocblas_double_complex>(rocblas_operation transA, rocblas_int m, rocblas_int n)
{
    return (8.0 * m * n + 6.0 * (transA == rocblas_operation_none ? m : n)) / 1e9;
}

/* \brief floating point counts of TRSV */
template <typename T>
constexpr double trsv_gflop_count(rocblas_int m)
{
    return (m * m) / 1e9;
}

/* \brief floating point counts of SY(HE)MV */
template <typename T>
constexpr double symv_gflop_count(rocblas_int n)
{
    return (2.0 * n * n + 2.0 * n) / 1e9;
}

/* \brief floating point counts of GER */
template <typename T>
constexpr double ger_gflop_count(rocblas_int m, rocblas_int n)
{
    rocblas_int min = (m < n) ? m : n;
    return (2.0 * m * n + min) / 1e9;
}

/* \brief floating point counts of SYR */
template <typename T>
constexpr double syr_gflop_count(rocblas_int n)
{
    return (n * (n + 1) + n) / 1e9;
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
constexpr double trmm_gflop_count(rocblas_int m, rocblas_int n, rocblas_side side)
{
    if(rocblas_side_left == side)
    {
        return (1.0 * m * n * (m + 1)) / 1e9;
    }
    else
    {
        return (1.0 * m * n * (n + 1)) / 1e9;
    }
}

template <>
constexpr double
    trmm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_side side)
{
    if(rocblas_side_left == side)
    {
        return 4 * (1.0 * m * n * (m + 1)) / 1e9;
    }
    else
    {
        return 4 * (1.0 * m * n * (n + 1)) / 1e9;
    }
}

template <>
constexpr double
    trmm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_side side)
{
    if(rocblas_side_left == side)
    {
        return (1.0 * m * n * (m + 1)) / 1e9;
    }
    else
    {
        return (1.0 * m * n * (n + 1)) / 1e9;
    }
}

/* \brief floating point counts of TRSM */
template <typename T>
constexpr double trsm_gflop_count(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (1.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    trsm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (4.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    trsm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (4.0 * m * n * k) / 1e9;
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
