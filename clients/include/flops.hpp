/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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

/* \brief floating point counts of tpmv */
template <typename T>
constexpr double tpmv_gflop_count(rocblas_int m)
{
    return (m * m) / 1e9;
}

template <>
constexpr double tpmv_gflop_count<rocblas_float_complex>(rocblas_int m)
{
    return (2.0 * m * (2.0 * m + 1.0)) / 1e9;
}

template <>
constexpr double tpmv_gflop_count<rocblas_double_complex>(rocblas_int m)
{
    return (2.0 * m * (2.0 * m + 1.0)) / 1e9;
}

/* \brief floating point counts of trmv */
template <typename T>
constexpr double trmv_gflop_count(rocblas_int m)
{
    return (m * m) / 1e9;
}

template <>
constexpr double trmv_gflop_count<rocblas_float_complex>(rocblas_int m)
{
    return (2.0 * m * (2.0 * m + 1.0)) / 1e9;
}

template <>
constexpr double trmv_gflop_count<rocblas_double_complex>(rocblas_int m)
{
    return (2.0 * m * (2.0 * m + 1.0)) / 1e9;
}

/* \brief floating point counts of GBMV */
template <typename T>
constexpr double gbmv_gflop_count(
    rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku)
{
    rocblas_int dim_x = transA == rocblas_operation_none ? n : m;
    rocblas_int k1    = dim_x < kl ? dim_x : kl;
    rocblas_int k2    = dim_x < ku ? dim_x : ku;

    // kl and ku ops, plus main diagonal ops
    double d1 = ((2 * k1 * dim_x) - (k1 * (k1 + 1))) + dim_x;
    double d2 = ((2 * k2 * dim_x) - (k2 * (k2 + 1))) + 2 * dim_x;

    // add y operations
    return (d1 + d2 + 2 * dim_x) / 1e9;
}

template <>
constexpr double gbmv_gflop_count<rocblas_float_complex>(
    rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku)
{
    rocblas_int dim_x = transA == rocblas_operation_none ? n : m;
    rocblas_int k1    = dim_x < kl ? dim_x : kl;
    rocblas_int k2    = dim_x < ku ? dim_x : ku;

    double d1 = 4 * ((2 * k1 * dim_x) - (k1 * (k1 + 1))) + 6 * dim_x;
    double d2 = 4 * ((2 * k2 * dim_x) - (k2 * (k2 + 1))) + 8 * dim_x;

    return (d1 + d2 + 8 * dim_x) / 1e9;
}

template <>
constexpr double gbmv_gflop_count<rocblas_double_complex>(
    rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku)
{
    rocblas_int dim_x = transA == rocblas_operation_none ? n : m;
    rocblas_int k1    = dim_x < kl ? dim_x : kl;
    rocblas_int k2    = dim_x < ku ? dim_x : ku;

    double d1 = 4 * ((2 * k1 * dim_x) - (k1 * (k1 + 1))) + 6 * dim_x;
    double d2 = 4 * ((2 * k2 * dim_x) - (k2 * (k2 + 1))) + 8 * dim_x;

    return (d1 + d2 + 8 * dim_x) / 1e9;
}

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

/* \brief floating point counts of HBMV */
template <typename T>
constexpr double hbmv_gflop_count(rocblas_int n, rocblas_int k)
{
    rocblas_int k1 = k < n ? k : n;
    return (8.0 * ((2 * k1 + 1) * n - k1 * (k1 + 1)) + 8 * n) / 1e9;
}

/* \brief floating point counts of HEMV */
template <typename T>
constexpr double hemv_gflop_count(rocblas_int n)
{
    return (8.0 * n * n + 8.0 * n) / 1e9;
}

/* \brief floating point counts of HPMV */
template <typename T>
constexpr double hpmv_gflop_count(rocblas_int n)
{
    return (8.0 * n * n + 8.0 * n) / 1e9;
}

/* \brief floating point counts of HPR */
template <typename T>
constexpr double hpr_gflop_count(rocblas_int n)
{
    return (4.0 * n * n + 6.0 * n) / 1e9;
}

/* \brief floating point counts of TRSV */
template <typename T>
constexpr double trsv_gflop_count(rocblas_int m)
{
    return (m * m) / 1e9;
}

/* \brief floating point counts of TBMV */
template <typename T>
constexpr double tbmv_gflop_count(rocblas_int m, rocblas_int k)
{
    rocblas_int k1 = k < m ? k : m;
    return ((2 * m * k1 - k1 * (k1 + 1)) + m) / 1e9;
}

template <>
constexpr double tbmv_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int k)
{
    rocblas_int k1 = k < m ? k : m;
    return (4 * (2 * m * k1 - k1 * (k1 + 1)) + 4 * m) / 1e9;
}

template <>
constexpr double tbmv_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int k)
{
    rocblas_int k1 = k < m ? k : m;
    return (4 * (2 * m * k1 - k1 * (k1 + 1)) + 4 * m) / 1e9;
}

/* \brief floating point counts of SY(HE)MV */
template <typename T>
constexpr double symv_gflop_count(rocblas_int n)
{
    return (2.0 * n * n + 2.0 * n) / 1e9;
}

template <>
constexpr double symv_gflop_count<rocblas_float_complex>(rocblas_int n)
{
    return 4.0 * symv_gflop_count<rocblas_float>(n);
}

template <>
constexpr double symv_gflop_count<rocblas_double_complex>(rocblas_int n)
{
    return symv_gflop_count<rocblas_float_complex>(n);
}

/* \brief floating point counts of SPMV */
template <typename T>
constexpr double spmv_gflop_count(rocblas_int n)
{
    return (2.0 * n * n + 2.0 * n) / 1e9;
}

/* \brief floating point counts of SBMV */
template <typename T>
constexpr double sbmv_gflop_count(rocblas_int n, rocblas_int k)
{
    rocblas_int k1 = k < n ? k : n;
    return (2.0 * ((2.0 * k1 + 1) * n - k1 * (k1 + 1)) + 2.0 * n) / 1e9;
}

/* \brief floating point counts of SPR */
template <typename T>
constexpr double spr_gflop_count(rocblas_int n)
{
    return (double(n) * (n + 1.0) + n) / 1e9;
}

template <>
constexpr double spr_gflop_count<rocblas_float_complex>(rocblas_int n)
{
    return (6.0 * n + 4.0 * n * (n + 1.0)) / 1e9;
}

template <>
constexpr double spr_gflop_count<rocblas_double_complex>(rocblas_int n)
{
    return spr_gflop_count<rocblas_float_complex>(n);
}

/* \brief floating point counts of SPR2 */
template <typename T>
constexpr double spr2_gflop_count(rocblas_int n)
{
    return (2.0 * (n + 1.0) * n + 2.0 * n) / 1e9;
}

/* \brief floating point counts of GER */
template <typename T, bool CONJ>
constexpr double ger_gflop_count(rocblas_int m, rocblas_int n)
{
    rocblas_int min = (m < n) ? m : n;
    return (2.0 * m * n + min) / 1e9;
}

template <>
constexpr double ger_gflop_count<rocblas_float_complex, false>(rocblas_int m, rocblas_int n)
{
    // conjugate not counted
    return 4.0 * ger_gflop_count<float, false>(m, n);
}

template <>
constexpr double ger_gflop_count<rocblas_float_complex, true>(rocblas_int m, rocblas_int n)
{

    return 4.0 * ger_gflop_count<float, false>(m, n) + n; // conjugate +n
}

template <>
constexpr double ger_gflop_count<rocblas_double_complex, false>(rocblas_int m, rocblas_int n)
{
    return ger_gflop_count<rocblas_float_complex, false>(m, n);
}

template <>
constexpr double ger_gflop_count<rocblas_double_complex, true>(rocblas_int m, rocblas_int n)
{
    return ger_gflop_count<rocblas_float_complex, true>(m, n);
}

/* \brief floating point counts of SYR */
template <typename T>
constexpr double syr_gflop_count(rocblas_int n)
{
    return (n * (n + 1.0) + n) / 1e9;
}

template <>
constexpr double syr_gflop_count<rocblas_float_complex>(rocblas_int n)
{
    return 4.0 * syr_gflop_count<float>(n);
}

template <>
constexpr double syr_gflop_count<rocblas_double_complex>(rocblas_int n)
{
    return syr_gflop_count<rocblas_float_complex>(n);
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
