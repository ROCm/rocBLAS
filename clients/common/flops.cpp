/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#include "rocblas.h"
#include "flops.h"

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 2 and
 * 3. No flops counts for Level 1 BLAS
*/

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMV */
template <>
double gemv_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n)
{
    return (double)(8.0 * m * n) / 1e9;
}

template <>
double gemv_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n)
{
    return (double)(8.0 * m * n) / 1e9;
}

/* \brief floating point counts of SY(HE)MV */

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMM */
template <>
double gemm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (double)(8.0 * m * n * k) / 1e9;
}

template <>
double gemm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (double)(8.0 * m * n * k) / 1e9;
}

/* \brief floating point counts of TRMM */
template <>
double trsm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (double)(4.0 * m * n * (k + 1)) / 1e9;
}

template <>
double trsm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (double)(4.0 * m * n * (k + 1)) / 1e9;
}

/* \brief floating point counts of TRTRI */
template <>
double trtri_gflop_count<rocblas_float_complex>(rocblas_int n)
{
    return (double)(8.0 * n * n * n) / 3.0 / 1e9;
}

template <>
double trtri_gflop_count<rocblas_double_complex>(rocblas_int n)
{
    return (double)(8.0 * n * n * n) / 3.0 / 1e9;
}
