/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCBLAS_HPP_
#define _ROCBLAS_HPP_

/* library headers */
#include "rocblas.h"

/*!\file
 *  This file exposes C++ templated BLAS interface with only the precision templated.
*/

/*
 * ===========================================================================
 *   README: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// scal
template <typename T>
rocblas_status (*rocblas_scal)(
    rocblas_handle handle, rocblas_int n, const T* alpha, T* x, rocblas_int incx);

template <>
static constexpr auto rocblas_scal<float> = rocblas_sscal;

template <>
static constexpr auto rocblas_scal<double> = rocblas_dscal;

/* not implemented
template <>
static constexpr auto rocblas_scal<rocblas_float_complex> = rocblas_cscal;

template <>
static constexpr auto rocblas_scal<rocblas_double_complex> = rocblas_zscal;
*/

// copy
template <typename T>
rocblas_status (*rocblas_copy)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
static constexpr auto rocblas_copy<float> = rocblas_scopy;

template <>
static constexpr auto rocblas_copy<double> = rocblas_dcopy;

/* not implemented
template <>
static constexpr auto rocblas_copy<rocblas_float_complex> = rocblas_ccopy;

template <>
static constexpr auto rocblas_copy<rocblas_double_complex> = rocblas_zcopy;
}
*/

// swap
template <typename T>
rocblas_status (*rocblas_swap)(
    rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
static constexpr auto rocblas_swap<float> = rocblas_sswap;

template <>
static constexpr auto rocblas_swap<double> = rocblas_dswap;

/* not implemented

template <>
static constexpr auto rocblas_swap<rocblas_float_complex> = rocblas_cswap;

template <>
static constexpr auto rocblas_swap<rocblas_double_complex> = rocblas_zswap;

*/

// dot
template <typename T>
rocblas_status (*rocblas_dot)(rocblas_handle handle,
                              rocblas_int n,
                              const T* x,
                              rocblas_int incx,
                              const T* y,
                              rocblas_int incy,
                              T* result);

template <>
static constexpr auto rocblas_dot<float> = rocblas_sdot;

template <>
static constexpr auto rocblas_dot<double> = rocblas_ddot;

/* not implemented
template <>
static constexpr auto rocblas_dot<rocblas_float_complex> = rocblas_cdotu;

template <>
static constexpr auto rocblas_dot<rocblas_double_complex> = rocblas_zdotu;
*/

// asum
template <typename T1, typename T2>
rocblas_status (*rocblas_asum)(
    rocblas_handle handle, rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <>
static constexpr auto rocblas_asum<float, float> = rocblas_sasum;

template <>
static constexpr auto rocblas_asum<double, double> = rocblas_dasum;

/* not implemented
template<>
static constexpr auto rocblas_asum<rocblas_float_complex, float> = rocblas_scasum;

template<>
static constexpr auto rocblas_asum<rocblas_double_complex, double> = rocblas_dzasum;
*/

// nrm2
template <typename T1, typename T2>
rocblas_status (*rocblas_nrm2)(
    rocblas_handle handle, rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <>
static constexpr auto rocblas_nrm2<float, float> = rocblas_snrm2;

template <>
static constexpr auto rocblas_nrm2<double, double> = rocblas_dnrm2;

/* not implemented
template <>
static constexpr auto rocblas_nrm2<rocblas_float_complex, float> = rocblas_scnrm2;

template <>
static constexpr auto rocblas_nrm2<rocblas_double_complex, double> = rocblas_dznrm2;
*/

// iamax and iamin need to be full functions rather than references, in order
// to allow them to be passed as template arguments
//
// iamax
template <typename T>
rocblas_status rocblas_iamax(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

template <>
inline rocblas_status rocblas_iamax(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result)
{
    return rocblas_isamax(handle, n, x, incx, result);
}

template <>
inline rocblas_status rocblas_iamax(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result)
{
    return rocblas_idamax(handle, n, x, incx, result);
}

// iamin
template <typename T>
rocblas_status rocblas_iamin(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

template <>
inline rocblas_status rocblas_iamin(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result)
{
    return rocblas_isamin(handle, n, x, incx, result);
}

template <>
inline rocblas_status rocblas_iamin(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result)
{
    return rocblas_idamin(handle, n, x, incx, result);
}

// axpy
template <typename T>
rocblas_status (*rocblas_axpy)(rocblas_handle handle,
                               rocblas_int n,
                               const T* alpha,
                               const T* x,
                               rocblas_int incx,
                               T* y,
                               rocblas_int incy);

template <>
static constexpr auto rocblas_axpy<rocblas_half> = rocblas_haxpy;

template <>
static constexpr auto rocblas_axpy<float> = rocblas_saxpy;

template <>
static constexpr auto rocblas_axpy<double> = rocblas_daxpy;

/* not implemented
template <>
static constexpr auto rocblas_axpy<rocblas_float_complex> = rocblas_caxpy;

template <>
static constexpr auto rocblas_axpy<rocblas_double_complex> = rocblas_zaxpy;
*/

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// ger
template <typename T>
rocblas_status (*rocblas_ger)(rocblas_handle handle,
                              rocblas_int m,
                              rocblas_int n,
                              const T* alpha,
                              const T* x,
                              rocblas_int incx,
                              const T* y,
                              rocblas_int incy,
                              T* A,
                              rocblas_int lda);

template <>
static constexpr auto rocblas_ger<float> = rocblas_sger;

template <>
static constexpr auto rocblas_ger<double> = rocblas_dger;

// syr
template <typename T>
rocblas_status (*rocblas_syr)(rocblas_handle handle,
                              rocblas_fill uplo,
                              rocblas_int n,
                              const T* alpha,
                              const T* x,
                              rocblas_int incx,
                              T* A,
                              rocblas_int lda);

template <>
static constexpr auto rocblas_syr<float> = rocblas_ssyr;

template <>
static constexpr auto rocblas_syr<double> = rocblas_dsyr;

// gemv
template <typename T>
rocblas_status (*rocblas_gemv)(rocblas_handle handle,
                               rocblas_operation transA,
                               rocblas_int m,
                               rocblas_int n,
                               const T* alpha,
                               const T* A,
                               rocblas_int lda,
                               const T* x,
                               rocblas_int incx,
                               const T* beta,
                               T* y,
                               rocblas_int incy);

template <>
static constexpr auto rocblas_gemv<float> = rocblas_sgemv;

template <>
static constexpr auto rocblas_gemv<double> = rocblas_dgemv;

// trsv
template <typename T>
rocblas_status (*rocblas_trsv)(rocblas_handle handle,
                               rocblas_fill uplo,
                               rocblas_operation transA,
                               rocblas_diagonal diag,
                               rocblas_int m,
                               const T* A,
                               rocblas_int lda,
                               T* x,
                               rocblas_int incx);

template <>
static constexpr auto rocblas_trsv<float> = rocblas_strsv;

template <>
static constexpr auto rocblas_trsv<double> = rocblas_dtrsv;

// symv
template <typename T>
rocblas_status (*rocblas_symv)(rocblas_handle handle,
                               rocblas_fill uplo,
                               rocblas_int n,
                               const T* alpha,
                               const T* A,
                               rocblas_int lda,
                               const T* x,
                               rocblas_int incx,
                               const T* beta,
                               T* y,
                               rocblas_int incy);

/* not implemented
template <>
static constexpr auto rocblas_symv<float> = rocblas_ssymv;

template <>
static constexpr auto rocblas_symv<double> = rocblas_dsymv;
*/

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// geam
template <typename T>
rocblas_status (*rocblas_geam)(rocblas_handle handle,
                               rocblas_operation transA,
                               rocblas_operation transB,
                               rocblas_int m,
                               rocblas_int n,
                               const T* alpha,
                               const T* A,
                               rocblas_int lda,
                               const T* beta,
                               const T* B,
                               rocblas_int ldb,
                               T* C,
                               rocblas_int ldc);

template <>
static constexpr auto rocblas_geam<float> = rocblas_sgeam;

template <>
static constexpr auto rocblas_geam<double> = rocblas_dgeam;

// gemm
template <typename T>
rocblas_status (*rocblas_gemm)(rocblas_handle handle,
                               rocblas_operation transA,
                               rocblas_operation transB,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               const T* alpha,
                               const T* A,
                               rocblas_int lda,
                               const T* B,
                               rocblas_int ldb,
                               const T* beta,
                               T* C,
                               rocblas_int ldc);

template <>
static constexpr auto rocblas_gemm<rocblas_half> = rocblas_hgemm;

template <>
static constexpr auto rocblas_gemm<float> = rocblas_sgemm;

template <>
static constexpr auto rocblas_gemm<double> = rocblas_dgemm;

// gemm_strided_batched
template <typename T>
rocblas_status (*rocblas_gemm_strided_batched)(rocblas_handle handle,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               const T* alpha,
                                               const T* A,
                                               rocblas_int lda,
                                               rocblas_int bsa,
                                               const T* B,
                                               rocblas_int ldb,
                                               rocblas_int bsb,
                                               const T* beta,
                                               T* C,
                                               rocblas_int ldc,
                                               rocblas_int bsc,
                                               rocblas_int batch_count);

template <>
static constexpr auto rocblas_gemm_strided_batched<rocblas_half> = rocblas_hgemm_strided_batched;

template <>
static constexpr auto rocblas_gemm_strided_batched<float> = rocblas_sgemm_strided_batched;

template <>
static constexpr auto rocblas_gemm_strided_batched<double> = rocblas_dgemm_strided_batched;

// trsm
template <typename T>
rocblas_status (*rocblas_trsm)(rocblas_handle handle,
                               rocblas_side side,
                               rocblas_fill uplo,
                               rocblas_operation transA,
                               rocblas_diagonal diag,
                               rocblas_int m,
                               rocblas_int n,
                               const T* alpha,
                               T* A,
                               rocblas_int lda,
                               T* B,
                               rocblas_int ldb);

template <>
static constexpr auto rocblas_trsm<float> = rocblas_strsm;

template <>
static constexpr auto rocblas_trsm<double> = rocblas_dtrsm;

// trtri
template <typename T>
rocblas_status (*rocblas_trtri)(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_diagonal diag,
                                rocblas_int n,
                                T* A,
                                rocblas_int lda,
                                T* invA,
                                rocblas_int ldinvA);

template <>
static constexpr auto rocblas_trtri<float> = rocblas_strtri;

template <>
static constexpr auto rocblas_trtri<double> = rocblas_dtrtri;

// trtri_batched
template <typename T>
rocblas_status (*rocblas_trtri_batched)(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_diagonal diag,
                                        rocblas_int n,
                                        T* A,
                                        rocblas_int lda,
                                        rocblas_int bsa,
                                        T* invA,
                                        rocblas_int ldinvA,
                                        rocblas_int bsinvA,
                                        rocblas_int batch_count);

template <>
static constexpr auto rocblas_trtri_batched<float> = rocblas_strtri_batched;

template <>
static constexpr auto rocblas_trtri_batched<double> = rocblas_dtrtri_batched;

#endif // _ROCBLAS_HPP_
