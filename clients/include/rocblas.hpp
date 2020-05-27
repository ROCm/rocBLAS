/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _ROCBLAS_HPP_
#define _ROCBLAS_HPP_

/* library headers */
#include "../../library/src/include/utility.h"
#include "rocblas.h"
#include "rocblas_fortran.hpp"

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
template <typename T, typename U = T, bool FORTRAN = false>
static rocblas_status (*rocblas_scal)(
    rocblas_handle handle, rocblas_int n, const U* alpha, T* x, rocblas_int incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<float> = rocblas_sscal;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<float, float, true> = rocblas_sscal_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<double> = rocblas_dscal;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<double, double, true> = rocblas_dscal_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<rocblas_float_complex> = rocblas_cscal;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal<rocblas_float_complex, rocblas_float_complex, true> = rocblas_cscal_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<rocblas_double_complex> = rocblas_zscal;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal<rocblas_double_complex, rocblas_double_complex, true> = rocblas_zscal_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<rocblas_float_complex, float> = rocblas_csscal;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal<rocblas_float_complex, float, true> = rocblas_csscal_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal<rocblas_double_complex, double> = rocblas_zdscal;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal<rocblas_double_complex, double, true> = rocblas_zdscal_fortran;

// scal_batched
template <typename T, typename U = T, bool FORTRAN = false>
static rocblas_status (*rocblas_scal_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const U*       alpha,
                                              T* const       x[],
                                              rocblas_int    incx,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal_batched<float> = rocblas_sscal_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<float, float, true> = rocblas_sscal_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal_batched<double> = rocblas_dscal_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<double, double, true> = rocblas_dscal_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<rocblas_float_complex> = rocblas_cscal_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal_batched<rocblas_float_complex,
                                                         rocblas_float_complex,
                                                         true> = rocblas_cscal_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<rocblas_double_complex> = rocblas_zscal_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_scal_batched<rocblas_double_complex,
                                                         rocblas_double_complex,
                                                         true> = rocblas_zscal_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<rocblas_float_complex, float> = rocblas_csscal_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<rocblas_float_complex, float, true> = rocblas_csscal_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<rocblas_double_complex, double> = rocblas_zdscal_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_batched<rocblas_double_complex, double, true> = rocblas_zdscal_batched_fortran;

// scal_strided_batched
template <typename T, typename U = T, bool FORTRAN = false>
static rocblas_status (*rocblas_scal_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const U*       alpha,
                                                      T*             x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<float> = rocblas_sscal_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<float, float, true> = rocblas_sscal_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<double> = rocblas_dscal_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<double, double, true> = rocblas_dscal_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_float_complex> = rocblas_cscal_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_float_complex,
                                 rocblas_float_complex,
                                 true> = rocblas_cscal_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_double_complex> = rocblas_zscal_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_double_complex,
                                 rocblas_double_complex,
                                 true> = rocblas_zscal_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_float_complex, float> = rocblas_csscal_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_float_complex,
                                 float,
                                 true> = rocblas_csscal_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_double_complex, double> = rocblas_zdscal_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_scal_strided_batched<rocblas_double_complex,
                                 double,
                                 true> = rocblas_zdscal_strided_batched_fortran;

// copy
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_copy)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy<float> = rocblas_scopy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy<float, true> = rocblas_scopy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy<double> = rocblas_dcopy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy<double, true> = rocblas_dcopy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy<rocblas_float_complex> = rocblas_ccopy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy<rocblas_float_complex, true> = rocblas_ccopy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy<rocblas_double_complex> = rocblas_zcopy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy<rocblas_double_complex, true> = rocblas_zcopy_fortran;

template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_copy_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy_batched<float> = rocblas_scopy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_batched<float, true> = rocblas_scopy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_copy_batched<double> = rocblas_dcopy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_batched<double, true> = rocblas_dcopy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_batched<rocblas_float_complex> = rocblas_ccopy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_batched<rocblas_float_complex, true> = rocblas_ccopy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_batched<rocblas_double_complex> = rocblas_zcopy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_batched<rocblas_double_complex, true> = rocblas_zcopy_batched_fortran;

template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_copy_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<float> = rocblas_scopy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<float, true> = rocblas_scopy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<double> = rocblas_dcopy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<double, true> = rocblas_dcopy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<rocblas_float_complex> = rocblas_ccopy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ccopy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<rocblas_double_complex> = rocblas_zcopy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_copy_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zcopy_strided_batched_fortran;

// swap
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_swap)(
    rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap<float> = rocblas_sswap;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap<float, true> = rocblas_sswap_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap<double> = rocblas_dswap;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap<double, true> = rocblas_dswap_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap<rocblas_float_complex> = rocblas_cswap;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap<rocblas_float_complex, true> = rocblas_cswap_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap<rocblas_double_complex> = rocblas_zswap;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap<rocblas_double_complex, true> = rocblas_zswap_fortran;

// swap_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_swap_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              T*             x[],
                                              rocblas_int    incx,
                                              T*             y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap_batched<float> = rocblas_sswap_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_batched<float, true> = rocblas_sswap_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_swap_batched<double> = rocblas_dswap_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_batched<double, true> = rocblas_dswap_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_batched<rocblas_float_complex> = rocblas_cswap_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_batched<rocblas_float_complex, true> = rocblas_cswap_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_batched<rocblas_double_complex> = rocblas_zswap_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_batched<rocblas_double_complex, true> = rocblas_zswap_batched_fortran;

// swap_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_swap_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      T*             x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<float> = rocblas_sswap_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<float, true> = rocblas_sswap_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<double> = rocblas_dswap_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<double, true> = rocblas_dswap_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<rocblas_float_complex> = rocblas_cswap_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cswap_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<rocblas_double_complex> = rocblas_zswap_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_swap_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zswap_strided_batched_fortran;

// dot
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dot)(rocblas_handle handle,
                                     rocblas_int    n,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       y,
                                     rocblas_int    incy,
                                     T*             result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<float> = rocblas_sdot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<float, true> = rocblas_sdot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<double> = rocblas_ddot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<double, true> = rocblas_ddot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<rocblas_half> = rocblas_hdot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<rocblas_half, true> = rocblas_hdot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<rocblas_bfloat16> = rocblas_bfdot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<rocblas_bfloat16, true> = rocblas_bfdot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<rocblas_float_complex> = rocblas_cdotu;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot<rocblas_float_complex, true> = rocblas_cdotu_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot<rocblas_double_complex> = rocblas_zdotu;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot<rocblas_double_complex, true> = rocblas_zdotu_fortran;

// dotc
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dotc)(rocblas_handle handle,
                                      rocblas_int    n,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dotc<rocblas_float_complex> = rocblas_cdotc;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc<rocblas_float_complex, true> = rocblas_cdotc_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dotc<rocblas_double_complex> = rocblas_zdotc;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc<rocblas_double_complex, true> = rocblas_zdotc_fortran;

// dot_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dot_batched)(rocblas_handle handle,
                                             rocblas_int    n,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             const T* const y[],
                                             rocblas_int    incy,
                                             rocblas_int    batch_count,
                                             T*             result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot_batched<float> = rocblas_sdot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot_batched<float, true> = rocblas_sdot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot_batched<double> = rocblas_ddot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<double, true> = rocblas_ddot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot_batched<rocblas_half> = rocblas_hdot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<rocblas_half, true> = rocblas_hdot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dot_batched<rocblas_bfloat16> = rocblas_bfdot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<rocblas_bfloat16, true> = rocblas_bfdot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<rocblas_float_complex> = rocblas_cdotu_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<rocblas_float_complex, true> = rocblas_cdotu_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<rocblas_double_complex> = rocblas_zdotu_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_batched<rocblas_double_complex, true> = rocblas_zdotu_batched_fortran;

// dotc_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dotc_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count,
                                              T*             result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_batched<rocblas_float_complex> = rocblas_cdotc_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_batched<rocblas_float_complex, true> = rocblas_cdotc_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_batched<rocblas_double_complex> = rocblas_zdotc_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_batched<rocblas_double_complex, true> = rocblas_zdotc_batched_fortran;

// dot_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dot_strided_batched)(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     const T*       y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stridey,
                                                     rocblas_int    batch_count,
                                                     T*             result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<float> = rocblas_sdot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<float, true> = rocblas_sdot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<double> = rocblas_ddot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<double, true> = rocblas_ddot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_half> = rocblas_hdot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_half, true> = rocblas_hdot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_bfloat16> = rocblas_bfdot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_bfloat16, true> = rocblas_bfdot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_float_complex> = rocblas_cdotu_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_float_complex,
                                true> = rocblas_cdotu_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_double_complex> = rocblas_zdotu_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dot_strided_batched<rocblas_double_complex,
                                true> = rocblas_zdotu_strided_batched_fortran;

// dotc
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dotc_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count,
                                                      T*             result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_strided_batched<rocblas_float_complex> = rocblas_cdotc_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cdotc_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_strided_batched<rocblas_double_complex> = rocblas_zdotc_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dotc_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zdotc_strided_batched_fortran;

// asum
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_asum)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, real_t<T>* result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum<float> = rocblas_sasum;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum<float, true> = rocblas_sasum_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum<double> = rocblas_dasum;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum<double, true> = rocblas_dasum_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum<rocblas_float_complex> = rocblas_scasum;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum<rocblas_float_complex, true> = rocblas_scasum_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum<rocblas_double_complex> = rocblas_dzasum;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum<rocblas_double_complex, true> = rocblas_dzasum_fortran;

// asum_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_asum_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              rocblas_int    batch_count,
                                              real_t<T>*     result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum_batched<float> = rocblas_sasum_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_batched<float, true> = rocblas_sasum_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_asum_batched<double> = rocblas_dasum_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_batched<double, true> = rocblas_dasum_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_batched<rocblas_float_complex> = rocblas_scasum_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_batched<rocblas_float_complex, true> = rocblas_scasum_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_batched<rocblas_double_complex> = rocblas_dzasum_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_batched<rocblas_double_complex, true> = rocblas_dzasum_batched_fortran;

// asum_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_asum_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      rocblas_int    batch_count,
                                                      real_t<T>*     result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<float> = rocblas_sasum_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<float, true> = rocblas_sasum_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<double> = rocblas_dasum_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<double, true> = rocblas_dasum_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<rocblas_float_complex> = rocblas_scasum_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<rocblas_float_complex,
                                 true> = rocblas_scasum_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<rocblas_double_complex> = rocblas_dzasum_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_asum_strided_batched<rocblas_double_complex,
                                 true> = rocblas_dzasum_strided_batched_fortran;

// nrm2
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_nrm2)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, real_t<T>* result);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2<float> = rocblas_snrm2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2<float, true> = rocblas_snrm2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2<double> = rocblas_dnrm2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2<double, true> = rocblas_dnrm2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2<rocblas_float_complex> = rocblas_scnrm2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2<rocblas_float_complex, true> = rocblas_scnrm2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2<rocblas_double_complex> = rocblas_dznrm2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2<rocblas_double_complex, true> = rocblas_dznrm2_fortran;

// nrm2_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_nrm2_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              rocblas_int    batch_count,
                                              real_t<T>*     results);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2_batched<float> = rocblas_snrm2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_batched<float, true> = rocblas_snrm2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_nrm2_batched<double> = rocblas_dnrm2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_batched<double, true> = rocblas_dnrm2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_batched<rocblas_float_complex> = rocblas_scnrm2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_batched<rocblas_float_complex, true> = rocblas_scnrm2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_batched<rocblas_double_complex> = rocblas_dznrm2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_batched<rocblas_double_complex, true> = rocblas_dznrm2_batched_fortran;

// nrm2_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_nrm2_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      rocblas_int    batch_count,
                                                      real_t<T>*     results);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<float> = rocblas_snrm2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<float, true> = rocblas_snrm2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<double> = rocblas_dnrm2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<double, true> = rocblas_dnrm2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<rocblas_float_complex> = rocblas_scnrm2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<rocblas_float_complex,
                                 true> = rocblas_scnrm2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<rocblas_double_complex> = rocblas_dznrm2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_nrm2_strided_batched<rocblas_double_complex,
                                 true> = rocblas_dznrm2_strided_batched_fortran;

// iamax and iamin need to be full functions rather than references, in order
// to allow them to be passed as template arguments
//
// iamax and iamin.
//

//
// Define the signature type.
//
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_t = rocblas_status (*)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

//
// iamax
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_t<T, FORTRAN> rocblas_iamax;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax<float> = rocblas_isamax;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax<float, true> = rocblas_isamax_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax<double> = rocblas_idamax;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax<double, true> = rocblas_idamax_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax<rocblas_float_complex> = rocblas_icamax;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax<rocblas_float_complex, true> = rocblas_icamax_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax<rocblas_double_complex> = rocblas_izamax;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax<rocblas_double_complex, true> = rocblas_izamax_fortran;

//
// iamin
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_t<T, FORTRAN> rocblas_iamin;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin<float> = rocblas_isamin;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin<float, true> = rocblas_isamin_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin<double> = rocblas_idamin;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin<double, true> = rocblas_idamin_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin<rocblas_float_complex> = rocblas_icamin;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin<rocblas_float_complex, true> = rocblas_icamin_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin<rocblas_double_complex> = rocblas_izamin;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin<rocblas_double_complex, true> = rocblas_izamin_fortran;

//
// Define the signature type for the iamax_iamin batched.
//
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_batched_t = rocblas_status (*)(rocblas_handle  handle,
                                                         rocblas_int     n,
                                                         const T* const* x,
                                                         rocblas_int     incx,
                                                         rocblas_int     batch_count,
                                                         rocblas_int*    result);

//
// iamax
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_batched_t<T, FORTRAN> rocblas_iamax_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax_batched<float> = rocblas_isamax_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_batched<float, true> = rocblas_isamax_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamax_batched<double> = rocblas_idamax_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_batched<double, true> = rocblas_idamax_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_batched<rocblas_float_complex> = rocblas_icamax_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_batched<rocblas_float_complex, true> = rocblas_icamax_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_batched<rocblas_double_complex> = rocblas_izamax_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_batched<rocblas_double_complex, true> = rocblas_izamax_batched_fortran;

//
// iamin
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_batched_t<T, FORTRAN> rocblas_iamin_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin_batched<float> = rocblas_isamin_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_batched<float, true> = rocblas_isamin_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_iamin_batched<double> = rocblas_idamin_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_batched<double, true> = rocblas_idamin_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_batched<rocblas_float_complex> = rocblas_icamin_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_batched<rocblas_float_complex, true> = rocblas_icamin_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_batched<rocblas_double_complex> = rocblas_izamin_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_batched<rocblas_double_complex, true> = rocblas_izamin_batched_fortran;

//
// Define the signature type for the iamax_iamin strided batched.
//
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_strided_batched_t = rocblas_status (*)(rocblas_handle handle,
                                                                 rocblas_int    n,
                                                                 const T*       x,
                                                                 rocblas_int    incx,
                                                                 rocblas_stride stridex,
                                                                 rocblas_int    batch_count,
                                                                 rocblas_int*   result);

//
// iamax
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_strided_batched_t<T, FORTRAN> rocblas_iamax_strided_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<float> = rocblas_isamax_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<float, true> = rocblas_isamax_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<double> = rocblas_idamax_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<double, true> = rocblas_idamax_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<rocblas_float_complex> = rocblas_icamax_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<rocblas_float_complex,
                                  true> = rocblas_icamax_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<rocblas_double_complex> = rocblas_izamax_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamax_strided_batched<rocblas_double_complex,
                                  true> = rocblas_izamax_strided_batched_fortran;

//
// iamin
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_strided_batched_t<T, FORTRAN> rocblas_iamin_strided_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<float> = rocblas_isamin_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<float, true> = rocblas_isamin_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<double> = rocblas_idamin_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<double, true> = rocblas_idamin_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<rocblas_float_complex> = rocblas_icamin_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<rocblas_float_complex,
                                  true> = rocblas_icamin_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<rocblas_double_complex> = rocblas_izamin_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_iamin_strided_batched<rocblas_double_complex,
                                  true> = rocblas_izamin_strided_batched_fortran;

// axpy
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_axpy)(rocblas_handle handle,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<rocblas_half, false> = rocblas_haxpy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<rocblas_half, true> = rocblas_haxpy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<float, false> = rocblas_saxpy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<float, true> = rocblas_saxpy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<double, false> = rocblas_daxpy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<double, true> = rocblas_daxpy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<rocblas_float_complex, false> = rocblas_caxpy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy<rocblas_float_complex, true> = rocblas_caxpy_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy<rocblas_double_complex, false> = rocblas_zaxpy;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy<rocblas_double_complex, true> = rocblas_zaxpy_fortran;

// axpy batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_axpy_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<rocblas_half, false> = rocblas_haxpy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<rocblas_half, true> = rocblas_haxpy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy_batched<float, false> = rocblas_saxpy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<float, true> = rocblas_saxpy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_axpy_batched<double, false> = rocblas_daxpy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<double, true> = rocblas_daxpy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<rocblas_float_complex, false> = rocblas_caxpy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<rocblas_float_complex, true> = rocblas_caxpy_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<rocblas_double_complex, false> = rocblas_zaxpy_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_batched<rocblas_double_complex, true> = rocblas_zaxpy_batched_fortran;

// axpy strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_axpy_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<rocblas_half, false> = rocblas_haxpy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<rocblas_half, true> = rocblas_haxpy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<float, false> = rocblas_saxpy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<float, true> = rocblas_saxpy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<double, false> = rocblas_daxpy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<double, true> = rocblas_daxpy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<rocblas_float_complex, false> = rocblas_caxpy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<rocblas_float_complex,
                                 true> = rocblas_caxpy_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<rocblas_double_complex, false> = rocblas_zaxpy_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_axpy_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zaxpy_strided_batched_fortran;

// rot
template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
static rocblas_status (*rocblas_rot)(rocblas_handle handle,
                                     rocblas_int    n,
                                     T*             x,
                                     rocblas_int    incx,
                                     T*             y,
                                     rocblas_int    incy,
                                     const U*       c,
                                     const V*       s);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot<float> = rocblas_srot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot<float, float, float, true> = rocblas_srot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot<double> = rocblas_drot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<double, double, double, true> = rocblas_drot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_float_complex, float, rocblas_float_complex> = rocblas_crot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_float_complex, float, rocblas_float_complex, true> = rocblas_crot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_float_complex, float, float> = rocblas_csrot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_float_complex, float, float, true> = rocblas_csrot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_double_complex, double, rocblas_double_complex> = rocblas_zrot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot<rocblas_double_complex,
                                                double,
                                                rocblas_double_complex,
                                                true> = rocblas_zrot_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_double_complex, double, double> = rocblas_zdrot;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot<rocblas_double_complex, double, double, true> = rocblas_zdrot_fortran;

// rot_batched
template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
static rocblas_status (*rocblas_rot_batched)(rocblas_handle handle,
                                             rocblas_int    n,
                                             T* const       x[],
                                             rocblas_int    incx,
                                             T* const       y[],
                                             rocblas_int    incy,
                                             const U*       c,
                                             const V*       s,
                                             rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot_batched<float> = rocblas_srot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<float, float, float, true> = rocblas_srot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot_batched<double> = rocblas_drot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<double, double, double, true> = rocblas_drot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<rocblas_float_complex, float, rocblas_float_complex> = rocblas_crot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot_batched<rocblas_float_complex,
                                                        float,
                                                        rocblas_float_complex,
                                                        true> = rocblas_crot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<rocblas_float_complex, float, float> = rocblas_csrot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<rocblas_float_complex, float, float, true> = rocblas_csrot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<rocblas_double_complex,
                        double,
                        rocblas_double_complex> = rocblas_zrot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot_batched<rocblas_double_complex,
                                                        double,
                                                        rocblas_double_complex,
                                                        true> = rocblas_zrot_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_batched<rocblas_double_complex, double, double> = rocblas_zdrot_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rot_batched<rocblas_double_complex,
                                                        double,
                                                        double,
                                                        true> = rocblas_zdrot_batched_fortran;

// rot_strided_batched
template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
static rocblas_status (*rocblas_rot_strided_batched)(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     T*             x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stride_y,
                                                     const U*       c,
                                                     const V*       s,
                                                     rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<float> = rocblas_srot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<float, float, float, true> = rocblas_srot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<double> = rocblas_drot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<double,
                                double,
                                double,
                                true> = rocblas_drot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_float_complex,
                                float,
                                rocblas_float_complex> = rocblas_crot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_float_complex,
                                float,
                                rocblas_float_complex,
                                true> = rocblas_crot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_float_complex,
                                float,
                                float> = rocblas_csrot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_float_complex,
                                float,
                                float,
                                true> = rocblas_csrot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_double_complex,
                                double,
                                rocblas_double_complex> = rocblas_zrot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_double_complex,
                                double,
                                rocblas_double_complex,
                                true> = rocblas_zrot_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_double_complex,
                                double,
                                double> = rocblas_zdrot_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rot_strided_batched<rocblas_double_complex,
                                double,
                                double,
                                true> = rocblas_zdrot_strided_batched_fortran;

// rotg
template <typename T, typename U = T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotg)(rocblas_handle handle, T* a, T* b, U* c, T* s);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg<float> = rocblas_srotg;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg<float, float, true> = rocblas_srotg_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg<double> = rocblas_drotg;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg<double, double, true> = rocblas_drotg_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg<rocblas_float_complex, float> = rocblas_crotg;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg<rocblas_float_complex, float, true> = rocblas_crotg_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg<rocblas_double_complex, double> = rocblas_zrotg;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg<rocblas_double_complex, double, true> = rocblas_zrotg_fortran;

// rotg_batched
template <typename T, typename U = T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotg_batched)(rocblas_handle handle,
                                              T* const       a[],
                                              T* const       b[],
                                              U* const       c[],
                                              T* const       s[],
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg_batched<float> = rocblas_srotg_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_batched<float, float, true> = rocblas_srotg_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotg_batched<double> = rocblas_drotg_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_batched<double, double, true> = rocblas_drotg_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_batched<rocblas_float_complex, float> = rocblas_crotg_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_batched<rocblas_float_complex, float, true> = rocblas_crotg_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_batched<rocblas_double_complex, double> = rocblas_zrotg_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_batched<rocblas_double_complex, double, true> = rocblas_zrotg_batched_fortran;

//rotg_strided_batched
template <typename T, typename U = T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotg_strided_batched)(rocblas_handle handle,
                                                      T*             a,
                                                      rocblas_stride stride_a,
                                                      T*             b,
                                                      rocblas_stride stride_b,
                                                      U*             c,
                                                      rocblas_stride stride_c,
                                                      T*             s,
                                                      rocblas_stride stride_s,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<float> = rocblas_srotg_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<float, float, true> = rocblas_srotg_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<double> = rocblas_drotg_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<double, double, true> = rocblas_drotg_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<rocblas_float_complex, float> = rocblas_crotg_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<rocblas_float_complex,
                                 float,
                                 true> = rocblas_crotg_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<rocblas_double_complex, double> = rocblas_zrotg_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotg_strided_batched<rocblas_double_complex,
                                 double,
                                 true> = rocblas_zrotg_strided_batched_fortran;

//rotm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotm)(rocblas_handle handle,
                                      rocblas_int    n,
                                      T*             x,
                                      rocblas_int    incx,
                                      T*             y,
                                      rocblas_int    incy,
                                      const T*       param);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotm<float> = rocblas_srotm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotm<float, true> = rocblas_srotm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotm<double> = rocblas_drotm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotm<double, true> = rocblas_drotm_fortran;

// rotm_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotm_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              T* const       x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              const T* const param[],
                                              rocblas_int    batch_count);
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotm_batched<float> = rocblas_srotm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotm_batched<float, true> = rocblas_srotm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotm_batched<double> = rocblas_drotm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotm_batched<double, true> = rocblas_drotm_batched_fortran;

// rotm_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotm_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      T*             x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      const T*       param,
                                                      rocblas_stride stride_param,
                                                      rocblas_int    batch_count);
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotm_strided_batched<float> = rocblas_srotm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotm_strided_batched<float, true> = rocblas_srotm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotm_strided_batched<double> = rocblas_drotm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotm_strided_batched<double, true> = rocblas_drotm_strided_batched_fortran;

//rotmg
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotmg)(
    rocblas_handle handle, T* d1, T* d2, T* x1, const T* y1, T* param);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotmg<float> = rocblas_srotmg;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotmg<float, true> = rocblas_srotmg_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotmg<double> = rocblas_drotmg;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotmg<double, true> = rocblas_drotmg_fortran;

//rotmg_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotmg_batched)(rocblas_handle handle,
                                               T* const       d1[],
                                               T* const       d2[],
                                               T* const       x1[],
                                               const T* const y1[],
                                               T* const       param[],
                                               rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotmg_batched<float> = rocblas_srotmg_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotmg_batched<float, true> = rocblas_srotmg_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_rotmg_batched<double> = rocblas_drotmg_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotmg_batched<double, true> = rocblas_drotmg_batched_fortran;

//rotmg_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_rotmg_strided_batched)(rocblas_handle handle,
                                                       T*             d1,
                                                       rocblas_stride stride_d1,
                                                       T*             d2,
                                                       rocblas_stride stride_d2,
                                                       T*             x1,
                                                       rocblas_stride stride_x1,
                                                       const T*       y1,
                                                       rocblas_stride stride_y1,
                                                       T*             param,
                                                       rocblas_stride stride_param,
                                                       rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotmg_strided_batched<float> = rocblas_srotmg_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotmg_strided_batched<float, true> = rocblas_srotmg_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotmg_strided_batched<double> = rocblas_drotmg_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_rotmg_strided_batched<double, true> = rocblas_drotmg_strided_batched_fortran;

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// ger
template <typename T, bool CONJ, bool FORTRAN = false>
static rocblas_status (*rocblas_ger)(rocblas_handle handle,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       y,
                                     rocblas_int    incy,
                                     T*             A,
                                     rocblas_int    lda);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<float, false> = rocblas_sger;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<float, false, true> = rocblas_sger_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<double, false> = rocblas_dger;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<double, false, true> = rocblas_dger_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<rocblas_float_complex, false> = rocblas_cgeru;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger<rocblas_float_complex, false, true> = rocblas_cgeru_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<rocblas_double_complex, false> = rocblas_zgeru;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger<rocblas_double_complex, false, true> = rocblas_zgeru_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<rocblas_float_complex, true> = rocblas_cgerc;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger<rocblas_float_complex, true, true> = rocblas_cgerc_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger<rocblas_double_complex, true> = rocblas_zgerc;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger<rocblas_double_complex, true, true> = rocblas_zgerc_fortran;

template <typename T, bool CONJ, bool FORTRAN = false>
static rocblas_status (*rocblas_ger_batched)(rocblas_handle handle,
                                             rocblas_int    m,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             const T* const y[],
                                             rocblas_int    incy,
                                             T* const       A[],
                                             rocblas_int    lda,
                                             rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger_batched<float, false> = rocblas_sger_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<float, false, true> = rocblas_sger_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_ger_batched<double, false> = rocblas_dger_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<double, false, true> = rocblas_dger_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_float_complex, false> = rocblas_cgeru_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_float_complex, false, true> = rocblas_cgeru_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_double_complex, false> = rocblas_zgeru_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_double_complex, false, true> = rocblas_zgeru_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_float_complex, true> = rocblas_cgerc_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_float_complex, true, true> = rocblas_cgerc_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_double_complex, true> = rocblas_zgerc_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_batched<rocblas_double_complex, true, true> = rocblas_zgerc_batched_fortran;

template <typename T, bool CONJ, bool FORTRAN = false>
static rocblas_status (*rocblas_ger_strided_batched)(rocblas_handle handle,
                                                     rocblas_int    m,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     const T*       y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stride_y,
                                                     T*             A,
                                                     rocblas_int    lda,
                                                     rocblas_stride stride_a,
                                                     rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<float, false> = rocblas_sger_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<float, false, true> = rocblas_sger_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<double, false> = rocblas_dger_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<double, false, true> = rocblas_dger_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_float_complex, false> = rocblas_cgeru_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_float_complex,
                                false,
                                true> = rocblas_cgeru_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_double_complex, false> = rocblas_zgeru_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_double_complex,
                                false,
                                true> = rocblas_zgeru_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_float_complex, true> = rocblas_cgerc_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_float_complex,
                                true,
                                true> = rocblas_cgerc_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_double_complex, true> = rocblas_zgerc_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_ger_strided_batched<rocblas_double_complex,
                                true,
                                true> = rocblas_zgerc_strided_batched_fortran;

// spr
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spr)(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     T*             AP);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<float> = rocblas_sspr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<float, true> = rocblas_sspr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<double> = rocblas_dspr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<double, true> = rocblas_dspr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<rocblas_float_complex> = rocblas_cspr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<rocblas_float_complex, true> = rocblas_cspr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr<rocblas_double_complex> = rocblas_zspr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr<rocblas_double_complex, true> = rocblas_zspr_fortran;

// spr_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spr_batched)(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             T* const       AP[],
                                             rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr_batched<float> = rocblas_sspr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr_batched<float, true> = rocblas_sspr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr_batched<double> = rocblas_dspr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_batched<double, true> = rocblas_dspr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_batched<rocblas_float_complex> = rocblas_cspr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_batched<rocblas_float_complex, true> = rocblas_cspr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_batched<rocblas_double_complex> = rocblas_zspr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_batched<rocblas_double_complex, true> = rocblas_zspr_batched_fortran;

// spr_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spr_strided_batched)(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             AP,
                                                     rocblas_stride stride_A,
                                                     rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<float> = rocblas_sspr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<float, true> = rocblas_sspr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<double> = rocblas_dspr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<double, true> = rocblas_dspr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<rocblas_float_complex> = rocblas_cspr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<rocblas_float_complex, true> = rocblas_cspr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<rocblas_double_complex> = rocblas_zspr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr_strided_batched<rocblas_double_complex,
                                true> = rocblas_zspr_strided_batched_fortran;

// spr2
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spr2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             AP);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr2<float> = rocblas_sspr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr2<float, true> = rocblas_sspr2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr2<double> = rocblas_dspr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr2<double, true> = rocblas_dspr2_fortran;

// spr2_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spr2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T* const       AP[],
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr2_batched<float> = rocblas_sspr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr2_batched<float, true> = rocblas_sspr2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spr2_batched<double> = rocblas_dspr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr2_batched<double, true> = rocblas_dspr2_batched_fortran;

// spr2_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spr2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      T*             AP,
                                                      rocblas_stride stride_A,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr2_strided_batched<float> = rocblas_sspr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr2_strided_batched<float, true> = rocblas_sspr2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr2_strided_batched<double> = rocblas_dspr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spr2_strided_batched<double, true> = rocblas_dspr2_strided_batched_fortran;

// syr
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr)(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     T*             A,
                                     rocblas_int    lda);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<float> = rocblas_ssyr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<float, true> = rocblas_ssyr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<double> = rocblas_dsyr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<double, true> = rocblas_dsyr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<rocblas_float_complex> = rocblas_csyr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<rocblas_float_complex, true> = rocblas_csyr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr<rocblas_double_complex> = rocblas_zsyr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr<rocblas_double_complex, true> = rocblas_zsyr_fortran;

// syr strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr_strided_batched)(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     T*             A,
                                                     rocblas_int    lda,
                                                     rocblas_stride strideA,
                                                     rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<float> = rocblas_ssyr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<float, true> = rocblas_ssyr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<double> = rocblas_dsyr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<double, true> = rocblas_dsyr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<rocblas_float_complex> = rocblas_csyr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<rocblas_float_complex, true> = rocblas_csyr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<rocblas_double_complex> = rocblas_zsyr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_strided_batched<rocblas_double_complex,
                                true> = rocblas_zsyr_strided_batched_fortran;

// syr batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr_batched)(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             T*             A[],
                                             rocblas_int    lda,
                                             rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr_batched<float> = rocblas_ssyr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr_batched<float, true> = rocblas_ssyr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr_batched<double> = rocblas_dsyr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_batched<double, true> = rocblas_dsyr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_batched<rocblas_float_complex> = rocblas_csyr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_batched<rocblas_float_complex, true> = rocblas_csyr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_batched<rocblas_double_complex> = rocblas_zsyr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr_batched<rocblas_double_complex, true> = rocblas_zsyr_batched_fortran;

// syr2
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             A,
                                      rocblas_int    lda);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2<float> = rocblas_ssyr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2<float, true> = rocblas_ssyr2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2<double> = rocblas_dsyr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2<double, true> = rocblas_dsyr2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2<rocblas_float_complex> = rocblas_csyr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2<rocblas_float_complex, true> = rocblas_csyr2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2<rocblas_double_complex> = rocblas_zsyr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2<rocblas_double_complex, true> = rocblas_zsyr2_fortran;

// syr2 batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T*             A[],
                                              rocblas_int    lda,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2_batched<float> = rocblas_ssyr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_batched<float, true> = rocblas_ssyr2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2_batched<double> = rocblas_dsyr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_batched<double, true> = rocblas_dsyr2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_batched<rocblas_float_complex> = rocblas_csyr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_batched<rocblas_float_complex, true> = rocblas_csyr2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_batched<rocblas_double_complex> = rocblas_zsyr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_batched<rocblas_double_complex, true> = rocblas_zsyr2_batched_fortran;

// syr2 strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      T*             A,
                                                      rocblas_int    lda,
                                                      rocblas_stride strideA,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<float> = rocblas_ssyr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<float, true> = rocblas_ssyr2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<double> = rocblas_dsyr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<double, true> = rocblas_dsyr2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<rocblas_float_complex> = rocblas_csyr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<rocblas_float_complex,
                                 true> = rocblas_csyr2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<rocblas_double_complex> = rocblas_zsyr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zsyr2_strided_batched_fortran;

// gbmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gbmv)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      rocblas_int       kl,
                                      rocblas_int       ku,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          x,
                                      rocblas_int       incx,
                                      const T*          beta,
                                      T*                y,
                                      rocblas_int       incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv<float> = rocblas_sgbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv<float, true> = rocblas_sgbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv<double> = rocblas_dgbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv<double, true> = rocblas_dgbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv<rocblas_float_complex> = rocblas_cgbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv<rocblas_float_complex, true> = rocblas_cgbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv<rocblas_double_complex> = rocblas_zgbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv<rocblas_double_complex, true> = rocblas_zgbmv_fortran;

// gbmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gbmv_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       kl,
                                              rocblas_int       ku,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T* const    x[],
                                              rocblas_int       incx,
                                              const T*          beta,
                                              T* const          y[],
                                              rocblas_int       incy,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv_batched<float> = rocblas_sgbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_batched<float, true> = rocblas_sgbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gbmv_batched<double> = rocblas_dgbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_batched<double, true> = rocblas_dgbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_batched<rocblas_float_complex> = rocblas_cgbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_batched<rocblas_float_complex, true> = rocblas_cgbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_batched<rocblas_double_complex> = rocblas_zgbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_batched<rocblas_double_complex, true> = rocblas_zgbmv_batched_fortran;

// gbmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gbmv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       kl,
                                                      rocblas_int       ku,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      const T*          x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      const T*          beta,
                                                      T*                y,
                                                      rocblas_int       incy,
                                                      rocblas_stride    stride_y,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<float> = rocblas_sgbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<float, true> = rocblas_sgbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<double> = rocblas_dgbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<double, true> = rocblas_dgbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<rocblas_float_complex> = rocblas_cgbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cgbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<rocblas_double_complex> = rocblas_zgbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gbmv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zgbmv_strided_batched_fortran;

// gemv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gemv)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          x,
                                      rocblas_int       incx,
                                      const T*          beta,
                                      T*                y,
                                      rocblas_int       incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv<float> = rocblas_sgemv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv<float, true> = rocblas_sgemv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv<double> = rocblas_dgemv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv<double, true> = rocblas_dgemv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv<rocblas_float_complex> = rocblas_cgemv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv<rocblas_float_complex, true> = rocblas_cgemv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv<rocblas_double_complex> = rocblas_zgemv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv<rocblas_double_complex, true> = rocblas_zgemv_fortran;

// gemv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gemv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      const T*          beta,
                                                      T*                y,
                                                      rocblas_int       incy,
                                                      rocblas_stride    stride_y,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<float> = rocblas_sgemv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<float, true> = rocblas_sgemv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<double> = rocblas_dgemv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<double, true> = rocblas_dgemv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<rocblas_float_complex> = rocblas_cgemv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cgemv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<rocblas_double_complex> = rocblas_zgemv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zgemv_strided_batched_fortran;

// gemv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gemv_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T* const    x[],
                                              rocblas_int       incx,
                                              const T*          beta,
                                              T* const          y[],
                                              rocblas_int       incy,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv_batched<float> = rocblas_sgemv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_batched<float, true> = rocblas_sgemv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemv_batched<double> = rocblas_dgemv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_batched<double, true> = rocblas_dgemv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_batched<rocblas_float_complex> = rocblas_cgemv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_batched<rocblas_float_complex, true> = rocblas_cgemv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_batched<rocblas_double_complex> = rocblas_zgemv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemv_batched<rocblas_double_complex, true> = rocblas_zgemv_batched_fortran;

// tpmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tpmv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      const T*          A,
                                      T*                x,
                                      rocblas_int       incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv<float> = rocblas_stpmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv<float, true> = rocblas_stpmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv<double> = rocblas_dtpmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv<double, true> = rocblas_dtpmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv<rocblas_float_complex> = rocblas_ctpmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv<rocblas_float_complex, true> = rocblas_ctpmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv<rocblas_double_complex> = rocblas_ztpmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv<rocblas_double_complex, true> = rocblas_ztpmv_fortran;

// tpmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tpmv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      const T*          A,
                                                      rocblas_stride    stridea,
                                                      T*                x,
                                                      rocblas_stride    stridex,
                                                      rocblas_int       incx,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<float> = rocblas_stpmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<float, true> = rocblas_stpmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<double> = rocblas_dtpmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<double, true> = rocblas_dtpmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<rocblas_float_complex> = rocblas_ctpmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctpmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<rocblas_double_complex> = rocblas_ztpmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztpmv_strided_batched_fortran;

// tpmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tpmv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              const T* const*   A,
                                              T* const*         x,
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv_batched<float> = rocblas_stpmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_batched<float, true> = rocblas_stpmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpmv_batched<double> = rocblas_dtpmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_batched<double, true> = rocblas_dtpmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_batched<rocblas_float_complex> = rocblas_ctpmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_batched<rocblas_float_complex, true> = rocblas_ctpmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_batched<rocblas_double_complex> = rocblas_ztpmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpmv_batched<rocblas_double_complex, true> = rocblas_ztpmv_batched_fortran;

// hbmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hbmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hbmv<rocblas_float_complex> = rocblas_chbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv<rocblas_float_complex, true> = rocblas_chbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hbmv<rocblas_double_complex> = rocblas_zhbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv<rocblas_double_complex, true> = rocblas_zhbmv_fortran;

// hbmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hbmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_batched<rocblas_float_complex> = rocblas_chbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_batched<rocblas_float_complex, true> = rocblas_chbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_batched<rocblas_double_complex> = rocblas_zhbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_batched<rocblas_double_complex, true> = rocblas_zhbmv_batched_fortran;

// hbmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hbmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      rocblas_int    k,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_A,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_strided_batched<rocblas_float_complex> = rocblas_chbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_chbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_strided_batched<rocblas_double_complex> = rocblas_zhbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hbmv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zhbmv_strided_batched_fortran;

// hemv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hemv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hemv<rocblas_float_complex> = rocblas_chemv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv<rocblas_float_complex, true> = rocblas_chemv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hemv<rocblas_double_complex> = rocblas_zhemv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv<rocblas_double_complex, true> = rocblas_zhemv_fortran;

// hemv_batched
template <typename T, bool FOTRAN = false>
static rocblas_status (*rocblas_hemv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_batched<rocblas_float_complex> = rocblas_chemv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_batched<rocblas_float_complex, true> = rocblas_chemv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_batched<rocblas_double_complex> = rocblas_zhemv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_batched<rocblas_double_complex, true> = rocblas_zhemv_batched_fortran;

// hemv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hemv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_A,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_strided_batched<rocblas_float_complex> = rocblas_chemv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_chemv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_strided_batched<rocblas_double_complex> = rocblas_zhemv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zhemv_strided_batched_fortran;

// her
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_her)(rocblas_handle   handle,
                                     rocblas_fill     uplo,
                                     rocblas_int      n,
                                     const real_t<T>* alpha,
                                     const T*         x,
                                     rocblas_int      incx,
                                     T*               A,
                                     rocblas_int      lda);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her<rocblas_float_complex> = rocblas_cher;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her<rocblas_float_complex, true> = rocblas_cher_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her<rocblas_double_complex> = rocblas_zher;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her<rocblas_double_complex, true> = rocblas_zher_fortran;

// her_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_her_batched)(rocblas_handle   handle,
                                             rocblas_fill     uplo,
                                             rocblas_int      n,
                                             const real_t<T>* alpha,
                                             const T* const   x[],
                                             rocblas_int      incx,
                                             T* const         A[],
                                             rocblas_int      lda,
                                             rocblas_int      batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_batched<rocblas_float_complex> = rocblas_cher_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_batched<rocblas_float_complex, true> = rocblas_cher_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_batched<rocblas_double_complex> = rocblas_zher_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_batched<rocblas_double_complex, true> = rocblas_zher_batched_fortran;

// her_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_her_strided_batched)(rocblas_handle   handle,
                                                     rocblas_fill     uplo,
                                                     rocblas_int      n,
                                                     const real_t<T>* alpha,
                                                     const T*         x,
                                                     rocblas_int      incx,
                                                     rocblas_stride   stride_x,
                                                     T*               A,
                                                     rocblas_int      lda,
                                                     rocblas_stride   stride_A,
                                                     rocblas_int      batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_strided_batched<rocblas_float_complex> = rocblas_cher_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_strided_batched<rocblas_float_complex, true> = rocblas_cher_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_strided_batched<rocblas_double_complex> = rocblas_zher_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her_strided_batched<rocblas_double_complex,
                                true> = rocblas_zher_strided_batched_fortran;

// her2
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_her2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             A,
                                      rocblas_int    lda);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her2<rocblas_float_complex> = rocblas_cher2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2<rocblas_float_complex, true> = rocblas_cher2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her2<rocblas_double_complex> = rocblas_zher2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2<rocblas_double_complex, true> = rocblas_zher2_fortran;

// her2_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_her2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T* const       A[],
                                              rocblas_int    lda,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_batched<rocblas_float_complex> = rocblas_cher2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_batched<rocblas_float_complex, true> = rocblas_cher2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_batched<rocblas_double_complex> = rocblas_zher2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_batched<rocblas_double_complex, true> = rocblas_zher2_batched_fortran;

// her2_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_her2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      T*             A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_A,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_strided_batched<rocblas_float_complex> = rocblas_cher2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cher2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_strided_batched<rocblas_double_complex> = rocblas_zher2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zher2_strided_batched_fortran;

// hpmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpmv<rocblas_float_complex> = rocblas_chpmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv<rocblas_float_complex, true> = rocblas_chpmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpmv<rocblas_double_complex> = rocblas_zhpmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv<rocblas_double_complex, true> = rocblas_zhpmv_fortran;

// hpmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_batched<rocblas_float_complex> = rocblas_chpmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_batched<rocblas_float_complex, true> = rocblas_chpmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_batched<rocblas_double_complex> = rocblas_zhpmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_batched<rocblas_double_complex, true> = rocblas_zhpmv_batched_fortran;

// hpmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_stride stride_A,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_strided_batched<rocblas_float_complex> = rocblas_chpmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_chpmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_strided_batched<rocblas_double_complex> = rocblas_zhpmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpmv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zhpmv_strided_batched_fortran;

// hpr
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpr)(rocblas_handle   handle,
                                     rocblas_fill     uplo,
                                     rocblas_int      n,
                                     const real_t<T>* alpha,
                                     const T*         x,
                                     rocblas_int      incx,
                                     T*               AP);
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpr<rocblas_float_complex> = rocblas_chpr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpr<rocblas_float_complex, true> = rocblas_chpr_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpr<rocblas_double_complex> = rocblas_zhpr;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr<rocblas_double_complex, true> = rocblas_zhpr_fortran;

// hpr_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpr_batched)(rocblas_handle   handle,
                                             rocblas_fill     uplo,
                                             rocblas_int      n,
                                             const real_t<T>* alpha,
                                             const T* const   x[],
                                             rocblas_int      incx,
                                             T* const         AP[],
                                             rocblas_int      batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_batched<rocblas_float_complex> = rocblas_chpr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_batched<rocblas_float_complex, true> = rocblas_chpr_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_batched<rocblas_double_complex> = rocblas_zhpr_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_batched<rocblas_double_complex, true> = rocblas_zhpr_batched_fortran;

// hpr_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpr_strided_batched)(rocblas_handle   handle,
                                                     rocblas_fill     uplo,
                                                     rocblas_int      n,
                                                     const real_t<T>* alpha,
                                                     const T*         x,
                                                     rocblas_int      incx,
                                                     rocblas_stride   stride_x,
                                                     T*               AP,
                                                     rocblas_stride   stride_A,
                                                     rocblas_int      batch_count);
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_strided_batched<rocblas_float_complex> = rocblas_chpr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_strided_batched<rocblas_float_complex, true> = rocblas_chpr_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_strided_batched<rocblas_double_complex> = rocblas_zhpr_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr_strided_batched<rocblas_double_complex,
                                true> = rocblas_zhpr_strided_batched_fortran;

// hpr2
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpr2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             AP);
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpr2<rocblas_float_complex> = rocblas_chpr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2<rocblas_float_complex, true> = rocblas_chpr2_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hpr2<rocblas_double_complex> = rocblas_zhpr2;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2<rocblas_double_complex, true> = rocblas_zhpr2_fortran;

// hpr2_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpr2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T*             AP,
                                              rocblas_int    batch_count);
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_batched<rocblas_float_complex> = rocblas_chpr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_batched<rocblas_float_complex, true> = rocblas_chpr2_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_batched<rocblas_double_complex> = rocblas_zhpr2_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_batched<rocblas_double_complex, true> = rocblas_zhpr2_batched_fortran;

// hpr2_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hpr2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      T*             AP,
                                                      rocblas_stride stride_A,
                                                      rocblas_int    batch_count);
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_strided_batched<rocblas_float_complex> = rocblas_chpr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_strided_batched<rocblas_float_complex,
                                 true> = rocblas_chpr2_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_strided_batched<rocblas_double_complex> = rocblas_zhpr2_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hpr2_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zhpr2_strided_batched_fortran;

// trmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trmv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv<float> = rocblas_strmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv<float, true> = rocblas_strmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv<double> = rocblas_dtrmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv<double, true> = rocblas_dtrmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv<rocblas_float_complex> = rocblas_ctrmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv<rocblas_float_complex, true> = rocblas_ctrmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv<rocblas_double_complex> = rocblas_ztrmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv<rocblas_double_complex, true> = rocblas_ztrmv_fortran;

// trmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trmv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stridea,
                                                      T*                x,
                                                      rocblas_stride    stridex,
                                                      rocblas_int       incx,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<float> = rocblas_strmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<float, true> = rocblas_strmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<double> = rocblas_dtrmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<double, true> = rocblas_dtrmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<rocblas_float_complex> = rocblas_ctrmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctrmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<rocblas_double_complex> = rocblas_ztrmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztrmv_strided_batched_fortran;

// trmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trmv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              const T* const*   A,
                                              rocblas_int       lda,
                                              T* const*         x,
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv_batched<float> = rocblas_strmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_batched<float, true> = rocblas_strmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmv_batched<double> = rocblas_dtrmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_batched<double, true> = rocblas_dtrmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_batched<rocblas_float_complex> = rocblas_ctrmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_batched<rocblas_float_complex, true> = rocblas_ctrmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_batched<rocblas_double_complex> = rocblas_ztrmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmv_batched<rocblas_double_complex, true> = rocblas_ztrmv_batched_fortran;

// tbmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tbmv)(rocblas_fill      uplo,
                                      rocblas_handle    handle,
                                      rocblas_diagonal  diag,
                                      rocblas_operation transA,
                                      rocblas_int       m,
                                      rocblas_int       k,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          x,
                                      rocblas_int       incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv<float> = rocblas_stbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv<float, true> = rocblas_stbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv<double> = rocblas_dtbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv<double, true> = rocblas_dtbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv<rocblas_float_complex> = rocblas_ctbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv<rocblas_float_complex, true> = rocblas_ctbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv<rocblas_double_complex> = rocblas_ztbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv<rocblas_double_complex, true> = rocblas_ztbmv_fortran;

// tbmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tbmv_batched)(rocblas_fill      uplo,
                                              rocblas_handle    handle,
                                              rocblas_diagonal  diag,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       k,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T* const    x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv_batched<float> = rocblas_stbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_batched<float, true> = rocblas_stbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbmv_batched<double> = rocblas_dtbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_batched<double, true> = rocblas_dtbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_batched<rocblas_float_complex> = rocblas_ctbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_batched<rocblas_float_complex, true> = rocblas_ctbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_batched<rocblas_double_complex> = rocblas_ztbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_batched<rocblas_double_complex, true> = rocblas_ztbmv_batched_fortran;

// tbmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tbmv_strided_batched)(rocblas_fill      uplo,
                                                      rocblas_handle    handle,
                                                      rocblas_diagonal  diag,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       k,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      const T*          x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<float> = rocblas_stbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<float, true> = rocblas_stbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<double> = rocblas_dtbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<double, true> = rocblas_dtbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<rocblas_float_complex> = rocblas_ctbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<rocblas_double_complex> = rocblas_ztbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbmv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztbmv_strided_batched_fortran;

// tbsv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tbsv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv<float> = rocblas_stbsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv<float, true> = rocblas_stbsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv<double> = rocblas_dtbsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv<double, true> = rocblas_dtbsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv<rocblas_float_complex> = rocblas_ctbsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv<rocblas_float_complex, true> = rocblas_ctbsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv<rocblas_double_complex> = rocblas_ztbsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv<rocblas_double_complex, true> = rocblas_ztbsv_fortran;

// tbsv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tbsv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv_batched<float> = rocblas_stbsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_batched<float, true> = rocblas_stbsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tbsv_batched<double> = rocblas_dtbsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_batched<double, true> = rocblas_dtbsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_batched<rocblas_float_complex> = rocblas_ctbsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_batched<rocblas_float_complex, true> = rocblas_ctbsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_batched<rocblas_double_complex> = rocblas_ztbsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_batched<rocblas_double_complex, true> = rocblas_ztbsv_batched_fortran;

// tbsv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tbsv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<float> = rocblas_stbsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<float, true> = rocblas_stbsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<double> = rocblas_dtbsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<double, true> = rocblas_dtbsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<rocblas_float_complex> = rocblas_ctbsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctbsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<rocblas_double_complex> = rocblas_ztbsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tbsv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztbsv_strided_batched_fortran;

// tpsv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tpsv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       n,
                                      const T*          AP,
                                      T*                x,
                                      rocblas_int       incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv<float> = rocblas_stpsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv<float, true> = rocblas_stpsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv<double> = rocblas_dtpsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv<double, true> = rocblas_dtpsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv<rocblas_float_complex> = rocblas_ctpsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv<rocblas_float_complex, true> = rocblas_ctpsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv<rocblas_double_complex> = rocblas_ztpsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv<rocblas_double_complex, true> = rocblas_ztpsv_fortran;

// tpsv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tpsv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              const T* const    AP[],
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv_batched<float> = rocblas_stpsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_batched<float, true> = rocblas_stpsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_tpsv_batched<double> = rocblas_dtpsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_batched<double, true> = rocblas_dtpsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_batched<rocblas_float_complex> = rocblas_ctpsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_batched<rocblas_float_complex, true> = rocblas_ctpsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_batched<rocblas_double_complex> = rocblas_ztpsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_batched<rocblas_double_complex, true> = rocblas_ztpsv_batched_fortran;

// tpsv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_tpsv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       n,
                                                      const T*          AP,
                                                      rocblas_stride    stride_A,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<float> = rocblas_stpsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<float, true> = rocblas_stpsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<double> = rocblas_dtpsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<double, true> = rocblas_dtpsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<rocblas_float_complex> = rocblas_ctpsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctpsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<rocblas_double_complex> = rocblas_ztpsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_tpsv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztpsv_strided_batched_fortran;

// trsv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trsv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv<float> = rocblas_strsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv<float, true> = rocblas_strsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv<double> = rocblas_dtrsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv<double, true> = rocblas_dtrsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv<rocblas_float_complex> = rocblas_ctrsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv<rocblas_float_complex, true> = rocblas_ctrsv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv<rocblas_double_complex> = rocblas_ztrsv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv<rocblas_double_complex, true> = rocblas_ztrsv_fortran;

// trsv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trsv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv_batched<float> = rocblas_strsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_batched<float, true> = rocblas_strsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsv_batched<double> = rocblas_dtrsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_batched<double, true> = rocblas_dtrsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_batched<rocblas_float_complex> = rocblas_ctrsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_batched<rocblas_float_complex, true> = rocblas_ctrsv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_batched<rocblas_double_complex> = rocblas_ztrsv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_batched<rocblas_double_complex, true> = rocblas_ztrsv_batched_fortran;

// trsv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trsv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<float> = rocblas_strsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<float, true> = rocblas_strsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<double> = rocblas_dtrsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<double, true> = rocblas_dtrsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<rocblas_float_complex> = rocblas_ctrsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctrsv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<rocblas_double_complex> = rocblas_ztrsv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztrsv_strided_batched_fortran;

// spmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spmv<float> = rocblas_sspmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spmv<float, true> = rocblas_sspmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spmv<double> = rocblas_dspmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spmv<double, true> = rocblas_dspmv_fortran;

// spmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T*             y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spmv_batched<float> = rocblas_sspmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spmv_batched<float, true> = rocblas_sspmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_spmv_batched<double> = rocblas_dspmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spmv_batched<double, true> = rocblas_dspmv_batched_fortran;

// spmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_spmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spmv_strided_batched<float> = rocblas_sspmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spmv_strided_batched<float, true> = rocblas_sspmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spmv_strided_batched<double> = rocblas_dspmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_spmv_strided_batched<double, true> = rocblas_dspmv_strided_batched_fortran;

// sbmv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_sbmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_sbmv<float> = rocblas_ssbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_sbmv<float, true> = rocblas_ssbmv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_sbmv<double> = rocblas_dsbmv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_sbmv<double, true> = rocblas_dsbmv_fortran;

// sbmv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_sbmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T*             y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_sbmv_batched<float> = rocblas_ssbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_sbmv_batched<float, true> = rocblas_ssbmv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_sbmv_batched<double> = rocblas_dsbmv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_sbmv_batched<double, true> = rocblas_dsbmv_batched_fortran;

// sbmv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_sbmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      rocblas_int    k,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_sbmv_strided_batched<float> = rocblas_ssbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_sbmv_strided_batched<float, true> = rocblas_ssbmv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_sbmv_strided_batched<double> = rocblas_dsbmv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_sbmv_strided_batched<double, true> = rocblas_dsbmv_strided_batched_fortran;

// symv
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_symv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv<float> = rocblas_ssymv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv<float, true> = rocblas_ssymv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv<double> = rocblas_dsymv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv<double, true> = rocblas_dsymv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv<rocblas_float_complex> = rocblas_csymv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv<rocblas_float_complex, true> = rocblas_csymv_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv<rocblas_double_complex> = rocblas_zsymv;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv<rocblas_double_complex, true> = rocblas_zsymv_fortran;

// symv_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_symv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T*             y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv_batched<float> = rocblas_ssymv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_batched<float, true> = rocblas_ssymv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symv_batched<double> = rocblas_dsymv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_batched<double, true> = rocblas_dsymv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_batched<rocblas_float_complex> = rocblas_csymv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_batched<rocblas_float_complex, true> = rocblas_csymv_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_batched<rocblas_double_complex> = rocblas_zsymv_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_batched<rocblas_double_complex, true> = rocblas_zsymv_batched_fortran;

// symv_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_symv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<float> = rocblas_ssymv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<float, true> = rocblas_ssymv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<double> = rocblas_dsymv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<double, true> = rocblas_dsymv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<rocblas_float_complex> = rocblas_csymv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<rocblas_float_complex,
                                 true> = rocblas_csymv_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<rocblas_double_complex> = rocblas_zsymv_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symv_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zsymv_strided_batched_fortran;

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// dgmm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dgmm)(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      T*             C,
                                      rocblas_int    ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm<float> = rocblas_sdgmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm<float, true> = rocblas_sdgmm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm<double> = rocblas_ddgmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm<double, true> = rocblas_ddgmm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm<rocblas_float_complex> = rocblas_cdgmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm<rocblas_float_complex, true> = rocblas_cdgmm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm<rocblas_double_complex> = rocblas_zdgmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm<rocblas_double_complex, true> = rocblas_zdgmm_fortran;

template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dgmm_batched)(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              T* const       C[],
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm_batched<float> = rocblas_sdgmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_batched<float, true> = rocblas_sdgmm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_dgmm_batched<double> = rocblas_ddgmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_batched<double, true> = rocblas_ddgmm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_batched<rocblas_float_complex> = rocblas_cdgmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_batched<rocblas_float_complex, true> = rocblas_cdgmm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_batched<rocblas_double_complex> = rocblas_zdgmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_batched<rocblas_double_complex, true> = rocblas_zdgmm_batched_fortran;

template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_dgmm_strided_batched)(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      T*             C,
                                                      rocblas_int    ldc,
                                                      rocblas_stride stride_c,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<float> = rocblas_sdgmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<float, true> = rocblas_sdgmm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<double> = rocblas_ddgmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<double, true> = rocblas_ddgmm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<rocblas_float_complex> = rocblas_cdgmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cdgmm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<rocblas_double_complex> = rocblas_zdgmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_dgmm_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zdgmm_strided_batched_fortran;

// geam
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_geam)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          beta,
                                      const T*          B,
                                      rocblas_int       ldb,
                                      T*                C,
                                      rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam<float> = rocblas_sgeam;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam<float, true> = rocblas_sgeam_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam<double> = rocblas_dgeam;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam<double, true> = rocblas_dgeam_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam<rocblas_float_complex> = rocblas_cgeam;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam<rocblas_float_complex, true> = rocblas_cgeam_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam<rocblas_double_complex> = rocblas_zgeam;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam<rocblas_double_complex, true> = rocblas_zgeam_fortran;

template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_geam_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T*          beta,
                                              const T* const    B[],
                                              rocblas_int       ldb,
                                              T* const          C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam_batched<float> = rocblas_sgeam_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_batched<float, true> = rocblas_sgeam_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_geam_batched<double> = rocblas_dgeam_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_batched<double, true> = rocblas_dgeam_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_batched<rocblas_float_complex> = rocblas_cgeam_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_batched<rocblas_float_complex, true> = rocblas_cgeam_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_batched<rocblas_double_complex> = rocblas_zgeam_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_batched<rocblas_double_complex, true> = rocblas_zgeam_batched_fortran;

template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_geam_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          beta,
                                                      const T*          B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<float> = rocblas_sgeam_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<float, true> = rocblas_sgeam_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<double> = rocblas_dgeam_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<double, true> = rocblas_dgeam_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<rocblas_float_complex> = rocblas_cgeam_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cgeam_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<rocblas_double_complex> = rocblas_zgeam_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_geam_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zgeam_strided_batched_fortran;

// gemm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gemm)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          B,
                                      rocblas_int       ldb,
                                      const T*          beta,
                                      T*                C,
                                      rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<rocblas_half> = rocblas_hgemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<rocblas_half, true> = rocblas_hgemm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<float> = rocblas_sgemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<float, true> = rocblas_sgemm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<double> = rocblas_dgemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<double, true> = rocblas_dgemm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<rocblas_float_complex> = rocblas_cgemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm<rocblas_float_complex, true> = rocblas_cgemm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<rocblas_double_complex> = rocblas_zgemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm<rocblas_double_complex, true> = rocblas_zgemm_fortran;

// gemm_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gemm_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T* const    B[],
                                              rocblas_int       ldb,
                                              const T*          beta,
                                              T*                C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm_batched<rocblas_half> = rocblas_hgemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<rocblas_half, true> = rocblas_hgemm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm_batched<float> = rocblas_sgemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<float, true> = rocblas_sgemm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm_batched<double> = rocblas_dgemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<double, true> = rocblas_dgemm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<rocblas_float_complex> = rocblas_cgemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<rocblas_float_complex, true> = rocblas_cgemm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<rocblas_double_complex> = rocblas_zgemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_batched<rocblas_double_complex, true> = rocblas_zgemm_batched_fortran;

// gemm_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_gemm_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    bsa,
                                                      const T*          B,
                                                      rocblas_int       ldb,
                                                      rocblas_int       bsb,
                                                      const T*          beta,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    bsc,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<rocblas_half> = rocblas_hgemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<rocblas_half, true> = rocblas_hgemm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<float> = rocblas_sgemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<float, true> = rocblas_sgemm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<double> = rocblas_dgemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<double, true> = rocblas_dgemm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<rocblas_float_complex> = rocblas_cgemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<rocblas_float_complex,
                                 true> = rocblas_cgemm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<rocblas_double_complex> = rocblas_zgemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_gemm_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zgemm_strided_batched_fortran;

// hemm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hemm)(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       B,
                                      rocblas_int    ldb,
                                      const T*       beta,
                                      T*             C,
                                      rocblas_int    ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hemm<rocblas_float_complex> = rocblas_chemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm<rocblas_float_complex, true> = rocblas_chemm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_hemm<rocblas_double_complex> = rocblas_zhemm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm<rocblas_double_complex, true> = rocblas_zhemm_fortran;

// hemm batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hemm_batched)(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_fill   uplo,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T* const alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const B[],
                                              rocblas_int    ldb,
                                              const T* const beta,
                                              T* const       C[],
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_batched<rocblas_float_complex> = rocblas_chemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_batched<rocblas_float_complex, true> = rocblas_chemm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_batched<rocblas_double_complex> = rocblas_zhemm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_batched<rocblas_double_complex, true> = rocblas_zhemm_batched_fortran;

// hemm strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_hemm_strided_batched)(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const T* const alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       B,
                                                      rocblas_int    ldb,
                                                      rocblas_stride stride_b,
                                                      const T* const beta,
                                                      T*             C,
                                                      rocblas_int    ldc,
                                                      rocblas_stride stride_c,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_strided_batched<rocblas_float_complex> = rocblas_chemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_strided_batched<rocblas_float_complex,
                                 true> = rocblas_chemm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_strided_batched<rocblas_double_complex> = rocblas_zhemm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_hemm_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zhemm_strided_batched_fortran;

// herk
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_herk)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const U*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const U*          beta,
                                      T*                C,
                                      rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_herk<rocblas_float_complex, float> = rocblas_cherk;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk<rocblas_float_complex, float, true> = rocblas_cherk_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_herk<rocblas_double_complex, double> = rocblas_zherk;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk<rocblas_double_complex, double, true> = rocblas_zherk_fortran;

// herk batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_herk_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const U* const    alpha,
                                              const T*          A[],
                                              rocblas_int       lda,
                                              const U*          beta,
                                              T*                C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_batched<rocblas_float_complex, float> = rocblas_cherk_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_batched<rocblas_float_complex, float, true> = rocblas_cherk_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_batched<rocblas_double_complex, double> = rocblas_zherk_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_batched<rocblas_double_complex, double, true> = rocblas_zherk_batched_fortran;

// herk strided batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_herk_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const U* const    alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const U*          beta,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_strided_batched<rocblas_float_complex, float> = rocblas_cherk_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_strided_batched<rocblas_float_complex,
                                 float,
                                 true> = rocblas_cherk_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_strided_batched<rocblas_double_complex, double> = rocblas_zherk_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herk_strided_batched<rocblas_double_complex,
                                 double,
                                 true> = rocblas_zherk_strided_batched_fortran;

// her2k
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_her2k)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const U*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her2k<rocblas_float_complex, float> = rocblas_cher2k;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k<rocblas_float_complex, float, true> = rocblas_cher2k_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_her2k<rocblas_double_complex, double> = rocblas_zher2k;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k<rocblas_double_complex, double, true> = rocblas_zher2k_fortran;

// her2k batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_her2k_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T* const    alpha,
                                               const T*          A[],
                                               rocblas_int       lda,
                                               const T*          B[],
                                               rocblas_int       ldb,
                                               const U*          beta,
                                               T*                C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_batched<rocblas_float_complex, float> = rocblas_cher2k_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_batched<rocblas_float_complex, float, true> = rocblas_cher2k_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_batched<rocblas_double_complex, double> = rocblas_zher2k_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_batched<rocblas_double_complex, double, true> = rocblas_zher2k_batched_fortran;

// her2k strided batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_her2k_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const U*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_strided_batched<rocblas_float_complex, float> = rocblas_cher2k_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_strided_batched<rocblas_float_complex,
                                  float,
                                  true> = rocblas_cher2k_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_strided_batched<rocblas_double_complex, double> = rocblas_zher2k_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_her2k_strided_batched<rocblas_double_complex,
                                  double,
                                  true> = rocblas_zher2k_strided_batched_fortran;

// herkx
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_herkx)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const U*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_herkx<rocblas_float_complex, float> = rocblas_cherkx;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx<rocblas_float_complex, float, true> = rocblas_cherkx_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_herkx<rocblas_double_complex, double> = rocblas_zherkx;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx<rocblas_double_complex, double, true> = rocblas_zherkx_fortran;

// herkx batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_herkx_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T* const    alpha,
                                               const T*          A[],
                                               rocblas_int       lda,
                                               const T*          B[],
                                               rocblas_int       ldb,
                                               const U*          beta,
                                               T*                C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_batched<rocblas_float_complex, float> = rocblas_cherkx_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_batched<rocblas_float_complex, float, true> = rocblas_cherkx_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_batched<rocblas_double_complex, double> = rocblas_zherkx_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_batched<rocblas_double_complex, double, true> = rocblas_zherkx_batched_fortran;

// herkx strided batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
static rocblas_status (*rocblas_herkx_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const U*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_strided_batched<rocblas_float_complex, float> = rocblas_cherkx_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_strided_batched<rocblas_float_complex,
                                  float,
                                  true> = rocblas_cherkx_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_strided_batched<rocblas_double_complex, double> = rocblas_zherkx_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_herkx_strided_batched<rocblas_double_complex,
                                  double,
                                  true> = rocblas_zherkx_strided_batched_fortran;

// symm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_symm)(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       B,
                                      rocblas_int    ldb,
                                      const T*       beta,
                                      T*             C,
                                      rocblas_int    ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm<float> = rocblas_ssymm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm<float, true> = rocblas_ssymm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm<double> = rocblas_dsymm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm<double, true> = rocblas_dsymm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm<rocblas_float_complex> = rocblas_csymm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm<rocblas_float_complex, true> = rocblas_csymm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm<rocblas_double_complex> = rocblas_zsymm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm<rocblas_double_complex, true> = rocblas_zsymm_fortran;

// symm batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_symm_batched)(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_fill   uplo,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T* const alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const B[],
                                              rocblas_int    ldb,
                                              const T* const beta,
                                              T* const       C[],
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm_batched<float> = rocblas_ssymm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_batched<float, true> = rocblas_ssymm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_symm_batched<double> = rocblas_dsymm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_batched<double, true> = rocblas_dsymm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_batched<rocblas_float_complex> = rocblas_csymm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_batched<rocblas_float_complex, true> = rocblas_csymm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_batched<rocblas_double_complex> = rocblas_zsymm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_batched<rocblas_double_complex, true> = rocblas_zsymm_batched_fortran;

// symm strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_symm_strided_batched)(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const T* const alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       B,
                                                      rocblas_int    ldb,
                                                      rocblas_stride stride_b,
                                                      const T* const beta,
                                                      T*             C,
                                                      rocblas_int    ldc,
                                                      rocblas_stride stride_c,
                                                      rocblas_int    batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<float> = rocblas_ssymm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<float, true> = rocblas_ssymm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<double> = rocblas_dsymm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<double, true> = rocblas_dsymm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<rocblas_float_complex> = rocblas_csymm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<rocblas_float_complex,
                                 true> = rocblas_csymm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<rocblas_double_complex> = rocblas_zsymm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_symm_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zsymm_strided_batched_fortran;

// syrk
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syrk)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          beta,
                                      T*                C,
                                      rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk<float> = rocblas_ssyrk;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk<float, true> = rocblas_ssyrk_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk<double> = rocblas_dsyrk;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk<double, true> = rocblas_dsyrk_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk<rocblas_float_complex> = rocblas_csyrk;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk<rocblas_float_complex, true> = rocblas_csyrk_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk<rocblas_double_complex> = rocblas_zsyrk;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk<rocblas_double_complex, true> = rocblas_zsyrk_fortran;

// syrk batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syrk_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T* const    alpha,
                                              const T*          A[],
                                              rocblas_int       lda,
                                              const T*          beta,
                                              T*                C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk_batched<float> = rocblas_ssyrk_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_batched<float, true> = rocblas_ssyrk_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrk_batched<double> = rocblas_dsyrk_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_batched<double, true> = rocblas_dsyrk_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_batched<rocblas_float_complex> = rocblas_csyrk_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_batched<rocblas_float_complex, true> = rocblas_csyrk_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_batched<rocblas_double_complex> = rocblas_zsyrk_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_batched<rocblas_double_complex, true> = rocblas_zsyrk_batched_fortran;

// syrk strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syrk_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T* const    alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          beta,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<float> = rocblas_ssyrk_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<float, true> = rocblas_ssyrk_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<double> = rocblas_dsyrk_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<double, true> = rocblas_dsyrk_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<rocblas_float_complex> = rocblas_csyrk_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<rocblas_float_complex,
                                 true> = rocblas_csyrk_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<rocblas_double_complex> = rocblas_zsyrk_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrk_strided_batched<rocblas_double_complex,
                                 true> = rocblas_zsyrk_strided_batched_fortran;

// syr2k
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr2k)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const T*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k<float> = rocblas_ssyr2k;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k<float, true> = rocblas_ssyr2k_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k<double> = rocblas_dsyr2k;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k<double, true> = rocblas_dsyr2k_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k<rocblas_float_complex> = rocblas_csyr2k;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k<rocblas_float_complex, true> = rocblas_csyr2k_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k<rocblas_double_complex> = rocblas_zsyr2k;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k<rocblas_double_complex, true> = rocblas_zsyr2k_fortran;

// syr2k batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr2k_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T* const    alpha,
                                               const T*          A[],
                                               rocblas_int       lda,
                                               const T*          B[],
                                               rocblas_int       ldb,
                                               const T*          beta,
                                               T*                C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k_batched<float> = rocblas_ssyr2k_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_batched<float, true> = rocblas_ssyr2k_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syr2k_batched<double> = rocblas_dsyr2k_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_batched<double, true> = rocblas_dsyr2k_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_batched<rocblas_float_complex> = rocblas_csyr2k_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_batched<rocblas_float_complex, true> = rocblas_csyr2k_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_batched<rocblas_double_complex> = rocblas_zsyr2k_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_batched<rocblas_double_complex, true> = rocblas_zsyr2k_batched_fortran;

// syr2k strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syr2k_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const T*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<float> = rocblas_ssyr2k_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<float, true> = rocblas_ssyr2k_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<double> = rocblas_dsyr2k_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<double, true> = rocblas_dsyr2k_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<rocblas_float_complex> = rocblas_csyr2k_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<rocblas_float_complex,
                                  true> = rocblas_csyr2k_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<rocblas_double_complex> = rocblas_zsyr2k_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syr2k_strided_batched<rocblas_double_complex,
                                  true> = rocblas_zsyr2k_strided_batched_fortran;

// syrkx
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syrkx)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const T*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx<float> = rocblas_ssyrkx;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx<float, true> = rocblas_ssyrkx_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx<double> = rocblas_dsyrkx;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx<double, true> = rocblas_dsyrkx_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx<rocblas_float_complex> = rocblas_csyrkx;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx<rocblas_float_complex, true> = rocblas_csyrkx_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx<rocblas_double_complex> = rocblas_zsyrkx;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx<rocblas_double_complex, true> = rocblas_zsyrkx_fortran;

// syrkx batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syrkx_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T* const    alpha,
                                               const T*          A[],
                                               rocblas_int       lda,
                                               const T*          B[],
                                               rocblas_int       ldb,
                                               const T*          beta,
                                               T*                C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx_batched<float> = rocblas_ssyrkx_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_batched<float, true> = rocblas_ssyrkx_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_syrkx_batched<double> = rocblas_dsyrkx_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_batched<double, true> = rocblas_dsyrkx_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_batched<rocblas_float_complex> = rocblas_csyrkx_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_batched<rocblas_float_complex, true> = rocblas_csyrkx_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_batched<rocblas_double_complex> = rocblas_zsyrkx_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_batched<rocblas_double_complex, true> = rocblas_zsyrkx_batched_fortran;

// syrkx strided batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_syrkx_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const T*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<float> = rocblas_ssyrkx_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<float, true> = rocblas_ssyrkx_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<double> = rocblas_dsyrkx_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<double, true> = rocblas_dsyrkx_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<rocblas_float_complex> = rocblas_csyrkx_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<rocblas_float_complex,
                                  true> = rocblas_csyrkx_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<rocblas_double_complex> = rocblas_zsyrkx_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_syrkx_strided_batched<rocblas_double_complex,
                                  true> = rocblas_zsyrkx_strided_batched_fortran;

// trmm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trmm)(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      T*                A,
                                      rocblas_int       lda,
                                      T*                B,
                                      rocblas_int       ldb);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm<float> = rocblas_strmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm<float, true> = rocblas_strmm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm<double> = rocblas_dtrmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm<double, true> = rocblas_dtrmm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm<rocblas_float_complex> = rocblas_ctrmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm<rocblas_float_complex, true> = rocblas_ctrmm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm<rocblas_double_complex> = rocblas_ztrmm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm<rocblas_double_complex, true> = rocblas_ztrmm_fortran;

// trmm_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trmm_batched)(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transa,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              const T* const    a[],
                                              rocblas_int       lda,
                                              T* const          c[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm_batched<float> = rocblas_strmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_batched<float, true> = rocblas_strmm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trmm_batched<double> = rocblas_dtrmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_batched<double, true> = rocblas_dtrmm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_batched<rocblas_float_complex> = rocblas_ctrmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_batched<rocblas_float_complex, true> = rocblas_ctrmm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_batched<rocblas_double_complex> = rocblas_ztrmm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_batched<rocblas_double_complex, true> = rocblas_ztrmm_batched_fortran;

// trmm_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trmm_strided_batched)(rocblas_handle    handle,
                                                      rocblas_side      side,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transa,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T*          a,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      T*                c,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<float> = rocblas_strmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<float, true> = rocblas_strmm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<double> = rocblas_dtrmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<double, true> = rocblas_dtrmm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<rocblas_float_complex> = rocblas_ctrmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctrmm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<rocblas_double_complex> = rocblas_ztrmm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trmm_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztrmm_strided_batched_fortran;

// trsm
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trsm)(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      T*                A,
                                      rocblas_int       lda,
                                      T*                B,
                                      rocblas_int       ldb);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm<float> = rocblas_strsm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm<float, true> = rocblas_strsm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm<double> = rocblas_dtrsm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm<double, true> = rocblas_dtrsm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm<rocblas_float_complex> = rocblas_ctrsm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm<rocblas_float_complex, true> = rocblas_ctrsm_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm<rocblas_double_complex> = rocblas_ztrsm;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm<rocblas_double_complex, true> = rocblas_ztrsm_fortran;

// trsm_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trsm_batched)(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T* const    alpha,
                                              T*                A[],
                                              rocblas_int       lda,
                                              T*                B[],
                                              rocblas_int       ldb,
                                              rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm_batched<float> = rocblas_strsm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_batched<float, true> = rocblas_strsm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trsm_batched<double> = rocblas_dtrsm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_batched<double, true> = rocblas_dtrsm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_batched<rocblas_float_complex> = rocblas_ctrsm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_batched<rocblas_float_complex, true> = rocblas_ctrsm_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_batched<rocblas_double_complex> = rocblas_ztrsm_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_batched<rocblas_double_complex, true> = rocblas_ztrsm_batched_fortran;

// trsm_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trsm_strided_batched)(rocblas_handle    handle,
                                                      rocblas_side      side,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      T*                A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      T*                B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<float> = rocblas_strsm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<float, true> = rocblas_strsm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<double> = rocblas_dtrsm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<double, true> = rocblas_dtrsm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<rocblas_float_complex> = rocblas_ctrsm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<rocblas_float_complex,
                                 true> = rocblas_ctrsm_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<rocblas_double_complex> = rocblas_ztrsm_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trsm_strided_batched<rocblas_double_complex,
                                 true> = rocblas_ztrsm_strided_batched_fortran;

// trtri
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trtri)(rocblas_handle   handle,
                                       rocblas_fill     uplo,
                                       rocblas_diagonal diag,
                                       rocblas_int      n,
                                       T*               A,
                                       rocblas_int      lda,
                                       T*               invA,
                                       rocblas_int      ldinvA);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri<float> = rocblas_strtri;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri<float, true> = rocblas_strtri_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri<double> = rocblas_dtrtri;
template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri<double, true> = rocblas_dtrtri_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri<rocblas_float_complex> = rocblas_ctrtri;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri<rocblas_float_complex, true> = rocblas_ctrtri_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri<rocblas_double_complex> = rocblas_ztrtri;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri<rocblas_double_complex, true> = rocblas_ztrtri_fortran;

// trtri_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trtri_batched)(rocblas_handle   handle,
                                               rocblas_fill     uplo,
                                               rocblas_diagonal diag,
                                               rocblas_int      n,
                                               T*               A[],
                                               rocblas_int      lda,
                                               T*               invA[],
                                               rocblas_int      ldinvA,
                                               rocblas_int      batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri_batched<float> = rocblas_strtri_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_batched<float, true> = rocblas_strtri_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_trtri_batched<double> = rocblas_dtrtri_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_batched<double, true> = rocblas_dtrtri_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_batched<rocblas_float_complex> = rocblas_ctrtri_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_batched<rocblas_float_complex, true> = rocblas_ctrtri_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_batched<rocblas_double_complex> = rocblas_ztrtri_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_batched<rocblas_double_complex, true> = rocblas_ztrtri_batched_fortran;

// trtri_strided_batched
template <typename T, bool FORTRAN = false>
static rocblas_status (*rocblas_trtri_strided_batched)(rocblas_handle   handle,
                                                       rocblas_fill     uplo,
                                                       rocblas_diagonal diag,
                                                       rocblas_int      n,
                                                       T*               A,
                                                       rocblas_int      lda,
                                                       rocblas_stride   bsa,
                                                       T*               invA,
                                                       rocblas_int      ldinvA,
                                                       rocblas_stride   bsinvA,
                                                       rocblas_int      batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<float> = rocblas_strtri_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<float, true> = rocblas_strtri_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<double> = rocblas_dtrtri_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<double, true> = rocblas_dtrtri_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<rocblas_float_complex> = rocblas_ctrtri_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<rocblas_float_complex,
                                  true> = rocblas_ctrtri_strided_batched_fortran;

template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<rocblas_double_complex> = rocblas_ztrtri_strided_batched;
template <>
ROCBLAS_CLANG_STATIC constexpr auto
    rocblas_trtri_strided_batched<rocblas_double_complex,
                                  true> = rocblas_ztrtri_strided_batched_fortran;

#endif // _ROCBLAS_HPP_
