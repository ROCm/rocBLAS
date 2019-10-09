/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
template <typename T, typename U = T>
rocblas_status (*rocblas_scal)(
    rocblas_handle handle, rocblas_int n, const U* alpha, T* x, rocblas_int incx);

template <>
static constexpr auto rocblas_scal<float> = rocblas_sscal;

template <>
static constexpr auto rocblas_scal<double> = rocblas_dscal;

template <>
static constexpr auto rocblas_scal<rocblas_float_complex> = rocblas_cscal;

template <>
static constexpr auto rocblas_scal<rocblas_double_complex> = rocblas_zscal;

template <>
static constexpr auto rocblas_scal<rocblas_float_complex, float> = rocblas_csscal;

template <>
static constexpr auto rocblas_scal<rocblas_double_complex, double> = rocblas_zdscal;

// scal_batched
template <typename T, typename U = T>
rocblas_status (*rocblas_scal_batched)(rocblas_handle handle,
                                       rocblas_int    n,
                                       const U*       alpha,
                                       T* const       x[],
                                       rocblas_int    incx,
                                       rocblas_int    batch_count);

template <>
static constexpr auto rocblas_scal_batched<float> = rocblas_sscal_batched;

template <>
static constexpr auto rocblas_scal_batched<double> = rocblas_dscal_batched;

template <>
static constexpr auto rocblas_scal_batched<rocblas_float_complex> = rocblas_cscal_batched;

template <>
static constexpr auto rocblas_scal_batched<rocblas_double_complex> = rocblas_zscal_batched;

template <>
static constexpr auto rocblas_scal_batched<rocblas_float_complex, float> = rocblas_csscal_batched;

template <>
static constexpr auto rocblas_scal_batched<rocblas_double_complex, double> = rocblas_zdscal_batched;

// scal_strided_batched
template <typename T, typename U = T>
rocblas_status (*rocblas_scal_strided_batched)(rocblas_handle handle,
                                               rocblas_int    n,
                                               const U*       alpha,
                                               T*             x,
                                               rocblas_int    incx,
                                               rocblas_stride stride_x,
                                               rocblas_int    batch_count);

template <>
static constexpr auto rocblas_scal_strided_batched<float> = rocblas_sscal_strided_batched;

template <>
static constexpr auto rocblas_scal_strided_batched<double> = rocblas_dscal_strided_batched;

template <>
static constexpr auto
    rocblas_scal_strided_batched<rocblas_float_complex> = rocblas_cscal_strided_batched;

template <>
static constexpr auto
    rocblas_scal_strided_batched<rocblas_double_complex> = rocblas_zscal_strided_batched;

template <>
static constexpr auto
    rocblas_scal_strided_batched<rocblas_float_complex, float> = rocblas_csscal_strided_batched;

template <>
static constexpr auto
    rocblas_scal_strided_batched<rocblas_double_complex, double> = rocblas_zdscal_strided_batched;

// copy
template <typename T>
rocblas_status (*rocblas_copy)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
static constexpr auto rocblas_copy<float> = rocblas_scopy;

template <>
static constexpr auto rocblas_copy<double> = rocblas_dcopy;

template <>
static constexpr auto rocblas_copy<rocblas_float_complex> = rocblas_ccopy;

template <>
static constexpr auto rocblas_copy<rocblas_double_complex> = rocblas_zcopy;

template <typename T>
rocblas_status (*rocblas_copy_batched)(rocblas_handle handle,
                                       rocblas_int    n,
                                       const T* const x[],
                                       rocblas_int    incx,
                                       T* const       y[],
                                       rocblas_int    incy,
                                       rocblas_int    batch_count);

template <>
static constexpr auto rocblas_copy_batched<float> = rocblas_scopy_batched;

template <>
static constexpr auto rocblas_copy_batched<double> = rocblas_dcopy_batched;

template <>
static constexpr auto rocblas_copy_batched<rocblas_float_complex> = rocblas_ccopy_batched;

template <>
static constexpr auto rocblas_copy_batched<rocblas_double_complex> = rocblas_zcopy_batched;

template <typename T>
rocblas_status (*rocblas_copy_strided_batched)(rocblas_handle handle,
                                               rocblas_int    n,
                                               const T*       x,
                                               rocblas_int    incx,
                                               rocblas_stride stridex,
                                               T*             y,
                                               rocblas_int    incy,
                                               rocblas_stride stridey,
                                               rocblas_int    batch_count);

template <>
static constexpr auto rocblas_copy_strided_batched<float> = rocblas_scopy_strided_batched;

template <>
static constexpr auto rocblas_copy_strided_batched<double> = rocblas_dcopy_strided_batched;

template <>
static constexpr auto
    rocblas_copy_strided_batched<rocblas_float_complex> = rocblas_ccopy_strided_batched;

template <>
static constexpr auto
    rocblas_copy_strided_batched<rocblas_double_complex> = rocblas_zcopy_strided_batched;

// swap
template <typename T>
rocblas_status (*rocblas_swap)(
    rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

template <>
static constexpr auto rocblas_swap<float> = rocblas_sswap;

template <>
static constexpr auto rocblas_swap<double> = rocblas_dswap;

template <>
static constexpr auto rocblas_swap<rocblas_float_complex> = rocblas_cswap;

template <>
static constexpr auto rocblas_swap<rocblas_double_complex> = rocblas_zswap;

// swap_batched
template <typename T>
rocblas_status (*rocblas_swap_batched)(rocblas_handle handle,
                                       rocblas_int    n,
                                       T*             x[],
                                       rocblas_int    incx,
                                       T*             y[],
                                       rocblas_int    incy,
                                       rocblas_int    batch_count);

template <>
static constexpr auto rocblas_swap_batched<float> = rocblas_sswap_batched;

template <>
static constexpr auto rocblas_swap_batched<double> = rocblas_dswap_batched;

template <>
static constexpr auto rocblas_swap_batched<rocblas_float_complex> = rocblas_cswap_batched;

template <>
static constexpr auto rocblas_swap_batched<rocblas_double_complex> = rocblas_zswap_batched;

// swap_strided_batched
template <typename T>
rocblas_status (*rocblas_swap_strided_batched)(rocblas_handle handle,
                                               rocblas_int    n,
                                               T*             x,
                                               rocblas_int    incx,
                                               rocblas_stride stridex,
                                               T*             y,
                                               rocblas_int    incy,
                                               rocblas_stride stridey,
                                               rocblas_int    batch_count);

template <>
static constexpr auto rocblas_swap_strided_batched<float> = rocblas_sswap_strided_batched;

template <>
static constexpr auto rocblas_swap_strided_batched<double> = rocblas_dswap_strided_batched;

template <>
static constexpr auto
    rocblas_swap_strided_batched<rocblas_float_complex> = rocblas_cswap_strided_batched;

template <>
static constexpr auto
    rocblas_swap_strided_batched<rocblas_double_complex> = rocblas_zswap_strided_batched;

// dot
template <typename T>
rocblas_status (*rocblas_dot)(rocblas_handle handle,
                              rocblas_int    n,
                              const T*       x,
                              rocblas_int    incx,
                              const T*       y,
                              rocblas_int    incy,
                              T*             result);

template <>
static constexpr auto rocblas_dot<float> = rocblas_sdot;

template <>
static constexpr auto rocblas_dot<double> = rocblas_ddot;

template <>
static constexpr auto rocblas_dot<rocblas_half> = rocblas_hdot;

template <>
static constexpr auto rocblas_dot<rocblas_bfloat16> = rocblas_bfdot;

template <>
static constexpr auto rocblas_dot<rocblas_float_complex> = rocblas_cdotu;

template <>
static constexpr auto rocblas_dot<rocblas_double_complex> = rocblas_zdotu;

// dotc
template <typename T>
rocblas_status (*rocblas_dotc)(rocblas_handle handle,
                               rocblas_int    n,
                               const T*       x,
                               rocblas_int    incx,
                               const T*       y,
                               rocblas_int    incy,
                               T*             result);

template <>
static constexpr auto rocblas_dotc<rocblas_float_complex> = rocblas_cdotc;

template <>
static constexpr auto rocblas_dotc<rocblas_double_complex> = rocblas_zdotc;

// dot_batched
template <typename T>
rocblas_status (*rocblas_dot_batched)(rocblas_handle handle,
                                      rocblas_int    n,
                                      const T* const x[],
                                      rocblas_int    incx,
                                      const T* const y[],
                                      rocblas_int    incy,
                                      rocblas_int    batch_count,
                                      T*             result);

template <>
static constexpr auto rocblas_dot_batched<float> = rocblas_sdot_batched;

template <>
static constexpr auto rocblas_dot_batched<double> = rocblas_ddot_batched;

template <>
static constexpr auto rocblas_dot_batched<rocblas_half> = rocblas_hdot_batched;

template <>
static constexpr auto rocblas_dot_batched<rocblas_bfloat16> = rocblas_bfdot_batched;

template <>
static constexpr auto rocblas_dot_batched<rocblas_float_complex> = rocblas_cdotu_batched;

template <>
static constexpr auto rocblas_dot_batched<rocblas_double_complex> = rocblas_zdotu_batched;

// dotc_batched
template <typename T>
rocblas_status (*rocblas_dotc_batched)(rocblas_handle handle,
                                       rocblas_int    n,
                                       const T* const x[],
                                       rocblas_int    incx,
                                       const T* const y[],
                                       rocblas_int    incy,
                                       rocblas_int    batch_count,
                                       T*             result);

template <>
static constexpr auto rocblas_dotc_batched<rocblas_float_complex> = rocblas_cdotc_batched;

template <>
static constexpr auto rocblas_dotc_batched<rocblas_double_complex> = rocblas_zdotc_batched;

// dot_strided_batched
template <typename T>
rocblas_status (*rocblas_dot_strided_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_dot_strided_batched<float> = rocblas_sdot_strided_batched;

template <>
static constexpr auto rocblas_dot_strided_batched<double> = rocblas_ddot_strided_batched;

template <>
static constexpr auto rocblas_dot_strided_batched<rocblas_half> = rocblas_hdot_strided_batched;

template <>
static constexpr auto rocblas_dot_strided_batched<rocblas_bfloat16> = rocblas_bfdot_strided_batched;

template <>
static constexpr auto
    rocblas_dot_strided_batched<rocblas_float_complex> = rocblas_cdotu_strided_batched;

template <>
static constexpr auto
    rocblas_dot_strided_batched<rocblas_double_complex> = rocblas_zdotu_strided_batched;

// dotc
template <typename T>
rocblas_status (*rocblas_dotc_strided_batched)(rocblas_handle handle,
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
static constexpr auto
    rocblas_dotc_strided_batched<rocblas_float_complex> = rocblas_cdotc_strided_batched;

template <>
static constexpr auto
    rocblas_dotc_strided_batched<rocblas_double_complex> = rocblas_zdotc_strided_batched;

// asum
template <typename T1, typename T2>
rocblas_status (*rocblas_asum)(
    rocblas_handle handle, rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <>
static constexpr auto rocblas_asum<float, float> = rocblas_sasum;

template <>
static constexpr auto rocblas_asum<double, double> = rocblas_dasum;

template <>
static constexpr auto rocblas_asum<rocblas_float_complex, float> = rocblas_scasum;

template <>
static constexpr auto rocblas_asum<rocblas_double_complex, double> = rocblas_dzasum;

// asum_batched
template <typename T1, typename T2>
rocblas_status (*rocblas_asum_batched)(rocblas_handle  handle,
                                       rocblas_int     n,
                                       const T1* const x[],
                                       rocblas_int     incx,
                                       rocblas_int     batch_count,
                                       T2*             result);

template <>
static constexpr auto rocblas_asum_batched<float, float> = rocblas_sasum_batched;

template <>
static constexpr auto rocblas_asum_batched<double, double> = rocblas_dasum_batched;

template <>
static constexpr auto rocblas_asum_batched<rocblas_float_complex, float> = rocblas_scasum_batched;

template <>
static constexpr auto rocblas_asum_batched<rocblas_double_complex, double> = rocblas_dzasum_batched;

// asum_strided_batched
template <typename T1, typename T2>
rocblas_status (*rocblas_asum_strided_batched)(rocblas_handle handle,
                                               rocblas_int    n,
                                               const T1*      x,
                                               rocblas_int    incx,
                                               rocblas_stride stridex,
                                               rocblas_int    batch_count,
                                               T2*            result);

template <>
static constexpr auto rocblas_asum_strided_batched<float, float> = rocblas_sasum_strided_batched;

template <>
static constexpr auto rocblas_asum_strided_batched<double, double> = rocblas_dasum_strided_batched;

template <>
static constexpr auto
    rocblas_asum_strided_batched<rocblas_float_complex, float> = rocblas_scasum_strided_batched;

template <>
static constexpr auto
    rocblas_asum_strided_batched<rocblas_double_complex, double> = rocblas_dzasum_strided_batched;

// nrm2
template <typename T1, typename T2>
rocblas_status (*rocblas_nrm2)(
    rocblas_handle handle, rocblas_int n, const T1* x, rocblas_int incx, T2* result);

template <>
static constexpr auto rocblas_nrm2<float, float> = rocblas_snrm2;

template <>
static constexpr auto rocblas_nrm2<double, double> = rocblas_dnrm2;

template <>
static constexpr auto rocblas_nrm2<rocblas_float_complex, float> = rocblas_scnrm2;

template <>
static constexpr auto rocblas_nrm2<rocblas_double_complex, double> = rocblas_dznrm2;

// nrm2_batched
template <typename T1, typename T2>
rocblas_status (*rocblas_nrm2_batched)(rocblas_handle  handle,
                                       rocblas_int     n,
                                       const T1* const x[],
                                       rocblas_int     incx,
                                       rocblas_int     batch_count,
                                       T2*             results);

template <>
static constexpr auto rocblas_nrm2_batched<float, float> = rocblas_snrm2_batched;

template <>
static constexpr auto rocblas_nrm2_batched<double, double> = rocblas_dnrm2_batched;

template <>
static constexpr auto rocblas_nrm2_batched<rocblas_float_complex, float> = rocblas_scnrm2_batched;

template <>
static constexpr auto rocblas_nrm2_batched<rocblas_double_complex, double> = rocblas_dznrm2_batched;

// nrm2_strided_batched
template <typename T1, typename T2>
rocblas_status (*rocblas_nrm2_strided_batched)(rocblas_handle handle,
                                               rocblas_int    n,
                                               const T1*      x,
                                               rocblas_int    incx,
                                               rocblas_stride stridex,
                                               rocblas_int    batch_count,
                                               T2*            results);

template <>
static constexpr auto rocblas_nrm2_strided_batched<float, float> = rocblas_snrm2_strided_batched;

template <>
static constexpr auto rocblas_nrm2_strided_batched<double, double> = rocblas_dnrm2_strided_batched;

template <>
static constexpr auto
    rocblas_nrm2_strided_batched<rocblas_float_complex, float> = rocblas_scnrm2_strided_batched;

template <>
static constexpr auto
    rocblas_nrm2_strided_batched<rocblas_double_complex, double> = rocblas_dznrm2_strided_batched;

// iamax and iamin need to be full functions rather than references, in order
// to allow them to be passed as template arguments
//
// iamax and iamin.
//

//
// Define the signature type.
//
template <typename T>
using rocblas_iamax_iamin_t = rocblas_status (*)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

//
// iamax
//
template <typename T>
rocblas_iamax_iamin_t<T> rocblas_iamax;

template <>
static constexpr auto rocblas_iamax<float> = rocblas_isamax;

template <>
static constexpr auto rocblas_iamax<double> = rocblas_idamax;

template <>
static constexpr auto rocblas_iamax<rocblas_float_complex> = rocblas_icamax;

template <>
static constexpr auto rocblas_iamax<rocblas_double_complex> = rocblas_izamax;

//
// iamin
//
template <typename T>
rocblas_iamax_iamin_t<T> rocblas_iamin;

template <>
static constexpr auto rocblas_iamin<float> = rocblas_isamin;

template <>
static constexpr auto rocblas_iamin<double> = rocblas_idamin;

template <>
static constexpr auto rocblas_iamin<rocblas_float_complex> = rocblas_icamin;

template <>
static constexpr auto rocblas_iamin<rocblas_double_complex> = rocblas_izamin;

// axpy
template <typename T>
rocblas_status (*rocblas_axpy)(rocblas_handle handle,
                               rocblas_int    n,
                               const T*       alpha,
                               const T*       x,
                               rocblas_int    incx,
                               T*             y,
                               rocblas_int    incy);

template <>
static constexpr auto rocblas_axpy<rocblas_half> = rocblas_haxpy;

template <>
static constexpr auto rocblas_axpy<float> = rocblas_saxpy;

template <>
static constexpr auto rocblas_axpy<double> = rocblas_daxpy;

template <>
static constexpr auto rocblas_axpy<rocblas_float_complex> = rocblas_caxpy;

template <>
static constexpr auto rocblas_axpy<rocblas_double_complex> = rocblas_zaxpy;

// rot
template <typename T, typename U = T, typename V = T>
rocblas_status (*rocblas_rot)(rocblas_handle handle,
                              rocblas_int    n,
                              T*             x,
                              rocblas_int    incx,
                              T*             y,
                              rocblas_int    incy,
                              const U*       c,
                              const V*       s);

template <>
static constexpr auto rocblas_rot<float> = rocblas_srot;

template <>
static constexpr auto rocblas_rot<double> = rocblas_drot;

template <>
static constexpr auto
    rocblas_rot<rocblas_float_complex, float, rocblas_float_complex> = rocblas_crot;

template <>
static constexpr auto rocblas_rot<rocblas_float_complex, float, float> = rocblas_csrot;

template <>
static constexpr auto
    rocblas_rot<rocblas_double_complex, double, rocblas_double_complex> = rocblas_zrot;

template <>
static constexpr auto rocblas_rot<rocblas_double_complex, double, double> = rocblas_zdrot;

// rot_batched
template <typename T, typename U = T, typename V = T>
rocblas_status (*rocblas_rot_batched)(rocblas_handle handle,
                                      rocblas_int    n,
                                      T* const       x[],
                                      rocblas_int    incx,
                                      T* const       y[],
                                      rocblas_int    incy,
                                      const U*       c,
                                      const V*       s,
                                      rocblas_int    batch_count);

template <>
static constexpr auto rocblas_rot_batched<float> = rocblas_srot_batched;

template <>
static constexpr auto rocblas_rot_batched<double> = rocblas_drot_batched;

template <>
static constexpr auto
    rocblas_rot_batched<rocblas_float_complex, float, rocblas_float_complex> = rocblas_crot_batched;

template <>
static constexpr auto
    rocblas_rot_batched<rocblas_float_complex, float, float> = rocblas_csrot_batched;

template <>
static constexpr auto rocblas_rot_batched<rocblas_double_complex,
                                          double,
                                          rocblas_double_complex> = rocblas_zrot_batched;

template <>
static constexpr auto
    rocblas_rot_batched<rocblas_double_complex, double, double> = rocblas_zdrot_batched;

// rot_strided_batched
template <typename T, typename U = T, typename V = T>
rocblas_status (*rocblas_rot_strided_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_rot_strided_batched<float> = rocblas_srot_strided_batched;

template <>
static constexpr auto rocblas_rot_strided_batched<double> = rocblas_drot_strided_batched;

template <>
static constexpr auto
    rocblas_rot_strided_batched<rocblas_float_complex,
                                float,
                                rocblas_float_complex> = rocblas_crot_strided_batched;

template <>
static constexpr auto rocblas_rot_strided_batched<rocblas_float_complex,
                                                  float,
                                                  float> = rocblas_csrot_strided_batched;

template <>
static constexpr auto
    rocblas_rot_strided_batched<rocblas_double_complex,
                                double,
                                rocblas_double_complex> = rocblas_zrot_strided_batched;

template <>
static constexpr auto rocblas_rot_strided_batched<rocblas_double_complex,
                                                  double,
                                                  double> = rocblas_zdrot_strided_batched;

// rotg
template <typename T, typename U = T>
rocblas_status (*rocblas_rotg)(rocblas_handle handle, T* a, T* b, U* c, T* s);

template <>
static constexpr auto rocblas_rotg<float> = rocblas_srotg;

template <>
static constexpr auto rocblas_rotg<double> = rocblas_drotg;

template <>
static constexpr auto rocblas_rotg<rocblas_float_complex, float> = rocblas_crotg;

template <>
static constexpr auto rocblas_rotg<rocblas_double_complex, double> = rocblas_zrotg;

// rotg_batched
template <typename T, typename U = T>
rocblas_status (*rocblas_rotg_batched)(rocblas_handle handle,
                                       T* const       a[],
                                       T* const       b[],
                                       U* const       c[],
                                       T* const       s[],
                                       rocblas_int    batch_count);

template <>
static constexpr auto rocblas_rotg_batched<float> = rocblas_srotg_batched;

template <>
static constexpr auto rocblas_rotg_batched<double> = rocblas_drotg_batched;

template <>
static constexpr auto rocblas_rotg_batched<rocblas_float_complex, float> = rocblas_crotg_batched;

template <>
static constexpr auto rocblas_rotg_batched<rocblas_double_complex, double> = rocblas_zrotg_batched;

//rotg_strided_batched
template <typename T, typename U = T>
rocblas_status (*rocblas_rotg_strided_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_rotg_strided_batched<float> = rocblas_srotg_strided_batched;

template <>
static constexpr auto rocblas_rotg_strided_batched<double> = rocblas_drotg_strided_batched;

template <>
static constexpr auto
    rocblas_rotg_strided_batched<rocblas_float_complex, float> = rocblas_crotg_strided_batched;

template <>
static constexpr auto
    rocblas_rotg_strided_batched<rocblas_double_complex, double> = rocblas_zrotg_strided_batched;

//rotm
template <typename T>
rocblas_status (*rocblas_rotm)(rocblas_handle handle,
                               rocblas_int    n,
                               T*             x,
                               rocblas_int    incx,
                               T*             y,
                               rocblas_int    incy,
                               const T*       param);

template <>
static constexpr auto rocblas_rotm<float> = rocblas_srotm;

template <>
static constexpr auto rocblas_rotm<double> = rocblas_drotm;

// rotm_batched
template <typename T>
rocblas_status (*rocblas_rotm_batched)(rocblas_handle handle,
                                       rocblas_int    n,
                                       T* const       x[],
                                       rocblas_int    incx,
                                       T* const       y[],
                                       rocblas_int    incy,
                                       const T* const param[],
                                       rocblas_int    batch_count);
template <>
static constexpr auto rocblas_rotm_batched<float> = rocblas_srotm_batched;

template <>
static constexpr auto rocblas_rotm_batched<double> = rocblas_drotm_batched;

// rotm_strided_batched
template <typename T>
rocblas_status (*rocblas_rotm_strided_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_rotm_strided_batched<float> = rocblas_srotm_strided_batched;

template <>
static constexpr auto rocblas_rotm_strided_batched<double> = rocblas_drotm_strided_batched;

//rotmg
template <typename T>
rocblas_status (*rocblas_rotmg)(rocblas_handle handle, T* d1, T* d2, T* x1, const T* y1, T* param);

template <>
static constexpr auto rocblas_rotmg<float> = rocblas_srotmg;

template <>
static constexpr auto rocblas_rotmg<double> = rocblas_drotmg;

//rotmg_batched
template <typename T>
rocblas_status (*rocblas_rotmg_batched)(rocblas_handle handle,
                                        T* const       d1[],
                                        T* const       d2[],
                                        T* const       x1[],
                                        const T* const y1[],
                                        T* const       param[],
                                        rocblas_int    batch_count);

template <>
static constexpr auto rocblas_rotmg_batched<float> = rocblas_srotmg_batched;

template <>
static constexpr auto rocblas_rotmg_batched<double> = rocblas_drotmg_batched;

//rotmg_strided_batched
template <typename T>
rocblas_status (*rocblas_rotmg_strided_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_rotmg_strided_batched<float> = rocblas_srotmg_strided_batched;

template <>
static constexpr auto rocblas_rotmg_strided_batched<double> = rocblas_drotmg_strided_batched;

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// ger
template <typename T>
rocblas_status (*rocblas_ger)(rocblas_handle handle,
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
static constexpr auto rocblas_ger<float> = rocblas_sger;

template <>
static constexpr auto rocblas_ger<double> = rocblas_dger;

template <typename T>
rocblas_status (*rocblas_ger_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_ger_batched<float> = rocblas_sger_batched;

template <>
static constexpr auto rocblas_ger_batched<double> = rocblas_dger_batched;

template <typename T>
rocblas_status (*rocblas_ger_strided_batched)(rocblas_handle handle,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T*       x,
                                              rocblas_int    incx,
                                              rocblas_int    stride_x,
                                              const T*       y,
                                              rocblas_int    incy,
                                              rocblas_int    stride_y,
                                              T*             A,
                                              rocblas_int    lda,
                                              rocblas_int    stride_a,
                                              rocblas_int    batch_count);

template <>
static constexpr auto rocblas_ger_strided_batched<float> = rocblas_sger_strided_batched;

template <>
static constexpr auto rocblas_ger_strided_batched<double> = rocblas_dger_strided_batched;

// syr
template <typename T>
rocblas_status (*rocblas_syr)(rocblas_handle handle,
                              rocblas_fill   uplo,
                              rocblas_int    n,
                              const T*       alpha,
                              const T*       x,
                              rocblas_int    incx,
                              T*             A,
                              rocblas_int    lda);

template <>
static constexpr auto rocblas_syr<float> = rocblas_ssyr;

template <>
static constexpr auto rocblas_syr<double> = rocblas_dsyr;

// syr strided batched
template <typename T>
rocblas_status (*rocblas_syr_strided_batched)(rocblas_handle handle,
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
static constexpr auto rocblas_syr_strided_batched<float> = rocblas_ssyr_strided_batched;

template <>
static constexpr auto rocblas_syr_strided_batched<double> = rocblas_dsyr_strided_batched;

// syr batched
template <typename T>
rocblas_status (*rocblas_syr_batched)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T* const x[],
                                      rocblas_int    incx,
                                      T*             A[],
                                      rocblas_int    lda,
                                      rocblas_int    batch_count);

template <>
static constexpr auto rocblas_syr_batched<float> = rocblas_ssyr_batched;

template <>
static constexpr auto rocblas_syr_batched<double> = rocblas_dsyr_batched;

// gemv
template <typename T>
rocblas_status (*rocblas_gemv)(rocblas_handle    handle,
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
static constexpr auto rocblas_gemv<float> = rocblas_sgemv;

template <>
static constexpr auto rocblas_gemv<double> = rocblas_dgemv;

template <>
static constexpr auto rocblas_gemv<rocblas_float_complex> = rocblas_cgemv;

template <>
static constexpr auto rocblas_gemv<rocblas_double_complex> = rocblas_zgemv;

// gemv_strided_batched
template <typename T>
rocblas_status (*rocblas_gemv_strided_batched)(rocblas_handle    handle,
                                               rocblas_operation transA,
                                               rocblas_int       m,
                                               rocblas_int       n,
                                               const T*          alpha,
                                               const T*          A,
                                               rocblas_int       lda,
                                               rocblas_int       stride_a,
                                               const T*          x,
                                               rocblas_int       incx,
                                               rocblas_int       stride_x,
                                               const T*          beta,
                                               T*                y,
                                               rocblas_int       incy,
                                               rocblas_int       stride_y,
                                               rocblas_int       batch_count);

template <>
static constexpr auto rocblas_gemv_strided_batched<float> = rocblas_sgemv_strided_batched;

template <>
static constexpr auto rocblas_gemv_strided_batched<double> = rocblas_dgemv_strided_batched;

template <>
static constexpr auto
    rocblas_gemv_strided_batched<rocblas_float_complex> = rocblas_cgemv_strided_batched;

template <>
static constexpr auto
    rocblas_gemv_strided_batched<rocblas_double_complex> = rocblas_zgemv_strided_batched;

// gemv_batched
template <typename T>
rocblas_status (*rocblas_gemv_batched)(rocblas_handle    handle,
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
static constexpr auto rocblas_gemv_batched<float> = rocblas_sgemv_batched;

template <>
static constexpr auto rocblas_gemv_batched<double> = rocblas_dgemv_batched;

template <>
static constexpr auto rocblas_gemv_batched<rocblas_float_complex> = rocblas_cgemv_batched;

template <>
static constexpr auto rocblas_gemv_batched<rocblas_double_complex> = rocblas_zgemv_batched;

// trsv
template <typename T>
rocblas_status (*rocblas_trsv)(rocblas_handle    handle,
                               rocblas_fill      uplo,
                               rocblas_operation transA,
                               rocblas_diagonal  diag,
                               rocblas_int       m,
                               const T*          A,
                               rocblas_int       lda,
                               T*                x,
                               rocblas_int       incx);

template <>
static constexpr auto rocblas_trsv<float> = rocblas_strsv;

template <>
static constexpr auto rocblas_trsv<double> = rocblas_dtrsv;

// symv
template <typename T>
rocblas_status (*rocblas_symv)(rocblas_handle handle,
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
rocblas_status (*rocblas_geam)(rocblas_handle    handle,
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
static constexpr auto rocblas_geam<float> = rocblas_sgeam;

template <>
static constexpr auto rocblas_geam<double> = rocblas_dgeam;

// gemm
template <typename T>
rocblas_status (*rocblas_gemm)(rocblas_handle    handle,
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
static constexpr auto rocblas_gemm<rocblas_half> = rocblas_hgemm;

template <>
static constexpr auto rocblas_gemm<float> = rocblas_sgemm;

template <>
static constexpr auto rocblas_gemm<double> = rocblas_dgemm;

template <>
static constexpr auto rocblas_gemm<rocblas_float_complex> = rocblas_cgemm;

template <>
static constexpr auto rocblas_gemm<rocblas_double_complex> = rocblas_zgemm;

// gemm_batched
template <typename T>
rocblas_status (*rocblas_gemm_batched)(rocblas_handle    handle,
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
static constexpr auto rocblas_gemm_batched<rocblas_half> = rocblas_hgemm_batched;

template <>
static constexpr auto rocblas_gemm_batched<float> = rocblas_sgemm_batched;

template <>
static constexpr auto rocblas_gemm_batched<double> = rocblas_dgemm_batched;

template <>
static constexpr auto rocblas_gemm_batched<rocblas_float_complex> = rocblas_cgemm_batched;

template <>
static constexpr auto rocblas_gemm_batched<rocblas_double_complex> = rocblas_zgemm_batched;

// gemm_strided_batched
template <typename T>
rocblas_status (*rocblas_gemm_strided_batched)(rocblas_handle    handle,
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
static constexpr auto rocblas_gemm_strided_batched<rocblas_half> = rocblas_hgemm_strided_batched;

template <>
static constexpr auto rocblas_gemm_strided_batched<float> = rocblas_sgemm_strided_batched;

template <>
static constexpr auto rocblas_gemm_strided_batched<double> = rocblas_dgemm_strided_batched;

template <>
static constexpr auto
    rocblas_gemm_strided_batched<rocblas_float_complex> = rocblas_cgemm_strided_batched;

template <>
static constexpr auto
    rocblas_gemm_strided_batched<rocblas_double_complex> = rocblas_zgemm_strided_batched;

#if 0
// trmm
template <typename T>
rocblas_status (*rocblas_trmm)(rocblas_handle    handle,
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
static constexpr auto rocblas_trmm<float> = rocblas_strmm;

template <>
static constexpr auto rocblas_trmm<double> = rocblas_dtrmm;
#endif

// trsm
template <typename T>
rocblas_status (*rocblas_trsm)(rocblas_handle    handle,
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
static constexpr auto rocblas_trsm<float> = rocblas_strsm;

template <>
static constexpr auto rocblas_trsm<double> = rocblas_dtrsm;

// trtri
template <typename T>
rocblas_status (*rocblas_trtri)(rocblas_handle   handle,
                                rocblas_fill     uplo,
                                rocblas_diagonal diag,
                                rocblas_int      n,
                                T*               A,
                                rocblas_int      lda,
                                T*               invA,
                                rocblas_int      ldinvA);

template <>
static constexpr auto rocblas_trtri<float> = rocblas_strtri;

template <>
static constexpr auto rocblas_trtri<double> = rocblas_dtrtri;

// trtri_batched
template <typename T>
rocblas_status (*rocblas_trtri_batched)(rocblas_handle   handle,
                                        rocblas_fill     uplo,
                                        rocblas_diagonal diag,
                                        rocblas_int      n,
                                        T*               A[],
                                        rocblas_int      lda,
                                        T*               invA[],
                                        rocblas_int      ldinvA,
                                        rocblas_int      batch_count);

template <>
static constexpr auto rocblas_trtri_batched<float> = rocblas_strtri_batched;

template <>
static constexpr auto rocblas_trtri_batched<double> = rocblas_dtrtri_batched;

// trtri_strided_batched
template <typename T>
rocblas_status (*rocblas_trtri_strided_batched)(rocblas_handle   handle,
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
static constexpr auto rocblas_trtri_strided_batched<float> = rocblas_strtri_strided_batched;

template <>
static constexpr auto rocblas_trtri_strided_batched<double> = rocblas_dtrtri_strided_batched;

#endif // _ROCBLAS_HPP_
