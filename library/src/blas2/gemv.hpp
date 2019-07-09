/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _GEMV_HPP_
#define _GEMV_HPP_
#include "rocblas.h"

template <typename T>
rocblas_status rocblas_gemv_template(rocblas_handle    handle,
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
rocblas_status rocblas_gemv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const float*      alpha,
                                     const float*      A,
                                     rocblas_int       lda,
                                     const float*      x,
                                     rocblas_int       incx,
                                     const float*      beta,
                                     float*            y,
                                     rocblas_int       incy)

{
    return rocblas_sgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
rocblas_status rocblas_gemv_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const double*     alpha,
                                     const double*     A,
                                     rocblas_int       lda,
                                     const double*     x,
                                     rocblas_int       incx,
                                     const double*     beta,
                                     double*           y,
                                     rocblas_int       incy)

{
    return rocblas_dgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

#endif // _GEMV_HPP_
