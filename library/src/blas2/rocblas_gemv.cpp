/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_gemv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_sgemv(rocblas_handle handle,
                                        rocblas_operation transA,
                                        rocblas_int m,
                                        rocblas_int n,
                                        const float* alpha,
                                        const float* A,
                                        rocblas_int lda,
                                        const float* x,
                                        rocblas_int incx,
                                        const float* beta,
                                        float* y,
                                        rocblas_int incy)
{
    return rocblas_gemv_template<float>(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

extern "C" rocblas_status rocblas_dgemv(rocblas_handle handle,
                                        rocblas_operation transA,
                                        rocblas_int m,
                                        rocblas_int n,
                                        const double* alpha,
                                        const double* A,
                                        rocblas_int lda,
                                        const double* x,
                                        rocblas_int incx,
                                        const double* beta,
                                        double* y,
                                        rocblas_int incy)
{
    return rocblas_gemv_template<double>(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
