/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_dot.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_sdot(rocblas_handle handle,
                                       rocblas_int n,
                                       const float* x,
                                       rocblas_int incx,
                                       const float* y,
                                       rocblas_int incy,
                                       float* result)
{
    return rocblas_dot_template<float>(handle, n, x, incx, y, incy, result);
}

extern "C" rocblas_status rocblas_ddot(rocblas_handle handle,
                                       rocblas_int n,
                                       const double* x,
                                       rocblas_int incx,
                                       const double* y,
                                       rocblas_int incy,
                                       double* result)
{
    return rocblas_dot_template<double>(handle, n, x, incx, y, incy, result);
}

extern "C" rocblas_status rocblas_cdotu(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_float_complex* x,
                                        rocblas_int incx,
                                        const rocblas_float_complex* y,
                                        rocblas_int incy,
                                        rocblas_float_complex* result)
{
    return rocblas_dot_template<rocblas_float_complex>(handle, n, x, incx, y, incy, result);
}

extern "C" rocblas_status rocblas_zdotu(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_double_complex* x,
                                        rocblas_int incx,
                                        const rocblas_double_complex* y,
                                        rocblas_int incy,
                                        rocblas_double_complex* result)
{
    return rocblas_dot_template<rocblas_double_complex>(handle, n, x, incx, y, incy, result);
}
