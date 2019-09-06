/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_scal.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sscal(
    rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx)
{
    return rocblas_scal_impl(handle, n, alpha, x, incx);
}

rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
{
    return rocblas_scal_impl(handle, n, alpha, x, incx);
}

rocblas_status rocblas_cscal(rocblas_handle               handle,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             rocblas_float_complex*       x,
                             rocblas_int                  incx)
{
    return rocblas_scal_impl(handle, n, alpha, x, incx);
}

rocblas_status rocblas_zscal(rocblas_handle                handle,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             rocblas_double_complex*       x,
                             rocblas_int                   incx)
{
    return rocblas_scal_impl(handle, n, alpha, x, incx);
}

// Scal with a real alpha & complex vector
rocblas_status rocblas_csscal(rocblas_handle         handle,
                              rocblas_int            n,
                              const float*           alpha,
                              rocblas_float_complex* x,
                              rocblas_int            incx)
{
    return rocblas_scal_impl(handle, n, alpha, x, incx);
}

rocblas_status rocblas_zdscal(rocblas_handle          handle,
                              rocblas_int             n,
                              const double*           alpha,
                              rocblas_double_complex* x,
                              rocblas_int             incx)
{
    return rocblas_scal_impl(handle, n, alpha, x, incx);
}

} // extern "C"
