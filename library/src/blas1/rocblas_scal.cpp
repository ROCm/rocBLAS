/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_scal.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status
rocblas_sscal(rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx)
{
    return rocblas_scal_template<float>(handle, n, alpha, x, incx);
}

extern "C" rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx)
{
    return rocblas_scal_template<double>(handle, n, alpha, x, incx);
}

extern "C" rocblas_status rocblas_cscal(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_float_complex* alpha,
                                        rocblas_float_complex* x,
                                        rocblas_int incx)
{
    return rocblas_scal_template<rocblas_float_complex>(handle, n, alpha, x, incx);
}

extern "C" rocblas_status rocblas_zscal(rocblas_handle handle,
                                        rocblas_int n,
                                        const rocblas_double_complex* alpha,
                                        rocblas_double_complex* x,
                                        rocblas_int incx)
{
    return rocblas_scal_template<rocblas_double_complex>(handle, n, alpha, x, incx);
}
