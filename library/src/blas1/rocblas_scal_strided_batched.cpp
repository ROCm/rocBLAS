/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_scal_strided_batched.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sscal_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   alpha,
                                             float*         x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             rocblas_int    batch_count)
{
    return rocblas_scal_strided_batched_impl(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_dscal_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  alpha,
                                             double*        x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             rocblas_int    batch_count)
{
    return rocblas_scal_strided_batched_impl(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_cscal_strided_batched(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             rocblas_float_complex*       x,
                                             rocblas_int                  incx,
                                             rocblas_int                  stridex,
                                             rocblas_int                  batch_count)
{
    return rocblas_scal_strided_batched_impl(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_zscal_strided_batched(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             rocblas_double_complex*       x,
                                             rocblas_int                   incx,
                                             rocblas_int                   stridex,
                                             rocblas_int                   batch_count)
{
    return rocblas_scal_strided_batched_impl(handle, n, alpha, x, incx, stridex, batch_count);
}

// Scal with a real alpha & complex vector
rocblas_status rocblas_csscal_strided_batched(rocblas_handle         handle,
                                              rocblas_int            n,
                                              const float*           alpha,
                                              rocblas_float_complex* x,
                                              rocblas_int            incx,
                                              rocblas_int            stridex,
                                              rocblas_int            batch_count)
{
    return rocblas_scal_strided_batched_impl(handle, n, alpha, x, incx, stridex, batch_count);
}

rocblas_status rocblas_zdscal_strided_batched(rocblas_handle          handle,
                                              rocblas_int             n,
                                              const double*           alpha,
                                              rocblas_double_complex* x,
                                              rocblas_int             incx,
                                              rocblas_int             stridex,
                                              rocblas_int             batch_count)
{
    return rocblas_scal_strided_batched_impl(handle, n, alpha, x, incx, stridex, batch_count);
}

} // extern "C"
