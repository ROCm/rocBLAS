/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_asum_batched.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sasum_batched(rocblas_handle     handle,
                                     rocblas_int        n,
                                     const float* const x[],
                                     rocblas_int        incx,
                                     float*             results,
                                     rocblas_int        batch_count)
{
    return rocblas_asum_batched(handle, n, x, 0, incx, results, batch_count);
}

rocblas_status rocblas_dasum_batched(rocblas_handle      handle,
                                     rocblas_int         n,
                                     const double* const x[],
                                     rocblas_int         incx,
                                     double*             results,
                                     rocblas_int         batch_count)
{
    return rocblas_asum_batched(handle, n, x, 0, incx, results, batch_count);
}

rocblas_status rocblas_scasum_batched(rocblas_handle                     handle,
                                      rocblas_int                        n,
                                      const rocblas_float_complex* const x[],
                                      rocblas_int                        incx,
                                      float*                             results,
                                      rocblas_int                        batch_count)
{
    return rocblas_asum_batched(handle, n, x, 0, incx, results, batch_count);
}

rocblas_status rocblas_dzasum_batched(rocblas_handle                      handle,
                                      rocblas_int                         n,
                                      const rocblas_double_complex* const x[],
                                      rocblas_int                         incx,
                                      double*                             results,
                                      rocblas_int                         batch_count)
{
    return rocblas_asum_batched(handle, n, x, 0, incx, results, batch_count);
}

} // extern "C"
