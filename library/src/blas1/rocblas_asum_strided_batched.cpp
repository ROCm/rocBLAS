/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_asum_strided_batched.hpp"


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sasum_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             float*         results,
                                             rocblas_int    batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, 0, incx, stridex, results, batch_count);
}

rocblas_status rocblas_dasum_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             double*        results,
                                             rocblas_int    batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, 0, incx, stridex, results, batch_count);
}

rocblas_status rocblas_scasum_strided_batched(rocblas_handle               handle,
                                              rocblas_int                  n,
                                              const rocblas_float_complex* x,
                                              rocblas_int                  incx,
                                              rocblas_int                  stridex,
                                              float*                       results,
                                              rocblas_int                  batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, 0, incx, stridex, results, batch_count);
}

rocblas_status rocblas_dzasum_strided_batched(rocblas_handle                handle,
                                              rocblas_int                   n,
                                              const rocblas_double_complex* x,
                                              rocblas_int                   incx,
                                              rocblas_int                   stridex,
                                              double*                       results,
                                              rocblas_int                   batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, 0, incx, stridex, results, batch_count);
}

} // extern "C"
