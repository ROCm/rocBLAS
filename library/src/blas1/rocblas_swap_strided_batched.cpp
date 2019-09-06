/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_swap_strided_batched.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sswap_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             float*         x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             float*         y,
                                             rocblas_int    incy,
                                             rocblas_int    stridey,
                                             rocblas_int    batch_count)
{
    return rocblas_swap_strided_batched(handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count);
}

rocblas_status rocblas_dswap_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             double*        x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             double*        y,
                                             rocblas_int    incy,
                                             rocblas_int    stridey,
                                             rocblas_int    batch_count)
{
    return rocblas_swap_strided_batched(handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count);
}

rocblas_status rocblas_cswap_strided_batched(rocblas_handle         handle,
                                             rocblas_int            n,
                                             rocblas_float_complex* x,
                                             rocblas_int            incx,
                                             rocblas_int            stridex,
                                             rocblas_float_complex* y,
                                             rocblas_int            incy,
                                             rocblas_int            stridey,
                                             rocblas_int            batch_count)
{
    return rocblas_swap_strided_batched(handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count);
}

rocblas_status rocblas_zswap_strided_batched(rocblas_handle          handle,
                                             rocblas_int             n,
                                             rocblas_double_complex* x,
                                             rocblas_int             incx,
                                             rocblas_int             stridex,
                                             rocblas_double_complex* y,
                                             rocblas_int             incy,
                                             rocblas_int             stridey,
                                             rocblas_int             batch_count)
{
    return rocblas_swap_strided_batched(handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count);
}

} // extern "C"
