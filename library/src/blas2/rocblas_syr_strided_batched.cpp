/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_syr_strided_batched.hpp"


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_ssyr_strided_batched(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const float*   alpha,
                            const float*   x,
                            rocblas_int    incx,
                            rocblas_int    stridex,
                            float*         A,
                            rocblas_int    lda,
                            rocblas_int    strideA,
                            rocblas_int    batch_count)
{
    return rocblas_syr_strided_batched(handle, uplo, n, alpha, x, 0, incx, stridex, A, 0, lda, strideA, batch_count);
}

rocblas_status rocblas_dsyr_strided_batched(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const double*  alpha,
                            const double*  x,
                            rocblas_int    incx,
                            rocblas_int    stridex,
                            double*        A,
                            rocblas_int    lda,
                            rocblas_int    strideA,
                            rocblas_int    batch_count)
{
    return rocblas_syr_strided_batched(handle, uplo, n, alpha, x, 0, incx, stridex, A, 0, lda, strideA, batch_count);
}

} // extern "C"
