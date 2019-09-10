/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_syr_batched.hpp"


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_ssyr_batched(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const float*   alpha,
                            const float* const x[],
                            rocblas_int    incx,
                            float*         A[],
                            rocblas_int    lda,
                            rocblas_int    batch_count)
{
    return rocblas_syr_batched(handle, uplo, n, alpha, x, 0, incx, A, 0, lda, batch_count);
}

rocblas_status rocblas_dsyr_batched(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const double*  alpha,
                            const double* const x[],
                            rocblas_int    incx,
                            double*        A[],
                            rocblas_int    lda,
                            rocblas_int    batch_count)
{
    return rocblas_syr_batched(handle, uplo, n, alpha, x, 0, incx, A, 0, lda, batch_count);
}

} // extern "C"
