/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_trtri_batched.hpp"

extern "C" {
rocblas_status rocblas_strtri_batched(rocblas_handle      handle,
                                      rocblas_fill        uplo,
                                      rocblas_diagonal    diag,
                                      rocblas_int         n,
                                      const float* const  A[],
                                      rocblas_int         lda,
                                      float*              invA[],
                                      rocblas_int         ldinvA,
                                      rocblas_int         batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_batched_impl<NB>(
        handle, uplo, diag, n, A, 0, lda, invA, 0, ldinvA, batch_count);
}

rocblas_status rocblas_dtrtri_batched(rocblas_handle      handle,
                                      rocblas_fill        uplo,
                                      rocblas_diagonal    diag,
                                      rocblas_int         n,
                                      const double* const A[],
                                      rocblas_int         lda,
                                      double*             invA[],
                                      rocblas_int         ldinvA,
                                      rocblas_int         batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_batched_impl<NB>(
        handle, uplo, diag, n, A, 0, lda, invA, 0, ldinvA, batch_count);
}

} // extern "C"
