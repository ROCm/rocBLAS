/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_trtri_strided_batched.hpp"

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C interface
 *    This function is called by trsm
 * ===========================================================================
 */

extern "C" {
rocblas_status rocblas_strtri_strided_batched(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const float*     A,
                                              rocblas_int      lda,
                                              rocblas_int      bsa,
                                              float*           invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      bsinvA,
                                              rocblas_int      batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_strided_batched_impl<NB>(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

rocblas_status rocblas_dtrtri_strided_batched(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const double*    A,
                                              rocblas_int      lda,
                                              rocblas_int      bsa,
                                              double*          invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      bsinvA,
                                              rocblas_int      batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_strided_batched_impl<NB>(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

} // extern "C"
