/* ************************************************************************
 *  * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *  *
 *  * ************************************************************************ */
#include "rocblas_trtri.hpp"

/*
 * ===========================================================================
 *    C interface
 * ===========================================================================
 */

extern "C" {
rocblas_status rocblas_strtri(rocblas_handle   handle,
                              rocblas_fill     uplo,
                              rocblas_diagonal diag,
                              rocblas_int      n,
                              const float*     A,
                              rocblas_int      lda,
                              float*           invA,
                              rocblas_int      ldinvA)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

rocblas_status rocblas_dtrtri(rocblas_handle   handle,
                              rocblas_fill     uplo,
                              rocblas_diagonal diag,
                              rocblas_int      n,
                              const double*    A,
                              rocblas_int      lda,
                              double*          invA,
                              rocblas_int      ldinvA)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

} // extern "C"
