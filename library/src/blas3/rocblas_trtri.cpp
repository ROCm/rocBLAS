/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "trtri.hpp"
#include "trtri_trsm.hpp"

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C interface
 * ===========================================================================
 */

// trtri is usually called by trsm

extern "C" rocblas_status rocblas_strtri(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_diagonal diag,
                                         rocblas_int n,
                                         const float* A,
                                         rocblas_int lda,
                                         float* invA,
                                         rocblas_int ldinvA)
{

    return trtri::rocblas_trtri_template<float>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

extern "C" rocblas_status rocblas_dtrtri(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_diagonal diag,
                                         rocblas_int n,
                                         const double* A,
                                         rocblas_int lda,
                                         double* invA,
                                         rocblas_int ldinvA)
{

    return trtri::rocblas_trtri_template<double>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}
