/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
namespace trtri { // must use namespace to avoid multply definiton
#include "trtri.hpp"
}

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C interface
 * ===========================================================================
 */

// because of shared memory size, the IB must be <= 64, typically 64 for s, d, c, but 32 for z
// typically, only a small matrix A is inverted by trtri, so if n is too big, trtri is not
// implemented
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

    return trtri::rocblas_trtri_template<float, 64>(handle, uplo, diag, n, A, lda, invA, ldinvA);
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

    return trtri::rocblas_trtri_template<double, 64>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}
