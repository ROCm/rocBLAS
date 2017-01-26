/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "trtri_batched.hpp"



/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C interface
     *    This function is called by trsm
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_strtri_batched(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_diagonal diag,
    rocblas_int n,
    float *A, rocblas_int lda, rocblas_int bsa,
    float *invA, rocblas_int ldinvA, rocblas_int bsinvA,
    rocblas_int batch_count)
{
    return rocblas_trtri_batched_template<float>(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}


extern "C"
rocblas_status
rocblas_dtrtri_batched(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_diagonal diag,
    rocblas_int n,
    double *A, rocblas_int lda, rocblas_int bsa,
    double *invA, rocblas_int ldinvA, rocblas_int bsinvA,
    rocblas_int batch_count)
{
    return rocblas_trtri_batched_template<double>(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

