/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _TRTRI_BATCHED_HPP_
#define _TRTRI_BATCHED_HPP_

#include <hip/hip_runtime.h>
#include "definitions.h"
#include "trtri_device.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

// flag indicate whether write into A or invA
template <typename T, rocblas_int NB>
__global__ void trtri_kernel_batched(rocblas_fill uplo,
                                     rocblas_diagonal diag,
                                     rocblas_int n,
                                     T* A,
                                     rocblas_int lda,
                                     rocblas_int bsa,
                                     T* invA,
                                     rocblas_int ldinvA,
                                     rocblas_int bsinvA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    T* individual_A    = A + hipBlockIdx_z * bsa;
    T* individual_invA = invA + hipBlockIdx_z * bsinvA;

    trtri_device<T, NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix  A

        inv(A);

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
    @param[in]
    diag      rocblas_diagonal.
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         rocblas_int.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    bsa       rocblas_int
             "batch stride a": stride from the start of one "A" matrix to the next
    @param[output]
    invA      pointer storing the inverse matrix A on the GPU.
    @param[in]
    ldinvA    rocblas_int
              specifies the leading dimension of invA.
    @param[in]
    bsinvA    rocblas_int
             "batch stride invA": stride from the start of one "invA" matrix to the next
    @param[in]
    batch_count       rocblas_int
              numbers of matrices in the batch
    ********************************************************************/

// because of shared memory size, the NB_X must be <= 64
#define NB_X 32

// assume invA has already been allocated, recommened for repeated calling of trtri product routine
template <typename T>
rocblas_status rocblas_trtri_batched_template(rocblas_handle handle,
                                              rocblas_fill uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int n,
                                              const T* A,
                                              rocblas_int lda,
                                              rocblas_int bsa,
                                              T* invA,
                                              rocblas_int ldinvA,
                                              rocblas_int bsinvA,
                                              rocblas_int batch_count)
{
    log_trace(handle,
              replaceX<T>("rocblas_Xtrtri"),
              uplo,
              diag,
              n,
              (const void*&)A,
              lda,
              bsa,
              (const void*&)invA,
              ldinvA,
              bsinvA,
              batch_count);

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if(n < 0)
        return rocblas_status_invalid_size;
    else if(A == nullptr)
        return rocblas_status_invalid_pointer;
    else if(lda < n)
        return rocblas_status_invalid_size;
    else if(bsa < lda * n)
        return rocblas_status_invalid_size;
    else if(invA == nullptr)
        return rocblas_status_invalid_pointer;
    else if(ldinvA < n)
        return rocblas_status_invalid_size;
    else if(bsinvA < ldinvA * n)
        return rocblas_status_invalid_size;
    else if(batch_count < 0)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible.
     */

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(n > NB_X)
    {
        printf("n is %d, n must be less than %d, will return\n", n, NB_X);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1, 1, batch_count);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    hipLaunchKernelGGL((trtri_kernel_batched<T, NB_X>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       bsa,
                       invA,
                       ldinvA,
                       bsinvA);

    return rocblas_status_success;
}

#endif // _TRTRI_BATCHED_HPP_
