/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"
#include "trtri_device.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

// because of shared memory size, the NB must be <= 64
constexpr int NB = 32;

// flag indicate whether write into A or invA
template <typename T>
__global__ void trtri_kernel_batched(rocblas_fill uplo,
                                     rocblas_diagonal diag,
                                     rocblas_int n,
                                     const T* A,
                                     rocblas_int lda,
                                     rocblas_int bsa,
                                     T* invA,
                                     rocblas_int ldinvA,
                                     rocblas_int bsinvA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A = A + hipBlockIdx_z * bsa;
    T* individual_invA    = invA + hipBlockIdx_z * bsinvA;

    trtri_device<T, NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <typename>
constexpr char rocblas_trtri_name[] = "unknown";
template <>
constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri";
template <>
constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri";

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
    if(!handle)
        return rocblas_status_invalid_handle;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle,
                  rocblas_trtri_name<T>,
                  uplo,
                  diag,
                  n,
                  A,
                  lda,
                  bsa,
                  invA,
                  ldinvA,
                  bsinvA,
                  batch_count);

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle,
                    rocblas_trtri_name<T>,
                    "uplo",
                    rocblas_fill_letter(uplo),
                    "diag",
                    rocblas_diag_letter(diag),
                    "N",
                    n,
                    "lda",
                    lda,
                    "bsa",
                    bsa,
                    "ldinvA",
                    ldinvA,
                    "bsinvA",
                    bsinvA,
                    "batch_count",
                    batch_count);

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    if(n < 0)
        return rocblas_status_invalid_size;
    if(!A)
        return rocblas_status_invalid_pointer;
    if(lda < n || bsa < lda * n)
        return rocblas_status_invalid_size;
    if(!invA)
        return rocblas_status_invalid_pointer;
    if(ldinvA < n || bsinvA < ldinvA * n || batch_count < 0)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible.
     */

    if(!n || !batch_count)
        return rocblas_status_success;

    if(n > NB)
    {
        printf("n is %d, n must be less than %d, will return\n", n, NB);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1, 1, batch_count);
    dim3 threads(NB);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    hipLaunchKernelGGL(trtri_kernel_batched,
                       grid,
                       threads,
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

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C interface
 *    This function is called by trsm
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strtri_batched(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const float* A,
                                      rocblas_int lda,
                                      rocblas_int bsa,
                                      float* invA,
                                      rocblas_int ldinvA,
                                      rocblas_int bsinvA,
                                      rocblas_int batch_count)
{
    return rocblas_trtri_batched_template(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

rocblas_status rocblas_dtrtri_batched(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const double* A,
                                      rocblas_int lda,
                                      rocblas_int bsa,
                                      double* invA,
                                      rocblas_int ldinvA,
                                      rocblas_int bsinvA,
                                      rocblas_int batch_count)
{
    return rocblas_trtri_batched_template(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

} // extern "C"
