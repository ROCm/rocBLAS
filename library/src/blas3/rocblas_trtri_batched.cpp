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
