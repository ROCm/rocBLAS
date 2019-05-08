/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"
#include "trtri_device.h"
#include "trtri_trsm.hpp"
#include "rocblas_trtri_batched.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace trtri {

constexpr int NB = 16;

// flag indicate whether write into A or invA
template <typename T>
__global__ void trtri_small_kernel_batched(rocblas_fill uplo,
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
    const T* individual_A = A + hipBlockIdx_x * bsa;
    T* individual_invA    = invA + hipBlockIdx_x * bsinvA;

    trtri_device<T, NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <typename T>
__global__ void trtri_remainder_kernel_batched(rocblas_fill uplo,
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
    const T* individual_A = A + hipBlockIdx_x * bsa;
    T* individual_invA    = invA + hipBlockIdx_x * bsinvA;

    trtri_device<T, 2 * NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <typename T>
rocblas_status rocblas_trtri_small_batched(rocblas_handle handle,
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

    if(n > NB)
    {
        printf("n is %d must be less than %d, will exit\n", n, NB);
        return rocblas_status_not_implemented;
    }

    hipStream_t rocblas_stream  = handle->rocblas_stream;
    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(rocblas_trtri_batched_fill<T>,
                       dim3(numBlocks, 1, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       rocblas_stream,
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       batch_count);

    dim3 grid(batch_count);
    dim3 threads(NB);

    hipLaunchKernelGGL(trtri_small_kernel_batched,
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

template <typename T, rocblas_int IB>
__global__ void trtri_diagonal_kernel_batched(rocblas_fill uplo,
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

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A

    rocblas_int tiles = n / IB / 2;
    const T* individual_A =
        A + (IB * 2 * lda + IB * 2) * (hipBlockIdx_x % tiles) + bsa * (hipBlockIdx_x / tiles);
    T* individual_invA = invA + (IB * 2 * ldinvA + IB * 2) * (hipBlockIdx_x % tiles) +
                         bsinvA * (hipBlockIdx_x / tiles);

    custom_trtri_device<T, IB>(uplo,
                               diag,
                               min(IB, n - (hipBlockIdx_x % tiles) * IB),
                               individual_A,
                               lda,
                               individual_invA,
                               ldinvA);
}

template <typename T>
rocblas_status rocblas_trtri_large_batched(rocblas_handle handle,
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
    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    dim3 grid_trtri(n / NB / 2 * batch_count);
    dim3 threads(NB * NB);

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA - Only deals with maximum even and complete NBxNB diagonals
    hipLaunchKernelGGL((trtri_diagonal_kernel_batched<T, NB>),
                       grid_trtri,
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

    rocblas_int remainder = n - (n / NB / 2) * 2 * NB;
    if(remainder > 0)
    {
        dim3 grid_remainder(batch_count);
        dim3 threads_remainder(remainder);

        hipLaunchKernelGGL(trtri_remainder_kernel_batched,
                           grid_remainder,
                           threads_remainder,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           remainder,
                           (const T*)A + (n - remainder) + (n - remainder) * lda,
                           lda,
                           bsa,
                           (T*)invA + (n - remainder) + (n - remainder) * ldinvA,
                           ldinvA,
                           bsinvA);
    }

    if(n <= 2 * NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(rocblas_trtri_batched_fill<T>,
                       dim3(numBlocks, 1, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       rocblas_stream,
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       batch_count);

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)

    constexpr rocblas_int IB = NB * 2;
    rocblas_int current_n;

    for(current_n = IB; current_n * 2 <= n; current_n *= 2)
    {
        rocblas_int tiles_per_batch = n / current_n / 2;

        if(tiles_per_batch > batch_count)
        {
            for(int i = 0; i < batch_count; i++)
            {
                trtri_strided_gemm_block<T>(
                    handle,
                    current_n,
                    current_n,
                    (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n + i * bsa
                                                                 : current_n * lda + i * bsa)),
                    lda,
                    2 * current_n * lda + 2 * current_n,
                    (const T*)(invA + ((uplo == rocblas_fill_lower)
                                           ? 0 + i * bsinvA
                                           : current_n * ldinvA + current_n + i * bsinvA)),
                    (const T*)(invA + ((uplo == rocblas_fill_lower)
                                           ? current_n * ldinvA + current_n + i * bsinvA
                                           : 0 + i * bsinvA)),
                    (T*)(invA + ((uplo == rocblas_fill_lower) ? current_n + i * bsinvA
                                                              : current_n * ldinvA + i * bsinvA)),
                    ldinvA,
                    2 * current_n * ldinvA + 2 * current_n,
                    (T*)(invA + ((uplo == rocblas_fill_lower)
                                     ? (n - current_n) * ldinvA + i * bsinvA
                                     : (n - current_n * tiles_per_batch) + i * bsinvA)),
                    ldinvA,
                    current_n,
                    tiles_per_batch);
            }
        }
        else
        {
            for(int i = 0; i < tiles_per_batch; i++)
            {
                rocblas_int stride_A, stride_invA;
                stride_A    = (2 * current_n * lda + 2 * current_n);
                stride_invA = (2 * current_n * ldinvA + 2 * current_n);
                trtri_strided_gemm_block<T>(
                    handle,
                    current_n,
                    current_n,
                    (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n + i * stride_A
                                                                 : current_n * lda + i * stride_A)),
                    lda,
                    bsa,
                    (const T*)(invA + ((uplo == rocblas_fill_lower)
                                           ? 0 + i * stride_invA
                                           : current_n * ldinvA + current_n + i * stride_invA)),
                    (const T*)(invA + ((uplo == rocblas_fill_lower)
                                           ? current_n * ldinvA + current_n + i * stride_invA
                                           : 0 + i * stride_invA)),
                    (T*)(invA + ((uplo == rocblas_fill_lower)
                                     ? current_n + i * stride_invA
                                     : current_n * ldinvA + i * stride_invA)),
                    ldinvA,
                    bsinvA,
                    (T*)(invA + ((uplo == rocblas_fill_lower)
                                     ? (n - current_n) * ldinvA + i * current_n
                                     : (n - current_n * tiles_per_batch) + i * current_n)),
                    ldinvA,
                    bsinvA,
                    batch_count);
            }
        }
    }

    hipLaunchKernelGGL(rocblas_trtri_batched_fill<T>,
                       dim3(numBlocks, 1, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       rocblas_stream,
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       batch_count);

    remainder                = n - current_n - ((n / NB) % 2 == 0 ? 0 : NB) - (n - (n / NB) * NB);
    rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16

    if(remainder || oddRemainder)
    {
        auto C_tmp = rocblas_unique_ptr{
            rocblas::device_malloc(
                sizeof(T) * batch_count *
                (remainder ? (remainder * current_n) : (oddRemainder * (n - remainder)))),
            rocblas::device_free};

        if(remainder > 0)
        {
            trtri_strided_gemm_block<T>(
                handle,
                (uplo == rocblas_fill_lower) ? remainder : current_n,
                (uplo == rocblas_fill_lower) ? current_n : remainder,
                (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n : current_n * lda)),
                lda,
                bsa,
                (const T*)(invA +
                           ((uplo == rocblas_fill_lower) ? 0 : current_n * ldinvA + current_n)),
                (const T*)(invA +
                           ((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n : 0)),
                (T*)(invA + ((uplo == rocblas_fill_lower) ? current_n : current_n * ldinvA)),
                ldinvA,
                bsinvA,
                (T*)(C_tmp.get()),
                (uplo == rocblas_fill_lower) ? remainder : current_n,
                remainder * current_n,
                batch_count);
        }

        if(oddRemainder > 0) // solve small oddRemainder
        {
            current_n = n - oddRemainder;

            trtri_strided_gemm_block<T>(
                handle,
                (uplo == rocblas_fill_lower) ? oddRemainder : current_n,
                (uplo == rocblas_fill_lower) ? current_n : oddRemainder,
                (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n : current_n * lda)),
                lda,
                bsa,
                (const T*)(invA +
                           ((uplo == rocblas_fill_lower) ? 0 : current_n * ldinvA + current_n)),
                (const T*)(invA +
                           ((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n : 0)),
                (T*)(invA + ((uplo == rocblas_fill_lower) ? current_n : current_n * ldinvA)),
                ldinvA,
                bsinvA,
                (T*)(C_tmp.get()),
                (uplo == rocblas_fill_lower) ? oddRemainder : current_n,
                oddRemainder * current_n,
                batch_count);
        }
    }

    return rocblas_status_success;
}

template <typename>
static constexpr char rocblas_trtri_name[] = "unknown";
template <>
static constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri";
template <>
static constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri";

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
     * Quick return if possNBle.
     */

    if(!n || !batch_count)
        return rocblas_status_success;

    if(n <= NB)
    {
        return rocblas_trtri_small_batched<T>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }
    else
    {
        return rocblas_trtri_large_batched<T>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }
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
    return trtri::rocblas_trtri_batched_template(
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
    return trtri::rocblas_trtri_batched_template(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

} // extern "C"
