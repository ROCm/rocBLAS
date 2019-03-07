/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"
#include "trtri_device.h"
#include "trtri_trsm.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

constexpr int NB = 16;

template <typename T>
__device__ void rocblas_tritri_batched_fill_upper(
    size_t offset, size_t idx, rocblas_int n, rocblas_int lda, rocblas_int bsa, T value, T* A)
{
    rocblas_int row = n - 2 - floor(sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    rocblas_int col = idx + row + 1 - n * (n - 1) / 2 + (n - row) * (n - row - 1) / 2;

    size_t final_offset = offset * bsa + (row * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__device__ void rocblas_tritri_batched_fill_lower(
    size_t offset, size_t idx, rocblas_int lda, rocblas_int bsa, T value, T* A)
{
    rocblas_int row = (rocblas_int)((-1 + sqrt(8 * idx + 1)) / 2);
    rocblas_int col = idx - row * (row + 1) / 2;

    size_t final_offset = offset * bsa + ((row + 1) * lda) + col;

    A[final_offset] = value;
}

// return the number of elements in a NxN matrix that do not belong to the triangular region
inline size_t num_non_tri_elements(rocblas_int n) { return (n * (n - 1) / 2); }

template <typename T>
__global__ void rocblas_tritri_batched_fill(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_long num_zero_elem,
                                            rocblas_int lda,
                                            rocblas_int bsa,
                                            T* A,
                                            rocblas_int batch_count)
{
    // if(!handle)
    //     return rocblas_status_invalid_handle;

    // number of elements in a given matrix that will be zeroed
    size_t num_elements_total_to_zero = num_zero_elem * batch_count;
    size_t tx                         = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    while(tx < num_elements_total_to_zero)
    {
        // determine which matrix in batch we're working on
        size_t offset = tx / num_zero_elem;
        // determine local matrix index
        size_t idx = tx % num_zero_elem;

        if(uplo == rocblas_fill_upper)
        {
            rocblas_tritri_batched_fill_lower<T>(offset, idx, lda, bsa, 0, A);
        }
        else if(uplo == rocblas_fill_lower)
        {
            rocblas_tritri_batched_fill_upper<T>(offset, idx, n, lda, bsa, 0, A);
        }
        tx += hipBlockDim_x * hipGridDim_x;
    }
}

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

    dim3 grid(batch_count);
    dim3 threads(NB);
    hipStream_t rocblas_stream = handle->rocblas_stream;

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
__device__ void gemm_trsm_kernel(rocblas_fill uplo,
                                 rocblas_int m,
                                 rocblas_int n,
                                 const T* A,
                                 rocblas_int lda,
                                 const T* B,
                                 rocblas_int ldb,
                                 const T* C,
                                 rocblas_int ldc,
                                 T* D,
                                 rocblas_int ldd)
{
    __shared__ T shared_tep[NB * NB];
    __shared__ T vec[NB];
    T reg[NB];

    rocblas_int tx = hipThreadIdx_x;

    // read B into registers, B is of m * n
    if(tx < m)
    {
        for(int col = 0; col < n; col++)
        {
            reg[col] = B[tx + col * ldb];
        }
    }

    // shared_tep = B * C; shared_tep is of m * n, C is of n * n
    for(int col = 0; col < n; col++)
    {
        // load C's column in vec
        if(tx < n)
            vec[tx] = C[col * ldc + tx];
        __syncthreads();

        T reg_tep = 0;
        // perform reduction
        if(uplo == rocblas_fill_lower)
        {
            for(int i = col; i < n; i++)
            {
                reg_tep += reg[i] * vec[i];
            }
        }
        else
        {
            for(int i = 0; i < col + 1; i++)
            {
                reg_tep += reg[i] * vec[i];
            }
        }

        if(tx < m)
        {
            shared_tep[tx + col * NB] = reg_tep;
        }
    }

    __syncthreads();

    // read A into registers A is of m * m
    if(tx < m)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int col = 0; col < tx + 1; col++)
            {
                reg[col] = A[tx + col * lda];
            }
        }
        else
        {
            for(int col = tx; col < m; col++)
            {
                reg[col] = A[tx + col * lda];
            }
        }
    }

    // D = A * shared_tep; shared_tep is of m * n
    for(int col = 0; col < n; col++)
    {

        T reg_tep = 0;
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i < tx + 1; i++)
            {
                reg_tep += reg[i] * shared_tep[i + col * NB];
            }
        }
        else
        {
            for(int i = tx; i < m; i++)
            {
                reg_tep += reg[i] * shared_tep[i + col * NB];
            }
        }

        if(tx < m)
        {
            D[tx + col * ldd] = (-1) * reg_tep;
        }
    }
}

template <typename T>
__global__ void gemm_trsm_batched(rocblas_fill uplo,
                                  rocblas_int m,
                                  rocblas_int n,
                                  const T* A,
                                  rocblas_int lda,
                                  const T* B,
                                  rocblas_int ldb,
                                  const T* C,
                                  rocblas_int ldc,
                                  T* D,
                                  rocblas_int ldd,
                                  rocblas_int bsa,
                                  rocblas_int bsinvA)
{

    gemm_trsm_kernel<T>(uplo,
                        m,
                        n,
                        A + bsinvA * hipBlockIdx_x,
                        lda,
                        B + bsa * hipBlockIdx_x,
                        ldb,
                        C + bsinvA * hipBlockIdx_x,
                        ldc,
                        D + bsinvA * hipBlockIdx_x,
                        ldd);
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

    if(n > 2 * NB && (n & (n - 1)) != 0)
    {
        printf("n is %d, sizes bigger than %d must be a power of 2, will return\n", n, 2 * NB);
        return rocblas_status_not_implemented;
    }

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

    // // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    // dim3 grid_gemm((n+NB*2-1)/(NB*2) * batch_count);
    constexpr rocblas_int IB = NB * 2;
    rocblas_int blocks =
        n / IB; // complete blocks - need to do all these together and then deal with partial blocks
    rocblas_int current_n;

    for(rocblas_int current_n = IB; current_n * 2 <= n; current_n *= 2)
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

    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(rocblas_tritri_batched_fill<T>,
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

    return rocblas_status_success;
}

template <typename>
constexpr char rocblas_trtri_name[] = "unknown";
template <>
constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri";
template <>
constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri";

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
