/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"
#include "trtri_device.h"
// #include "trtri.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

// because of shared memory size, the NB must be <= 64
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

template <typename T>
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

    // each hip thread Block compute a inverse of a NB * NB diagonal block of A
    // notice the last digaonal block may be smaller than NB*NB

    rocblas_int tiles = (n+NB-1)/NB;
    const T* individual_A =
        A + NB * lda * (hipBlockIdx_x % tiles) + NB * (hipBlockIdx_x % tiles) + bsa * (hipBlockIdx_x / tiles);
    T* individual_invA = invA + NB * ldinvA * (hipBlockIdx_x % tiles) + NB * (hipBlockIdx_x % tiles) +
                         bsinvA * (hipBlockIdx_x / tiles);

    trtri_device<T, NB>(
        uplo, diag, min(NB, n - (hipBlockIdx_x % tiles)  * NB), individual_A, lda, individual_invA, ldinvA);
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

    // if(n > 2 * NB) //remove for now
    // {
    //     printf("n is %d, n must be less than %d, will return\n", n, 2 * NB);
    //     return rocblas_status_not_implemented;
    // }

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    dim3 grid_trtri((n+NB-1)/NB * batch_count);
    dim3 threads(NB);

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA
    hipLaunchKernelGGL((trtri_diagonal_kernel_batched<T>),
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

    if(n <= NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    dim3 grid_gemm((n+NB*2-1)/(NB*2) * batch_count);
    rocblas_int blocks = n / NB; // complete blocks - need to do all these together and then deal with partial blocks

    rocblas_int m_gemm;
    rocblas_int n_gemm;
    T* A_gemm;
    const T* B_gemm;
    T* C_gemm;
    T* D_gemm;

    if(uplo == rocblas_fill_lower)
    {
        // perform D = -A*B*C  ==>  invA21 = -invA22*A21*invA11,
        m_gemm = (n - NB);
        n_gemm = NB;
        A_gemm = invA + NB + NB * ldinvA; // invA22
        B_gemm = A + NB;                  // A21
        C_gemm = invA;                    // invA11
        D_gemm = invA + NB;               // invA21
    }
    else
    {
        // perform D = -A*B*C  ==>  invA12 = -invA11*A12*invA22,
        m_gemm = NB;
        n_gemm = (n - NB);
        A_gemm = invA;                    // invA11
        B_gemm = A + lda * NB;            // A12
        C_gemm = invA + NB + NB * ldinvA; // invA22
        D_gemm = invA + NB * ldinvA;      // invA12
    }

    hipLaunchKernelGGL((gemm_trsm_batched<T>),
                       grid_gemm,
                       threads,
                       0,
                       rocblas_stream,
                       uplo,
                       m_gemm,
                       n_gemm,
                       A_gemm,
                       ldinvA,
                       B_gemm,
                       lda,
                       C_gemm,
                       ldinvA,
                       D_gemm,
                       ldinvA,
                       bsa,
                       bsinvA);

    return rocblas_status_success;
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
              handle to the rocblas lNBrary context queue.
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
     * Quick return if possNBle.
     */

    if(!n || !batch_count)
        return rocblas_status_success;

    if(n <= NB)
    {
        return rocblas_trtri_small_batched<T>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }
    else if(n <= 2 * NB)
    {
        return rocblas_trtri_large_batched<T>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }
    else
    {
        printf("n is %d, n must be less than %d, will return\n", n, 2 * NB);
        return rocblas_status_not_implemented;
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
