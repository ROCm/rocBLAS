/* ************************************************************************
* Copyright 2016-2019 Advanced Micro Devices, Inc.
* ************************************************************************ */

#ifndef __TRTRI_HOST_HPP__
#define __TRTRI_HOST_HPP__

#include "gemm.hpp"
#include "handle.h"
#include "rocblas.h"
#include "trtri_device.hpp"
#include "utility.h"

// return the number of elements in a NxN matrix that do not belong to the triangular region
constexpr size_t num_non_tri_elements(size_t n)
{
    return n * (n - 1) / 2;
}



//////////////////////////////
////////// BATCHED ///////////
//////////////////////////////

template <typename T>
__device__ void rocblas_tritri_batched_fill_upper(size_t      offset,
                                                  size_t      idx,
                                                  rocblas_int n,
                                                  rocblas_int lda,
                                                  rocblas_int sub_stride_A,
                                                  T           value,
                                                  T*          A)
{
    rocblas_int row = n - 2 - floor(sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    rocblas_int col = idx + row + 1 - n * (n - 1) / 2 + (n - row) * (n - row - 1) / 2;

    size_t final_offset = offset * sub_stride_A + (row * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__device__ void rocblas_tritri_batched_fill_lower(
    size_t offset, size_t idx, rocblas_int lda, rocblas_int sub_stride_A, T value, T* A)
{
    rocblas_int row = (rocblas_int)((-1 + sqrt(8 * idx + 1)) / 2);
    rocblas_int col = idx - row * (row + 1) / 2;

    size_t final_offset = offset * sub_stride_A + ((row + 1) * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__global__ void rocblas_trtri_batched_fill(rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           rocblas_long   num_zero_elem,
                                           rocblas_int    lda,
                                           rocblas_int    sub_stride_A,
                                           T*             A[],
                                           rocblas_int    offset_A,
                                           rocblas_int    sub_batch_count)
{
    // number of elements in a given matrix that will be zeroed
    size_t num_elements_total_to_zero = num_zero_elem * sub_batch_count;
    size_t tx                         = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    while(tx < num_elements_total_to_zero)
    {
        // determine which matrix in batch we're working on
        size_t offset = tx / num_zero_elem;
        // determine local matrix index
        size_t idx = tx % num_zero_elem;

        if(uplo == rocblas_fill_upper)
            rocblas_tritri_batched_fill_lower(
                offset, idx, lda, sub_stride_A, T(0), A[hipBlockIdx_y] + offset_A);
        else if(uplo == rocblas_fill_lower)
            rocblas_tritri_batched_fill_upper(
                offset, idx, n, lda, sub_stride_A, T(0), A[hipBlockIdx_y] + offset_A);

        tx += hipBlockDim_x * hipGridDim_x;
    }
}

// flag indicate whether write into A or invA
template <rocblas_int NB, typename T>
__global__ void trtri_small_kernel_batched(rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           const T* const   A[],
                                           rocblas_int      lda,
                                           rocblas_int      sub_stride_A,
                                           T*               invA[],
                                           rocblas_int      ldinvA,
                                           rocblas_int      sub_stride_invA,
                                           rocblas_int      offset_A = 0,
                                           rocblas_int      offset_invA = 0)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A    = A[hipBlockIdx_y] + hipBlockIdx_x * sub_stride_A + offset_A;
    T*       individual_invA = invA[hipBlockIdx_y] + hipBlockIdx_x * sub_stride_invA + offset_invA;

    trtri_device<NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T>
__global__ void trtri_remainder_kernel_batched(rocblas_fill     uplo,
                                               rocblas_diagonal diag,
                                               rocblas_int      n,
                                               const T* const   A[],
                                               rocblas_int      lda,
                                               rocblas_int      sub_stride_A,
                                               T*               invA[],
                                               rocblas_int      ldinvA,
                                               rocblas_int      sub_stride_invA,
                                               rocblas_int      offset_A = 0,
                                               rocblas_int      offset_invA = 0)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix
    const T* individual_A    = A[hipBlockIdx_y] + hipBlockIdx_x * sub_stride_A + offset_A;
    T*       individual_invA = invA[hipBlockIdx_y] + hipBlockIdx_x * sub_stride_invA + offset_invA;

    trtri_device<2 * NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_small_batched(rocblas_handle   handle,
                                           rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           const T* const   A[],
                                           rocblas_int      offset_A,
                                           rocblas_int      lda,
                                           rocblas_int      sub_stride_A,
                                           T*               invA[],
                                           rocblas_int      offset_invA,
                                           rocblas_int      ldinvA,
                                           rocblas_int      sub_stride_invA,
                                           rocblas_int      batch_count,
                                           rocblas_int      sub_batch_count)
{
    if(n > NB)
    {
        printf("n is %d must be less than %d, will exit\n", n, NB);
        return rocblas_status_not_implemented;
    }

    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * sub_batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(rocblas_trtri_batched_fill,
                       dim3(numBlocks, batch_count, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       offset_invA,
                       sub_batch_count);

    dim3 grid(sub_batch_count, batch_count);
    dim3 threads(NB);

    hipLaunchKernelGGL(trtri_small_kernel_batched<NB>,
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       sub_stride_A,
                       invA,
                       ldinvA,
                       sub_stride_invA,
                       offset_A,
                       offset_invA);

    return rocblas_status_success;
}

template <rocblas_int IB, typename T>
__global__ void trtri_diagonal_kernel_batched(rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T* const   A[],
                                              rocblas_int      lda,
                                              rocblas_int      sub_stride_A,
                                              T*               invA[],
                                              rocblas_int      ldinvA,
                                              rocblas_int      sub_stride_invA,
                                              rocblas_int      offset_A = 0,
                                              rocblas_int      offset_invA = 0)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A

    rocblas_int tiles        = n / IB / 2;
    const T*    individual_A = A[hipBlockIdx_y] + (IB * 2 * lda + IB * 2) * (hipBlockIdx_x % tiles)
                            + sub_stride_A * (hipBlockIdx_x / tiles) + offset_A;
    T* individual_invA = invA[hipBlockIdx_y] + (IB * 2 * ldinvA + IB * 2) * (hipBlockIdx_x % tiles)
                         + sub_stride_invA * (hipBlockIdx_x / tiles) + offset_invA;

    custom_trtri_device<IB>(uplo,
                            diag,
                            min(IB, n - (hipBlockIdx_x % tiles) * IB),
                            individual_A,
                            lda,
                            individual_invA,
                            ldinvA);
}

// compute square block of invA
template <typename T>
rocblas_status trtri_batched_gemm_block(rocblas_handle handle,
                                        rocblas_int    M,
                                        rocblas_int    N,
                                        const T*       A[],
                                        rocblas_int    ld_A,
                                        rocblas_int    sub_stride_A,
                                        const T*       invAg1[],
                                        const T*       invAg2a[],
                                        T*             invAg2c[],
                                        rocblas_int    ld_invA,
                                        rocblas_int    sub_stride_invA,
                                        T*             C[],
                                        rocblas_int    ld_C,
                                        rocblas_int    sub_stride_C,
                                        rocblas_int    batch_count,
                                        rocblas_int    sub_blocks,
                                        rocblas_int    offset_A = 0,
                                        rocblas_int    offset_invAg1 = 0,
                                        rocblas_int    offset_invAg2a = 0,
                                        rocblas_int    offset_invAg2c = 0,
                                        rocblas_int    offset_C = 0)
{
    rocblas_status     status;
    static constexpr T one          = 1;
    static constexpr T zero         = 0;
    static constexpr T negative_one = -1;

    // first batched gemm compute C = A21*invA11 (lower) or C = A12*invA22 (upper)
    // distance between each invA11 or invA22 is sub_stride_invA,  sub_stride_A for each A21 or A12, C
    // of size IB * IB

    T* host_A[batch_count];
    T* host_invAg1[batch_count];
    T* host_invAg2a[batch_count];
    T* host_invAg2c[batch_count];
    T* host_C[batch_count];

    hipError_t errA = hipMemcpy(host_A,       A,       batch_count * sizeof(T*), hipMemcpyDeviceToHost);
    hipError_t errB = hipMemcpy(host_invAg1,  invAg1,  batch_count * sizeof(T*), hipMemcpyDeviceToHost);
    hipError_t errC = hipMemcpy(host_invAg2a, invAg2a, batch_count * sizeof(T*), hipMemcpyDeviceToHost);
    hipError_t errD = hipMemcpy(host_invAg2c, invAg2c, batch_count * sizeof(T*), hipMemcpyDeviceToHost);
    hipError_t errE = hipMemcpy(host_C,       C,       batch_count * sizeof(T*), hipMemcpyDeviceToHost);

    if(get_rocblas_status_for_hip_status(errA) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errA);
    else if(get_rocblas_status_for_hip_status(errB) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errB);
    else if(get_rocblas_status_for_hip_status(errC) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errC);
    else if(get_rocblas_status_for_hip_status(errD) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errD);
    else if(get_rocblas_status_for_hip_status(errE) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errE);

    for(int b = 0; b < batch_count; b++)
    {
        status = rocblas_gemm_strided_batched_template(handle,
                                                       rocblas_operation_none,
                                                       rocblas_operation_none,
                                                       M,
                                                       N,
                                                       N,
                                                       &one,
                                                       (const T*)(host_A[b] + offset_A),
                                                       ld_A,
                                                       sub_stride_A,
                                                       (const T*)(host_invAg1[b] + offset_invAg1),
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       &zero,
                                                       (T*)(host_C[b] + offset_C),
                                                       ld_C,
                                                       sub_stride_C,
                                                       sub_blocks);

        status = rocblas_gemm_strided_batched_template(handle,
                                                       rocblas_operation_none,
                                                       rocblas_operation_none,
                                                       M,
                                                       N,
                                                       M,
                                                       &negative_one,
                                                       host_invAg2a[b] + offset_invAg2a,
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       host_C[b] + offset_C,
                                                       ld_C,
                                                       sub_stride_C,
                                                       &zero,
                                                       host_invAg2c[b] + offset_invAg2c,
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       sub_blocks);
    }

    return status;
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_large_batched(rocblas_handle   handle,
                                           rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           const T* const   A[],
                                           rocblas_int      offset_Ain,
                                           rocblas_int      lda,
                                           rocblas_int      sub_stride_Ain,
                                           T*               invA[],
                                           rocblas_int      offset_invAin,
                                           rocblas_int      ldinvA,
                                           rocblas_int      sub_stride_invAin,
                                           rocblas_int      batch_count,
                                           rocblas_int      sub_batch_count,
                                           T*               C_tmp[])
{
    dim3 grid_trtri(n / NB / 2 * sub_batch_count, batch_count);
    dim3 threads(NB * NB);

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA - Only deals with maximum even and complete NBxNB diagonals


    hipLaunchKernelGGL(trtri_diagonal_kernel_batched<NB>,
                       grid_trtri,
                       threads,
                       0,
                       handle->rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       sub_stride_Ain,
                       invA,
                       ldinvA,
                       sub_stride_invAin,
                       offset_Ain,
                       offset_invAin);

    rocblas_int remainder = n - (n / NB / 2) * 2 * NB;
    if(remainder > 0)
    {
        dim3 grid_remainder(sub_batch_count, batch_count);
        dim3 threads_remainder(remainder);

        rocblas_int offset_A2    = (n - remainder) + (n - remainder) * lda + offset_Ain;
        rocblas_int offset_invA2 = (n - remainder) + (n - remainder) * ldinvA + offset_invAin;
        hipLaunchKernelGGL(trtri_remainder_kernel_batched<NB>,
                           grid_remainder,
                           threads_remainder,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           diag,
                           remainder,
                           A,
                           lda,
                           sub_stride_Ain,
                           invA,
                           ldinvA,
                           sub_stride_invAin,
                           offset_A2,
                           offset_invA2);
    }

    if(n <= 2 * NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    size_t sub_block_size       = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * sub_batch_count;
    size_t num_sub_blocks       = (tri_elements_to_zero + sub_block_size - 1) / sub_block_size;
    hipLaunchKernelGGL(rocblas_trtri_batched_fill,
                       dim3(num_sub_blocks, batch_count, 1),
                       dim3(sub_block_size, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       offset_invAin,
                       sub_batch_count);

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    static constexpr auto IB = NB * 2;
    rocblas_int           current_n;

    for(current_n = IB; current_n * 2 <= n; current_n *= 2)
    {
        rocblas_int tiles_per_batch = n / current_n / 2;
        if(tiles_per_batch > sub_batch_count)
        {
            for(int i = 0; i < sub_batch_count; i++)
            {
                rocblas_int offset_A     = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_Ain :
                                                                     current_n * lda + i * sub_stride_Ain);
                rocblas_int offset_invA1 = (uplo == rocblas_fill_lower ? 0 + i * sub_stride_invAin :
                                                                         current_n * ldinvA + current_n + i * sub_stride_invAin);
                rocblas_int offset_invA2 = (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n + i * sub_stride_invAin :
                                                                         0 + i * sub_stride_invAin);
                rocblas_int offset_invA3 = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_invAin :
                                                                         current_n * ldinvA + i * sub_stride_invAin);
                rocblas_int offset_C     = (uplo == rocblas_fill_lower ? (n - current_n) * ldinvA + i * sub_stride_invAin :
                                                                         (n - current_n * tiles_per_batch) + i * sub_stride_invAin);

                offset_A += offset_Ain;
                offset_invA1 += offset_invAin;
                offset_invA2 += offset_invAin;
                offset_invA3 += offset_invAin;
                offset_C += offset_invAin;

                trtri_batched_gemm_block(
                    handle,
                    current_n,
                    current_n,
                    (const T**)A,
                    lda,
                    2 * current_n * lda + 2 * current_n,
                    (const T**)invA,
                    (const T**)invA,
                    (T**)invA,
                    ldinvA,
                    2 * current_n * ldinvA + 2 * current_n,
                    (T**)invA,
                    ldinvA,
                    current_n,
                    batch_count,
                    tiles_per_batch,
                    offset_A,
                    offset_invA1,
                    offset_invA2,
                    offset_invA3,
                    offset_C);
            }
        }
        else
        {
            for(int i = 0; i < tiles_per_batch; i++)
            {
                rocblas_int sub_stride_A2    = (2 * current_n * lda + 2 * current_n);
                rocblas_int sub_stride_invA2 = (2 * current_n * ldinvA + 2 * current_n);

                rocblas_int offset_A     = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_A2
                                                             : current_n * lda + i * sub_stride_A2);

                rocblas_int offset_invA1 = (uplo == rocblas_fill_lower
                                      ? 0 + i * sub_stride_invA2
                                      : current_n * ldinvA + current_n + i * sub_stride_invA2);

                rocblas_int offset_invA2 = (uplo == rocblas_fill_lower
                                      ? current_n * ldinvA + current_n + i * sub_stride_invA2
                                      : 0 + i * sub_stride_invA2);

                rocblas_int offset_invA3 = (uplo == rocblas_fill_lower ? current_n + i * sub_stride_invA2
                                                       : current_n * ldinvA + i * sub_stride_invA2);

                rocblas_int offset_C     = (uplo == rocblas_fill_lower
                                ? (n - current_n) * ldinvA + i * current_n
                                : (n - current_n * tiles_per_batch) + i * current_n);

                offset_A += offset_Ain;
                offset_invA1 += offset_invAin;
                offset_invA2 += offset_invAin;
                offset_invA3 += offset_invAin;
                offset_C += offset_invAin;

                trtri_batched_gemm_block(
                    handle,
                    current_n,
                    current_n,
                    (const T**)A,
                    lda,
                    sub_stride_Ain,
                    (const T**)invA,
                    (const T**)invA,
                    (T**)invA,
                    ldinvA,
                    sub_stride_invAin,
                    (T**)invA,
                    ldinvA,
                    sub_stride_invAin,
                    batch_count,
                    sub_batch_count,
                    offset_A,
                    offset_invA1,
                    offset_invA2,
                    offset_invA3,
                    offset_C);
            }
        }
    }

    hipLaunchKernelGGL(rocblas_trtri_batched_fill,
                       dim3(num_sub_blocks, batch_count, 1),
                       dim3(sub_block_size, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       offset_invAin,
                       sub_batch_count);

    remainder                = (n / NB) * NB - current_n - ((n / NB) % 2 == 0 ? 0 : NB);
    rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16

    if(remainder || oddRemainder)
    {
        if(remainder > 0)
        {
            rocblas_int offset_A     = (uplo == rocblas_fill_lower ? current_n : current_n * lda);
            rocblas_int offset_invA1 = (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n);
            rocblas_int offset_invA2 = (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0);
            rocblas_int offset_invA3 = (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA);

            offset_A += offset_Ain;
            offset_invA1 += offset_invAin;
            offset_invA2 += offset_invAin;
            offset_invA3 += offset_invAin;

            trtri_batched_gemm_block(
                handle,
                uplo == rocblas_fill_lower ? remainder : current_n,
                uplo == rocblas_fill_lower ? current_n : remainder,
                (const T**)A,
                lda,
                sub_stride_Ain,
                (const T**)invA,
                (const T**)invA,
                (T**)invA,
                ldinvA,
                sub_stride_invAin,
                (T**)C_tmp,
                uplo == rocblas_fill_lower ? remainder : current_n,
                remainder * current_n,
                batch_count,
                sub_batch_count,
                offset_A,
                offset_invA1,
                offset_invA2,
                offset_invA3,
                0);
        }

        if(oddRemainder > 0) // solve small oddRemainder
        {
            current_n = n - oddRemainder;
            rocblas_int offset_A     = (uplo == rocblas_fill_lower ? current_n : current_n * lda);
            rocblas_int offset_invA1 = (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n);
            rocblas_int offset_invA2 = (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0);
            rocblas_int offset_invA3 = (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA);

            offset_A += offset_Ain;
            offset_invA1 += offset_invAin;
            offset_invA2 += offset_invAin;
            offset_invA3 += offset_invAin;

            trtri_batched_gemm_block(
                handle,
                uplo == rocblas_fill_lower ? oddRemainder : current_n,
                uplo == rocblas_fill_lower ? current_n : oddRemainder,
                (const T**)A,
                lda,
                sub_stride_Ain,
                (const T**)invA,
                (const T**)invA,
                (T**)invA,
                ldinvA,
                sub_stride_invAin,
                (T**)C_tmp,
                uplo == rocblas_fill_lower ? oddRemainder : current_n,
                oddRemainder * current_n,
                batch_count,
                sub_batch_count,
                offset_A,
                offset_invA1,
                offset_invA2,
                offset_invA3,
                0);
        }
    }

    return rocblas_status_success;
}

template <rocblas_int NB>
constexpr size_t rocblas_trtri_batched_temp_size(rocblas_int n, rocblas_int batch_count)
{
    size_t size = 0;
    if(n > NB * 2 && batch_count > 0)
    {
        rocblas_int current_n = NB * 2;
        while(current_n * 2 <= n)
            current_n *= 2;
        rocblas_int remainder    = (n / NB) * NB - current_n - ((n / NB) % 2 == 0 ? 0 : NB);
        rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16
        if(remainder || oddRemainder)
            size = size_t(remainder ? remainder * current_n : oddRemainder * (n - remainder))
                   * batch_count;
    }
    return size;
}

//////////////////////////////
////// STRIDED BATCHED ///////
//////////////////////////////

template <typename T>
__device__ void rocblas_tritri_strided_batched_fill_upper(size_t      offset,
                                                          size_t      idx,
                                                          rocblas_int n,
                                                          rocblas_int lda,
                                                          rocblas_int sub_stride_A,
                                                          T           value,
                                                          T*          A)
{
    rocblas_int row = n - 2 - floor(sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    rocblas_int col = idx + row + 1 - n * (n - 1) / 2 + (n - row) * (n - row - 1) / 2;

    size_t final_offset = offset * sub_stride_A + (row * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__device__ void rocblas_tritri_strided_batched_fill_lower(
    size_t offset, size_t idx, rocblas_int lda, rocblas_int sub_stride_A, T value, T* A)
{
    rocblas_int row = (rocblas_int)((-1 + sqrt(8 * idx + 1)) / 2);
    rocblas_int col = idx - row * (row + 1) / 2;

    size_t final_offset = offset * sub_stride_A + ((row + 1) * lda) + col;

    A[final_offset] = value;
}

template <typename T>
__global__ void rocblas_trtri_strided_batched_fill(rocblas_handle handle,
                                                   rocblas_fill   uplo,
                                                   rocblas_int    n,
                                                   rocblas_long   num_zero_elem,
                                                   rocblas_int    lda,
                                                   rocblas_int    sub_stride_A,
                                                   T*             A,
                                                   rocblas_int    stride_A,
                                                   rocblas_int    sub_batch_count)
{
    // number of elements in a given matrix that will be zeroed
    size_t num_elements_total_to_zero = num_zero_elem * sub_batch_count;
    size_t tx                         = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    while(tx < num_elements_total_to_zero)
    {
        // determine which matrix in batch we're working on
        size_t offset = tx / num_zero_elem;
        // determine local matrix index
        size_t idx = tx % num_zero_elem;

        if(uplo == rocblas_fill_upper)
            rocblas_tritri_strided_batched_fill_lower(
                offset, idx, lda, sub_stride_A, T(0), A + hipBlockIdx_y * stride_A);
        else if(uplo == rocblas_fill_lower)
            rocblas_tritri_strided_batched_fill_upper(
                offset, idx, n, lda, sub_stride_A, T(0), A + hipBlockIdx_y * stride_A);

        tx += hipBlockDim_x * hipGridDim_x;
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_small_strided_batched(rocblas_handle   handle,
                                                   rocblas_fill     uplo,
                                                   rocblas_diagonal diag,
                                                   rocblas_int      n,
                                                   const T*         A,
                                                   rocblas_int      lda,
                                                   rocblas_int      stride_A,
                                                   rocblas_int      sub_stride_A,
                                                   T*               invA,
                                                   rocblas_int      ldinvA,
                                                   rocblas_int      stride_invA,
                                                   rocblas_int      sub_stride_invA,
                                                   rocblas_int      batch_count,
                                                   rocblas_int      sub_batch_count)
{
    if(n > NB)
    {
        printf("n is %d must be less than %d, will exit\n", n, NB);
        return rocblas_status_not_implemented;
    }

    size_t blockSize            = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * sub_batch_count;
    size_t numBlocks            = (tri_elements_to_zero + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(rocblas_trtri_strided_batched_fill,
                       dim3(numBlocks, batch_count, 1),
                       dim3(blockSize, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       stride_invA,
                       sub_batch_count);

    dim3 grid(sub_batch_count, batch_count);
    dim3 threads(NB);

    hipLaunchKernelGGL(trtri_small_kernel_strided_batched<NB>,
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       stride_A,
                       sub_stride_A,
                       invA,
                       ldinvA,
                       stride_invA,
                       sub_stride_invA);

    return rocblas_status_success;
}

// compute square block of invA
template <typename T>
rocblas_status trtri_strided_batched_gemm_block(rocblas_handle handle,
                                                rocblas_int    M,
                                                rocblas_int    N,
                                                const T*       A,
                                                rocblas_int    ld_A,
                                                rocblas_int    stride_A,
                                                rocblas_int    sub_stride_A,
                                                const T*       invAg1,
                                                const T*       invAg2a,
                                                T*             invAg2c,
                                                rocblas_int    ld_invA,
                                                rocblas_int    stride_invA,
                                                rocblas_int    sub_stride_invA,
                                                T*             C,
                                                rocblas_int    ld_C,
                                                rocblas_int    stride_C,
                                                rocblas_int    sub_stride_C,
                                                rocblas_int    batch_count,
                                                rocblas_int    sub_blocks)
{
    rocblas_status     status;
    static constexpr T one          = 1;
    static constexpr T zero         = 0;
    static constexpr T negative_one = -1;

    // first batched gemm compute C = A21*invA11 (lower) or C = A12*invA22 (upper)
    // distance between each invA11 or invA22 is sub_stride_invA,  sub_stride_A for each A21 or A12, C
    // of size IB * IB
    for(int b = 0; b < batch_count; b++)
    {
        status = rocblas_gemm_strided_batched_template(handle,
                                                       rocblas_operation_none,
                                                       rocblas_operation_none,
                                                       M,
                                                       N,
                                                       N,
                                                       &one,
                                                       (const T*)(A + b * stride_A),
                                                       ld_A,
                                                       sub_stride_A,
                                                       (const T*)(invAg1 + b * stride_invA),
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       &zero,
                                                       (T*)C + b * stride_C,
                                                       ld_C,
                                                       sub_stride_C,
                                                       sub_blocks);

        // second batched gemm compute  invA21 = -invA22 * C (lower) or invA12 = -invA11*C (upper)
        // distance between each invA21 or invA12 is stride_invA,
        status = rocblas_gemm_strided_batched_template(handle,
                                                       rocblas_operation_none,
                                                       rocblas_operation_none,
                                                       M,
                                                       N,
                                                       M,
                                                       &negative_one,
                                                       (const T*)(invAg2a + b * stride_invA),
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       (const T*)C + b * stride_C,
                                                       ld_C,
                                                       sub_stride_C,
                                                       &zero,
                                                       (T*)(invAg2c + b * stride_invA),
                                                       ld_invA,
                                                       sub_stride_invA,
                                                       sub_blocks);
    }

    return status;
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_large_strided_batched(rocblas_handle   handle,
                                                   rocblas_fill     uplo,
                                                   rocblas_diagonal diag,
                                                   rocblas_int      n,
                                                   const T*         A,
                                                   rocblas_int      lda,
                                                   rocblas_int      stride_A,
                                                   rocblas_int      sub_stride_A,
                                                   T*               invA,
                                                   rocblas_int      ldinvA,
                                                   rocblas_int      stride_invA,
                                                   rocblas_int      sub_stride_invA,
                                                   rocblas_int      batch_count,
                                                   rocblas_int      sub_batch_count,
                                                   T*               C_tmp)
{
    dim3 grid_trtri(n / NB / 2 * sub_batch_count, batch_count);
    dim3 threads(NB * NB);

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA - Only deals with maximum even and complete NBxNB diagonals
    hipLaunchKernelGGL(trtri_diagonal_kernel_strided_batched<NB>,
                       grid_trtri,
                       threads,
                       0,
                       handle->rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       stride_A,
                       sub_stride_A,
                       invA,
                       ldinvA,
                       stride_invA,
                       sub_stride_invA);

    rocblas_int remainder = n - (n / NB / 2) * 2 * NB;
    if(remainder > 0)
    {
        dim3 grid_remainder(sub_batch_count, batch_count);
        dim3 threads_remainder(remainder);

        hipLaunchKernelGGL(trtri_remainder_kernel_strided_batched<NB>,
                           grid_remainder,
                           threads_remainder,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           diag,
                           remainder,
                           (const T*)A + (n - remainder) + (n - remainder) * lda,
                           lda,
                           stride_A,
                           sub_stride_A,
                           (T*)invA + (n - remainder) + (n - remainder) * ldinvA,
                           ldinvA,
                           stride_invA,
                           sub_stride_invA);
    }

    if(n <= 2 * NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    size_t sub_block_size       = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * sub_batch_count;
    size_t num_sub_blocks       = (tri_elements_to_zero + sub_block_size - 1) / sub_block_size;
    hipLaunchKernelGGL(rocblas_trtri_strided_batched_fill,
                       dim3(num_sub_blocks, batch_count, 1),
                       dim3(sub_block_size, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       stride_invA,
                       sub_batch_count);

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    static constexpr auto IB = NB * 2;
    rocblas_int           current_n;

    for(current_n = IB; current_n * 2 <= n; current_n *= 2)
    {
        rocblas_int tiles_per_batch = n / current_n / 2;

        if(tiles_per_batch > sub_batch_count)
        {
            for(int i = 0; i < sub_batch_count; i++)
            {
                trtri_strided_batched_gemm_block(
                    handle,
                    current_n,
                    current_n,
                    (const T*)(A
                               + (uplo == rocblas_fill_lower ? current_n + i * sub_stride_A
                                                             : current_n * lda + i * sub_stride_A)),
                    lda,
                    stride_A,
                    2 * current_n * lda + 2 * current_n,
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? 0 + i * sub_stride_invA
                                      : current_n * ldinvA + current_n + i * sub_stride_invA)),
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? current_n * ldinvA + current_n + i * sub_stride_invA
                                      : 0 + i * sub_stride_invA)),
                    (T*)(invA
                         + (uplo == rocblas_fill_lower ? current_n + i * sub_stride_invA
                                                       : current_n * ldinvA + i * sub_stride_invA)),
                    ldinvA,
                    stride_invA,
                    2 * current_n * ldinvA + 2 * current_n,
                    (T*)(invA
                         + (uplo == rocblas_fill_lower
                                ? (n - current_n) * ldinvA + i * sub_stride_invA
                                : (n - current_n * tiles_per_batch) + i * sub_stride_invA)),
                    ldinvA,
                    stride_invA,
                    current_n,
                    batch_count,
                    tiles_per_batch);
            }
        }
        else
        {
            for(int i = 0; i < tiles_per_batch; i++)
            {
                sub_stride_A    = (2 * current_n * lda + 2 * current_n);
                sub_stride_invA = (2 * current_n * ldinvA + 2 * current_n);

                trtri_strided_batched_gemm_block(
                    handle,
                    current_n,
                    current_n,
                    (const T*)(A
                               + (uplo == rocblas_fill_lower ? current_n + i * sub_stride_A
                                                             : current_n * lda + i * sub_stride_A)),
                    lda,
                    stride_A,
                    sub_stride_A,
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? 0 + i * sub_stride_invA
                                      : current_n * ldinvA + current_n + i * sub_stride_invA)),
                    (const T*)(invA
                               + (uplo == rocblas_fill_lower
                                      ? current_n * ldinvA + current_n + i * sub_stride_invA
                                      : 0 + i * sub_stride_invA)),
                    (T*)(invA
                         + (uplo == rocblas_fill_lower ? current_n + i * sub_stride_invA
                                                       : current_n * ldinvA + i * sub_stride_invA)),
                    ldinvA,
                    stride_invA,
                    sub_stride_invA,
                    (T*)(invA
                         + (uplo == rocblas_fill_lower
                                ? (n - current_n) * ldinvA + i * current_n
                                : (n - current_n * tiles_per_batch) + i * current_n)),
                    ldinvA,
                    stride_invA,
                    sub_stride_invA,
                    batch_count,
                    sub_batch_count);
            }
        }
    }

    hipLaunchKernelGGL(rocblas_trtri_strided_batched_fill,
                       dim3(num_sub_blocks, batch_count, 1),
                       dim3(sub_block_size, 1, 1),
                       0,
                       handle->rocblas_stream,
                       handle,
                       (uplo == rocblas_fill_lower) ? rocblas_fill_upper : rocblas_fill_lower,
                       n,
                       num_non_tri_elements(n),
                       ldinvA,
                       n * ldinvA,
                       invA,
                       stride_invA,
                       sub_batch_count);

    remainder                = (n / NB) * NB - current_n - ((n / NB) % 2 == 0 ? 0 : NB);
    rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16

    if(remainder || oddRemainder)
    {
        if(remainder > 0)
        {
            trtri_strided_batched_gemm_block(
                handle,
                uplo == rocblas_fill_lower ? remainder : current_n,
                uplo == rocblas_fill_lower ? current_n : remainder,
                (const T*)(A + (uplo == rocblas_fill_lower ? current_n : current_n * lda)),
                lda,
                stride_A,
                sub_stride_A,
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n)),
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0)),
                (T*)(invA + (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA)),
                ldinvA,
                stride_invA,
                sub_stride_invA,
                C_tmp,
                uplo == rocblas_fill_lower ? remainder : current_n,
                0,
                remainder * current_n,
                batch_count,
                sub_batch_count);
        }

        if(oddRemainder > 0) // solve small oddRemainder
        {
            current_n = n - oddRemainder;
            trtri_strided_batched_gemm_block(
                handle,
                uplo == rocblas_fill_lower ? oddRemainder : current_n,
                uplo == rocblas_fill_lower ? current_n : oddRemainder,
                (const T*)(A + (uplo == rocblas_fill_lower ? current_n : current_n * lda)),
                lda,
                stride_A,
                sub_stride_A,
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? 0 : current_n * ldinvA + current_n)),
                (const T*)(invA
                           + (uplo == rocblas_fill_lower ? current_n * ldinvA + current_n : 0)),
                (T*)(invA + (uplo == rocblas_fill_lower ? current_n : current_n * ldinvA)),
                ldinvA,
                stride_invA,
                sub_stride_invA,
                C_tmp,
                uplo == rocblas_fill_lower ? oddRemainder : current_n,
                0,
                oddRemainder * current_n,
                batch_count,
                sub_batch_count);
        }
    }

    return rocblas_status_success;
}

template <rocblas_int NB>
constexpr size_t rocblas_trtri_strided_batched_temp_size(rocblas_int n, rocblas_int batch_count)
{
    size_t size = 0;
    if(n > NB * 2 && batch_count > 0)
    {
        rocblas_int current_n = NB * 2;
        while(current_n * 2 <= n)
            current_n *= 2;
        rocblas_int remainder    = (n / NB) * NB - current_n - ((n / NB) % 2 == 0 ? 0 : NB);
        rocblas_int oddRemainder = n - current_n - remainder; // should always be NB - 16
        if(remainder || oddRemainder)
            size = size_t(remainder ? remainder * current_n : oddRemainder * (n - remainder))
                   * batch_count;
    }
    return size;
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_batched_template(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T* const   A[],
                                              rocblas_int      offset_A,
                                              rocblas_int      lda,
                                              rocblas_int      sub_stride_A,
                                              T*               invA[],
                                              rocblas_int      offset_invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      sub_stride_invA,
                                              rocblas_int      batch_count,
                                              rocblas_int      sub_batch_count,
                                              T*               C_tmp[])
{
    if(!n || !sub_batch_count)
        return rocblas_status_success;
    if(n <= NB)
    {
        return rocblas_trtri_small_batched<NB>(handle,
                                               uplo,
                                               diag,
                                               n,
                                               A,
                                               offset_A,
                                               lda,
                                               sub_stride_A,
                                               invA,
                                               offset_invA,
                                               ldinvA,
                                               sub_stride_invA,
                                               batch_count,
                                               sub_batch_count);
    }
    else
    {
        return rocblas_trtri_large_batched<NB>(handle,
                                               uplo,
                                               diag,
                                               n,
                                               A,
                                               offset_A,
                                               lda,
                                               sub_stride_A,
                                               invA,
                                               offset_invA,
                                               ldinvA,
                                               sub_stride_invA,
                                               batch_count,
                                               sub_batch_count,
                                               C_tmp);
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_strided_batched_template(rocblas_handle   handle,
                                                      rocblas_fill     uplo,
                                                      rocblas_diagonal diag,
                                                      rocblas_int      n,
                                                      const T*         A,
                                                      rocblas_int      lda,
                                                      rocblas_int      stride_A,
                                                      rocblas_int      sub_stride_A,
                                                      T*               invA,
                                                      rocblas_int      ldinvA,
                                                      rocblas_int      stride_invA,
                                                      rocblas_int      sub_stride_invA,
                                                      rocblas_int      batch_count,
                                                      rocblas_int      sub_batch_count,
                                                      T*               C_tmp)
{
    if(!n || !sub_batch_count)
        return rocblas_status_success;
    if(n <= NB)
    {
        return rocblas_trtri_small_strided_batched<NB>(handle,
                                                       uplo,
                                                       diag,
                                                       n,
                                                       A,
                                                       lda,
                                                       stride_A,
                                                       sub_stride_A,
                                                       invA,
                                                       ldinvA,
                                                       stride_invA,
                                                       sub_stride_invA,
                                                       batch_count,
                                                       sub_batch_count);
    }
    else
    {
        return rocblas_trtri_large_strided_batched<NB>(handle,
                                                       uplo,
                                                       diag,
                                                       n,
                                                       A,
                                                       lda,
                                                       stride_A,
                                                       sub_stride_A,
                                                       invA,
                                                       ldinvA,
                                                       stride_invA,
                                                       sub_stride_invA,
                                                       batch_count,
                                                       sub_batch_count,
                                                       C_tmp);
    }
}

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_template(rocblas_handle   handle,
                                      rocblas_fill     uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int      n,
                                      const T*         A,
                                      rocblas_int      lda,
                                      T*               invA,
                                      rocblas_int      ldinvA,
                                      T*               C_tmp)
{
    return rocblas_trtri_strided_batched_template<NB>(
        handle, uplo, diag, n, A, lda, lda * n, 0, invA, ldinvA, ldinvA * n, 0, 1, 1, C_tmp);
}

#endif // \include guard