/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "definitions.h"
#include "trtri_device.h"
#include "trtri_trsm.hpp"
// #include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"
#include "rocblas_unique_ptr.hpp" // temp

namespace {

// because of shared memory size, the NB must be <= 64
constexpr int NB = 16;

// /*
// example of how to call:
// size_t blockSize = 128;
// size_t tri_elements_to_zero = num_non_tri_elements(n) * batches;
// size_t numBlocks = (tri_elements_to_zero + blockSize - 1) / blockSize;
// cout << "Executing kernel with n=" << n << ". numBlocks="  << numBlocks << ".";
// hipLaunchKernelGGL(rocblas_tritri_batched_fill<float>, dim3(numBlocks,1,1), dim3(blockSize,1,1), 0, 0,rocblas_fill_upper, i, num_non_tri_elements(i), lda, n*lda, mD, batches);
// */

template <typename T>
__device__ void rocblas_tritri_batched_fill_upper(size_t offset,
                                                            size_t idx,
                                                            rocblas_int n,
                                                            rocblas_int lda,
                                                            rocblas_int bsa,
                                                            T value,
                                                            T* A)
{
    rocblas_int row = n - 2 - floor(sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    rocblas_int col = idx + row + 1 - n * (n - 1) / 2 + (n - row) * (n - row - 1) / 2;

    size_t final_offset = offset * bsa + (row * lda) + col;

    A[final_offset] = value;

}

template <typename T>
__device__ void rocblas_tritri_batched_fill_lower(size_t offset,
                                                            size_t idx,
                                                            rocblas_int lda,
                                                            rocblas_int bsa,
                                                            T value,
                                                            T* A)
{
    rocblas_int row = (rocblas_int)((-1 + sqrt(8 * idx + 1)) / 2);
    rocblas_int col = idx - row * (row + 1) / 2;

    size_t final_offset = offset * bsa + ((row + 1) * lda) + col;

    A[final_offset] = value;

}

// return the number of elements in a NxN matrix that do not belong to the triangular region
inline size_t num_non_tri_elements(rocblas_int n)
{
    return (n * (n - 1) / 2);
}

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
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    while (tx < num_elements_total_to_zero)
    {
        // determine which matrix in batch we're working on
        size_t offset = tx / num_zero_elem;
        // determine local matrix index
        size_t idx = tx % num_zero_elem;

        if (uplo == rocblas_fill_upper)
        {
            rocblas_tritri_batched_fill_lower<T>(offset, idx, lda, bsa, 0, A);
        }
        else if (uplo == rocblas_fill_lower)
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

    trtri_device<T, 2*NB>(uplo, diag, n, individual_A, lda, individual_invA, ldinvA);
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

    rocblas_int tiles = n/NB/2;///(n+NB-1)/NB/2;
    const T* individual_A =
        A + NB*2 * lda * (hipBlockIdx_x % tiles) + NB*2 * (hipBlockIdx_x % tiles) + bsa * (hipBlockIdx_x / tiles);
    T* individual_invA = invA + NB*2 * ldinvA * (hipBlockIdx_x % tiles) + NB*2 * (hipBlockIdx_x % tiles) +
                         bsinvA * (hipBlockIdx_x / tiles);

    custom_trtri_device<T, NB>(
        uplo, diag, min(NB, n - (hipBlockIdx_x % tiles)  * NB), individual_A, lda, individual_invA, ldinvA); //TODO check min stuff
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
rocblas_status trtri_strided_gemm_block_t(rocblas_handle handle,
                                        rocblas_int M,
                                        const T* A,
                                        rocblas_int ld_A,
                                        rocblas_int stride_A,
                                        const T* invAg1,
                                        const T* invAg2a,
                                        T* invAg2c,
                                        rocblas_int ld_invA,
                                        rocblas_int stride_invA,
                                        T* C,
                                        rocblas_int ld_C,
                                        rocblas_int stride_C,
                                        rocblas_int batch)
{
    rocblas_status status;

    T one          = 1;
    T zero         = 0;
    T negative_one = -1;

#ifndef NDEBUG
    printf("first batched gemm\n");
#endif
    // first batched gemm compute C = A21*invA11 (lower) or C = A12*invA22 (upper)
    // distance between each invA11 or invA22 is stride_invA,  stride_A for each A21 or A12, C
    // of size IB * IB
    status = rocblas_gemm_strided_batched_template<T>(handle,
                                                      rocblas_operation_none,
                                                      rocblas_operation_none,
                                                      M,
                                                      M,
                                                      M,
                                                      &one,
                                                      (const T*)A,
                                                      ld_A,
                                                      stride_A,
                                                      (const T*)invAg1,
                                                      ld_invA,
                                                      stride_invA,
                                                      &zero,
                                                      (T*)C,
                                                      ld_C,
                                                      stride_C,
                                                      batch);

#ifndef NDEBUG
    printf("second batched gemm\n");
#endif
    // second batched gemm compute  invA21 = -invA22 * C (lower) or invA12 = -invA11*C (upper)
    // distance between each invA21 or invA12 is stride_invA,
    status = rocblas_gemm_strided_batched_template<T>(handle,
                                                      rocblas_operation_none,
                                                      rocblas_operation_none,
                                                      M,
                                                      M,
                                                      M,
                                                      &negative_one,
                                                      (const T*)invAg2a,
                                                      ld_invA,
                                                      stride_invA,
                                                      (const T*)C,
                                                      ld_C,
                                                      stride_C,
                                                      &zero,
                                                      (T*)invAg2c,
                                                      ld_invA,
                                                      stride_invA,
                                                      batch);

    return status;
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

    dim3 grid_trtri(n/NB/2 * batch_count);
    dim3 threads(NB*NB);

    std::cout<<"grid "<<n/NB/2 * batch_count<<" threads "<<NB*NB<<std::endl;

    // first stage: invert NB * NB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA - Only deals with maximum even and complete NBxNB diagonals
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

    rocblas_int tiles = n/NB/2;


    for(int i =0; i<(n/NB/2 * batch_count); i++)
        std::cout<<" A index "<<(NB*2 * lda * (i % tiles) + NB*2 * (i % tiles) + bsa * (i / tiles))<<" invA index "<<
        (NB*2 * ldinvA * (i % tiles) + NB*2 * (i % tiles) +
                         bsinvA * (i / tiles))<<" tiles "<<tiles<<" min n "<<min(NB, n - (i % tiles)  * NB)<<std::endl;
        
    rocblas_int r = n-(n/NB/2)*2*NB; // first divide should round down
    if(r>0)
    {
        std::cout<<"diag r "<<r<<std::endl;
        std::cout<<"A index  "<<(n/NB/2)*NB*2+(n/NB/2)*NB*2*lda<<" invA index "<<(n/NB/2)*NB*2+(n/NB/2)*NB*2*ldinvA<<std::endl;
        dim3 grid_remainder(batch_count);
        dim3 threads_remainder(r);

        hipLaunchKernelGGL(trtri_remainder_kernel_batched,
                        grid_remainder,
                        threads_remainder,
                        0,
                        rocblas_stream,
                        uplo,
                        diag,
                        r,
                        (const T*) A+(n/NB/2)*NB*2+(n/NB/2)*NB*2*lda,
                        lda,
                        bsa,
                        (T*) invA + (n/NB/2)*NB*2+(n/NB/2)*NB*2*ldinvA,
                        ldinvA,
                        bsinvA);
    }

    if(n <= 2*NB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    // // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    // dim3 grid_gemm((n+NB*2-1)/(NB*2) * batch_count);
    constexpr rocblas_int IB = NB*2;
    rocblas_int blocks = n / IB; // complete blocks - need to do all these together and then deal with partial blocks

    rocblas_int current_n;
    // if(blocks > 0)
    // {
    for(current_n = IB; current_n*2<=n; current_n*=2)
    {
        rocblas_int g = current_n/IB;
        rocblas_int tiles_per_batch = n / current_n / 2;
        // if(tiles_per_batch>1)
        // {
        for(int i = 0; i < batch_count; i++)
        {
            // std::cout<<i<<" th batched gemm current_n "<<current_n;
            trtri_strided_gemm_block<T>(
                handle,
                current_n,
                current_n,
                current_n,
                (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n + i * bsa
                                                            : current_n * lda + i * bsa)),
                lda,
                2*current_n * lda + 2*current_n,
                (const T*)(invA + ((uplo == rocblas_fill_lower) ? 0 + i * bsinvA
                                                                : current_n * ldinvA + current_n + i * bsinvA)),
                (const T*)(invA + ((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n + i * bsinvA
                                                                : 0 + i * bsinvA)),
                (T*)(invA + ((uplo == rocblas_fill_lower) ? current_n + i * bsinvA
                                                        : current_n * ldinvA + i * bsinvA)),
                ldinvA,
                2*current_n * ldinvA + 2*current_n,
                (T*)(invA + ((uplo == rocblas_fill_lower) ? (n-current_n)*ldinvA + i * bsinvA
                                                                : (n-current_n*tiles_per_batch) + i * bsinvA)),
                ldinvA,
                // 2*g*current_n * ldinvA + 2*g*current_n,
                current_n,
                tiles_per_batch);
            // std::cout<<" and clear "<<std::endl;
            // if(tiles_per_batch!=1)
            // {
            // for(int j = 0; j<tiles_per_batch; j++) // this will be replaced with one triangular fill at the end
            //     for(int k = 0; k<current_n; k++)
            //         hipMemsetAsync((T*)invA + ((uplo == rocblas_fill_lower)?(((n-current_n)+k)*ldinvA):((n-current_n*tiles_per_batch)+k*ldinvA)) + current_n*j + bsinvA*i, 0, current_n * sizeof(T), rocblas_stream); //(2*current_n * ldinvA + 2*current_n)
            // }
            std::cout<<" A index "<<((uplo == rocblas_fill_lower) ? current_n + i * bsa: current_n * lda + i * bsa)<<
            " stride_A "<< 2*current_n * lda + 2*current_n<<" invA11 index "<<
            ((uplo == rocblas_fill_lower) ? 0 + i * bsinvA: current_n * ldinvA + current_n + i * bsinvA)<<
            " invA22 index "<<((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n + i * bsinvA: 0 + i * bsinvA)<<
            " invA21 index "<<((uplo == rocblas_fill_lower) ? current_n + i * bsinvA: current_n * ldinvA + i * bsinvA)<< 
            " invA stride "<<2*g*current_n * ldinvA + 2*g*current_n<<" C index "<<
            ((uplo == rocblas_fill_lower) ? (n-current_n)*ldinvA + i * bsinvA: (n-current_n*tiles_per_batch) + i * bsinvA)
            <<" C stride "<<current_n<<
            " tiles_per_batch "<<tiles_per_batch<< " 2*g "<<2*g<<" current_n "<<current_n<<std::endl<<
            std::endl;

        }
        // }
    }

    r = n - current_n - ((n/NB)%2==0? 0:NB) - (n-(n/NB)*NB); //should subtract uneven block and remainder
    std::cout<<"test "<<r<<" test "<<((n/NB)%2==0? 0:NB)<<" n "<<n<<" current_n "<<current_n<<std::endl<<std::endl;

    if(r>0)
    {
        auto C_tmp = rocblas_unique_ptr{
        rocblas::device_malloc(sizeof(T) * r*current_n*batch_count),
        rocblas::device_free};
        trtri_strided_gemm_block<T>(
                handle,
                (uplo == rocblas_fill_lower) ? r:current_n,
                (uplo == rocblas_fill_lower) ? current_n:r,
                (uplo == rocblas_fill_lower) ? r:current_n,
                (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n 
                                                            : current_n * lda )),
                lda,
                bsa,
                (const T*)(invA + ((uplo == rocblas_fill_lower) ? 0 
                                                                : current_n * ldinvA + current_n )),
                (const T*)(invA + ((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n 
                                                                : 0 )),
                (T*)(invA + ((uplo == rocblas_fill_lower) ? current_n 
                                                        : current_n * ldinvA )), 
                ldinvA,
                bsinvA,
                // (T*)(invA + ((uplo == rocblas_fill_lower) ? (n-current_n)*ldinvA  //TODO
                //                                                 : n-current_n )),
                (T*)(C_tmp.get()),
                current_n,
                // ldinvA,
                r*current_n,
                batch_count);
        


        std::cout<<"\n\n remaining even: "<<" r "<<r<<" current_n "<<current_n<<std::endl<<
        " A index "<<((uplo == rocblas_fill_lower) ? current_n: current_n * lda )<<" invA11 index "<<
        ((uplo == rocblas_fill_lower) ? 0 : current_n * ldinvA + current_n )<<" inva22 index "<<((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n: 0 )
        <<" inva21 index "<<((uplo == rocblas_fill_lower) ? current_n : current_n * ldinvA )<<
        " C index "<<((uplo == rocblas_fill_lower) ? (n-current_n)*ldinvA : n-current_n )<<" bsa "<<bsa<<" bsinvA "<<
        bsinvA<<" batch_count "<<batch_count<<" lda "<<lda<<" ldinvA "<<ldinvA<<std::endl<<std::endl;

        // for(int j = 0; j<batch_count; j++) // this will be replaced with one triangular fill at the end
        //     for(int k = 0; k<((uplo == rocblas_fill_lower) ? current_n:r); k++)
        //         hipMemsetAsync((T*)invA + ((uplo == rocblas_fill_lower)?((n-current_n+k)*ldinvA):(n-current_n+k*ldinvA)) + bsinvA*j, 0, ((uplo == rocblas_fill_lower) ? r:current_n) * sizeof(T), rocblas_stream);
    }
    // }

    r=n-current_n-r;
    std::cout<<"Last r "<<r<<std::endl;
    // if(r>0) // solve small remainder 
    // {
    //     current_n = n- r;
    //     // auto C_tmp = rocblas_unique_ptr{
    //     // rocblas::device_malloc(sizeof(T) * r*current_n),
    //     // rocblas::device_free};
        
    //         std::cout<<"\n\n last remaining even: "<<" r "<<r<<" current_n "<<current_n<<std::endl<<
    //         " A index "<<((uplo == rocblas_fill_lower) ? current_n: current_n * lda )<<" invA11 index "<<
    //         ((uplo == rocblas_fill_lower) ? 0 : current_n * ldinvA + current_n )<<" inva22 index "<<((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n: 0 )
    //         <<" inva21 index "<<((uplo == rocblas_fill_lower) ? current_n : current_n * ldinvA )<<
    //         " C index "<<((uplo == rocblas_fill_lower) ? (n-current_n)*ldinvA : n-current_n )<<" bsa "<<bsa<<" bsinvA "<<
    //         bsinvA<<" batch_count "<<batch_count<<std::endl<<std::endl;
    //     trtri_strided_gemm_block<T>(
    //             handle,
    //             (uplo == rocblas_fill_lower) ? r:current_n,
    //             (uplo == rocblas_fill_lower) ? current_n:r,
    //             (uplo == rocblas_fill_lower) ? r:current_n,
    //             (const T*)(A + ((uplo == rocblas_fill_lower) ? current_n 
    //                                                         : current_n * lda )),
    //             lda,
    //             bsa,
    //             (const T*)(invA + ((uplo == rocblas_fill_lower) ? 0 
    //                                                             : current_n * ldinvA + current_n )),
    //             (const T*)(invA + ((uplo == rocblas_fill_lower) ? current_n * ldinvA + current_n 
    //                                                             : 0 )),
    //             (T*)(invA + ((uplo == rocblas_fill_lower) ? current_n 
    //                                                     : current_n * ldinvA )),
    //             ldinvA,
    //             bsinvA,
    //             // (T*)(invA + ((uplo == rocblas_fill_lower) ? (n-current_n)*ldinvA  //TODO
    //             //                                                 : n-current_n )),
    //             (T*)(C_tmp.get()),
    //             current_n,//ldinvA,
    //             r*current_n,
    //             batch_count);

    //     // for(int j = 0; j<batch_count; j++) // this will be replaced with one triangular fill at the end
    //     //     for(int k = 0; k<((uplo == rocblas_fill_lower) ? current_n:r); k++)
    //     //         hipMemsetAsync((T*)invA + ((uplo == rocblas_fill_lower)?((n-current_n+k)*ldinvA):(n-current_n+k*ldinvA)) + bsinvA*j, 0, ((uplo == rocblas_fill_lower) ? r:current_n) * sizeof(T), rocblas_stream);
    // }

    // for(int i = 0; i<batch_count; i++)
    // {
    //     for(int j = 0; j<n; j++)
    //         hipMemsetAsync((T*)invA + ((uplo == rocblas_fill_lower)?(j*ldinvA):(j+1+j*ldinvA)) + bsinvA*i, 0, ((uplo == rocblas_fill_lower) ? j:n-j-1) * sizeof(T), rocblas_stream);
    // }
    size_t blockSize = 128;
    size_t tri_elements_to_zero = num_non_tri_elements(n) * batch_count;
    size_t numBlocks = (tri_elements_to_zero + blockSize - 1) / blockSize;
    // cout << "Executing kernel with n=" << n << ". numBlocks="  << numBlocks << ".";
    hipLaunchKernelGGL(rocblas_tritri_batched_fill<T>, dim3(numBlocks,1,1), dim3(blockSize,1,1), 0, 0, handle, (uplo == rocblas_fill_lower)?rocblas_fill_upper:rocblas_fill_lower, n, num_non_tri_elements(n), ldinvA, n*ldinvA, invA, batch_count);

    // float* mH = (float*)malloc(n*n*sizeof(T));
    // hipMemcpy(mH, invA, n*n*sizeof(T), hipMemcpyDeviceToHost);
    // bool pass = true;

    // for (int b = 0; b < batch_count; b++)
    // {
    //     float* mHOffset = &mH[b * (n*ldinvA)];
    //     for (int x = 0; x < n; x++)
    //     {
    //         for (int y = 0; y < n; y++)
    //         {
    //             if (mHOffset[(x * ldinvA) + y] != 0.f) std::cout << "1";
    //                 else std::cout << "0";
    //             if (x > y)
    //             {
    //                 if (mHOffset[(x * ldinvA) + y] != 0.f)
    //                 {
    //                     // std::cout << std::endl << "error at: " << x << " " << y << " with offset " << b << ", value is " << mHOffset[(x * ldinvA) + y] << std::endl;
    //                     pass = false;
    //                     // break;
    //                 }
    //             }
    //         }
    //         std::cout<<std::endl;
    //     }
    // }
    // if (pass) std::cout << " passed." << std::endl;
    // if (!pass) std::cout << " failed." << std::endl;
    // free(mH);

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
    else// if(n <= 2 * NB)
    {
        return rocblas_trtri_large_batched<T>(
            handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
    }
    // else
    // {
    //     printf("n is %d, n must be less than %d, will return\n", n, 2 * NB);
    //     return rocblas_status_not_implemented;
    // }
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
