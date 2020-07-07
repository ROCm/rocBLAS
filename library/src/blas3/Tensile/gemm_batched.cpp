/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "gemm.hpp"
#include "logging.h"

namespace
{

    template <typename T,
              int DIM_M,
              int DIM_N,
              int BLK_M,
              int BLK_N,
              int BLK_K,
              int DIM_M_A,
              int DIM_N_A,
              int DIM_M_B,
              int DIM_N_B>
    __attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N))) __global__ static void
        gemm_batched_kernel(rocblas_int    M,
                            rocblas_int    N,
                            rocblas_int    K,
                            const T        alpha,
                            const T* const dA_array[],
                            rocblas_int    lda,
                            const T* const dB_array[],
                            rocblas_int    ldb,
                            const T        beta,
                            T* const       dC_array[],
                            rocblas_int    ldc,
                            rocblas_int    batch_count)
    {
        int thx  = threadIdx.x; // thread's m position in C
        int thy  = threadIdx.y; // thread's n position in C
        int idt  = DIM_M * thy + thx; // thread's number
        int blx  = blockIdx.x; // block's m position
        int bly  = blockIdx.y; // block's n position
        int blz  = blockIdx.z; // block's matrix in the batch
        int thxA = idt % DIM_M_A; // thread's m position for loading A
        int thyA = idt / DIM_M_A; // thread's n position for loading A
        int thxB = idt % DIM_M_B; // thread's m position for loading B
        int thyB = idt / DIM_M_B; // thread's n position for loading B

        const T* dA = dA_array[blz];
        const T* dB = dB_array[blz];
        T*       dC = dC_array[blz];

        if(alpha == 0 || K == 0)
        {
            if(beta == 0)
            {
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                {
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                    {
                        int coord_dCm                   = blx * BLK_M + m * DIM_M + thx;
                        int coord_dCn                   = bly * BLK_N + n * DIM_N + thy;
                        dC[coord_dCn * ldc + coord_dCm] = 0.0;
                    }
                }
            }
            else
            {
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                {
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                    {
                        int coord_dCm                   = blx * BLK_M + m * DIM_M + thx;
                        int coord_dCn                   = bly * BLK_N + n * DIM_N + thy;
                        dC[coord_dCn * ldc + coord_dCm] = beta * dC[coord_dCn * ldc + coord_dCm];
                    }
                }
            }
        }
        else
        {
            __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
            __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
            T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

            int coord_A = (blx * BLK_M + thyA * lda) + thxA;
            int coord_B = (bly * BLK_N * ldb + thyB * ldb) + thxB;

            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rC[n][m] = 0.0;

            int kk = 0;
            for(; kk < K; kk += BLK_K)
            {
                for(int n = 0; n < BLK_K; n += DIM_N_A)
                    for(int m = 0; m < BLK_M; m += DIM_M_A)
                        sA[n + thyA][m + thxA] = dA[coord_A + (n * lda + m)];

                for(int n = 0; n < BLK_N; n += DIM_N_B)
                    for(int m = 0; m < BLK_K; m += DIM_M_B)
                        sB[n + thyB][m + thxB] = dB[coord_B + (n * ldb + m)];

                __syncthreads();

                for(int k = 0; k < BLK_K; ++k)
                    for(int n = 0; n < BLK_N / DIM_N; ++n)
                        for(int m = 0; m < BLK_M / DIM_M; ++m)
                            rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

                __syncthreads();

                coord_A += BLK_K * lda;
                coord_B += BLK_K;
            }

            if(beta == 0)
            {
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                {
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                    {
                        int coord_dCm                   = blx * BLK_M + m * DIM_M + thx;
                        int coord_dCn                   = bly * BLK_N + n * DIM_N + thy;
                        dC[coord_dCn * ldc + coord_dCm] = alpha * rC[n][m];
                    }
                }
            }
            else
            {
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                {
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                    {
                        int coord_dCm = blx * BLK_M + m * DIM_M + thx;
                        int coord_dCn = bly * BLK_N + n * DIM_N + thy;
                        dC[coord_dCn * ldc + coord_dCm]
                            = alpha * rC[n][m] + beta * dC[coord_dCn * ldc + coord_dCm];
                    }
                }
            }
        }
    }

    template <typename T,
              int DIM_M,
              int DIM_N,
              int BLK_M,
              int BLK_N,
              int BLK_K,
              int DIM_M_A,
              int DIM_N_A,
              int DIM_M_B,
              int DIM_N_B,
              int alpha,
              int beta>
    __attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N))) __global__ static void
        gemm_batched_kernel(rocblas_int    M,
                            rocblas_int    N,
                            rocblas_int    K,
                            const T* const dA_array[],
                            rocblas_int    lda,
                            const T* const dB_array[],
                            rocblas_int    ldb,
                            T* const       dC_array[],
                            rocblas_int    ldc,
                            rocblas_int    batch_count)
    {
        int thx  = threadIdx.x; // thread's m position in C
        int thy  = threadIdx.y; // thread's n position in C
        int idt  = DIM_M * thy + thx; // thread's number
        int blx  = blockIdx.x; // block's m position
        int bly  = blockIdx.y; // block's n position
        int blz  = blockIdx.z; // block's matrix in the batch
        int thxA = idt % DIM_M_A; // thread's m position for loading A
        int thyA = idt / DIM_M_A; // thread's n position for loading A
        int thxB = idt % DIM_M_B; // thread's m position for loading B
        int thyB = idt / DIM_M_B; // thread's n position for loading B

        const T* dA = dA_array[blz];
        const T* dB = dB_array[blz];
        T*       dC = dC_array[blz];

        __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
        __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
        T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

        int coord_A = (blx * BLK_M + thyA * lda) + thxA;
        int coord_B = (bly * BLK_N * ldb + thyB * ldb) + thxB;

        for(int n = 0; n < BLK_N / DIM_N; ++n)
            for(int m = 0; m < BLK_M / DIM_M; ++m)
                rC[n][m] = 0.0;

        int kk = 0;
        for(; kk < K; kk += BLK_K)
        {
            for(int n = 0; n < BLK_K; n += DIM_N_A)
                for(int m = 0; m < BLK_M; m += DIM_M_A)
                    sA[n + thyA][m + thxA] = dA[coord_A + (n * lda + m)];

            for(int n = 0; n < BLK_N; n += DIM_N_B)
                for(int m = 0; m < BLK_K; m += DIM_M_B)
                    sB[n + thyB][m + thxB] = dB[coord_B + (n * ldb + m)];

            __syncthreads();

            for(int k = 0; k < BLK_K; ++k)
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                        rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

            __syncthreads();

            coord_A += BLK_K * lda;
            coord_B += BLK_K;
        }

        for(int n = 0; n < BLK_N / DIM_N; ++n)
        {
            for(int m = 0; m < BLK_M / DIM_M; ++m)
            {
                int coord_dCm = blx * BLK_M + m * DIM_M + thx;
                int coord_dCn = bly * BLK_N + n * DIM_N + thy;

                if(alpha == 1 && beta == 1)
                {
                    dC[coord_dCn * ldc + coord_dCm] += rC[n][m];
                }
                else if(alpha == 1 && beta == -1)
                {
                    dC[coord_dCn * ldc + coord_dCm] = -dC[coord_dCn * ldc + coord_dCm] + rC[n][m];
                }
                else if(alpha == -1 && beta == 0)
                {
                    dC[coord_dCn * ldc + coord_dCm] = -rC[n][m];
                }
                else if(alpha == 1 && beta == 0)
                {
                    dC[coord_dCn * ldc + coord_dCm] = rC[n][m];
                }
            }
        }
    }

    template <typename T>
    void gemm_batched_solution(rocblas_int    m,
                               rocblas_int    n,
                               rocblas_int    k,
                               const T        alpha,
                               const T* const dA_array[],
                               rocblas_int    lda,
                               const T* const dB_array[],
                               rocblas_int    ldb,
                               const T        beta,
                               T* const       dC_array[],
                               rocblas_int    ldc,
                               rocblas_int    batch_count,
                               hipStream_t    stream)
    {
        if((m % 64 == 0) && (n % 64 == 0) && (k % 4 == 0))
        {
            //m is mult of 64, n is mult of 64, k is mult of 4
            const int dim_m = 16;
            const int dim_n = 16;
            const int blk_m = 64;
            const int blk_n = 64;
            const int blk_k = 4;
            dim3      dimBlock(dim_m, dim_n, 1);
            dim3      dimGrid(m / blk_m, n / blk_n, batch_count);
            if(alpha == T(1.0) && beta == T(1.0))
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       1,
                                                                       1>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else if(alpha == 1.0 && beta == -1.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       1,
                                                                       -1>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else if(alpha == 1.0 && beta == 0.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       1,
                                                                       0>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else if(alpha == -1.0 && beta == 0.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       -1,
                                                                       0>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   beta,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
        }
        else if((m % 32 == 0) && (n % 32 == 0) && (k % 8 == 0))
        {
            // m is mult of 32, n is mult of 32, k is mult of 8
            const int dim_m = 16;
            const int dim_n = 16;
            const int blk_m = 32;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_m, dim_n, 1);
            dim3      dimGrid(m / blk_m, n / blk_n, batch_count);
            if(alpha == 1.0 && beta == 1.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       1,
                                                                       1>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else if(alpha == 1.0 && beta == -1.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       1,
                                                                       -1>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else if(alpha == 1.0 && beta == 0.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       1,
                                                                       0>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else if(alpha == -1.0 && beta == 0.0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n,
                                                                       -1,
                                                                       0>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batched_kernel<T,
                                                                       dim_m,
                                                                       dim_n,
                                                                       blk_m,
                                                                       blk_n,
                                                                       blk_k,
                                                                       blk_m,
                                                                       blk_k,
                                                                       blk_k,
                                                                       blk_n>),
                                   dimGrid,
                                   dimBlock,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   dA_array,
                                   lda,
                                   dB_array,
                                   ldb,
                                   beta,
                                   dC_array,
                                   ldc,
                                   batch_count);
            }
        }
    }

    template <typename>
    constexpr char rocblas_gemm_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_gemm_batched_name<rocblas_half>[] = "rocblas_hgemm_batched";
    template <>
    constexpr char rocblas_gemm_batched_name<float>[] = "rocblas_sgemm_batched";
    template <>
    constexpr char rocblas_gemm_batched_name<double>[] = "rocblas_dgemm_batched";
    template <>
    constexpr char rocblas_gemm_batched_name<rocblas_float_complex>[] = "rocblas_cgemm_batched";
    template <>
    constexpr char rocblas_gemm_batched_name<rocblas_double_complex>[] = "rocblas_zgemm_batched";

    /*******************************************************************************
    * Batched GEMM implementation
    ******************************************************************************/
    template <typename T>
    rocblas_status rocblas_gemm_batched_impl(rocblas_handle    handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             rocblas_int       ld_a,
                                             const T* const    B[],
                                             rocblas_int       ld_b,
                                             const T*          beta,
                                             T* const          C[],
                                             rocblas_int       ld_c,
                                             rocblas_int       b_c)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device
        T alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(
            copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Perform logging
        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto trans_a_letter = rocblas_transpose_letter(trans_a);
            auto trans_b_letter = rocblas_transpose_letter(trans_b);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_batched_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          log_trace_scalar_value(alpha),
                          A,
                          ld_a,
                          B,
                          ld_b,
                          log_trace_scalar_value(beta),
                          C,
                          ld_c,
                          b_c);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f gemm_batched -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          trans_a_letter,
                          "--transposeB",
                          trans_b_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "-k",
                          k,
                          LOG_BENCH_SCALAR_VALUE(alpha),
                          "--lda",
                          ld_a,
                          "--ldb",
                          ld_b,
                          LOG_BENCH_SCALAR_VALUE(beta),
                          "--ldc",
                          ld_c,
                          "--batch_count",
                          b_c);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_gemm_batched_name<T>,
                            "transA",
                            trans_a_letter,
                            "transB",
                            trans_b_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "K",
                            k,
                            "alpha",
                            value_category(*alpha),
                            "lda",
                            ld_a,
                            "ldb",
                            ld_b,
                            "beta",
                            value_category(*beta),
                            "ldc",
                            ld_c,
                            "batch_count",
                            b_c);
        }

        auto validArgs = validateArgs(
            handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, b_c);

        if(validArgs != rocblas_status_continue)
            return validArgs;

        // call rocBLAS source code if
        //     (NN)
        // and
        //     ((m is mult of 64 and n is mult of 64 and k is mult of 4)
        //     or
        //     (m is mult of 32 and n is mult of 32 and k is mult of 8))
        // and
        //     (m*n*k is small enough)
        if((trans_a == rocblas_operation_none) && (trans_b == rocblas_operation_none)
           && (((m % 64 == 0) && (n % 64 == 0) && (k % 4 == 0))
               || ((m % 32 == 0) && (n % 32 == 0) && (k % 8 == 0)))
           && (size_t(m) * size_t(n) * size_t(k) < 1024 * 1024 * 1024))
        {
            hipStream_t rocblas_stream = handle->rocblas_stream;

            gemm_batched_solution(
                m, n, k, *alpha, A, ld_a, B, ld_b, *beta, C, ld_c, b_c, rocblas_stream);

            return rocblas_status_success;
        }
        else
        {
            return rocblas_gemm_template<true>(handle,
                                               trans_a,
                                               trans_b,
                                               m,
                                               n,
                                               k,
                                               alpha,
                                               A,
                                               0,
                                               ld_a,
                                               0,
                                               B,
                                               0,
                                               ld_b,
                                               0,
                                               beta,
                                               C,
                                               0,
                                               ld_c,
                                               0,
                                               b_c);
        }
    }
}

/*******************************************************************************
 * Batched GEMM APIs
 ******************************************************************************/

extern "C" {
rocblas_status rocblas_hgemm_batched(rocblas_handle            handle,
                                     rocblas_operation         trans_a,
                                     rocblas_operation         trans_b,
                                     rocblas_int               m,
                                     rocblas_int               n,
                                     rocblas_int               k,
                                     const rocblas_half*       alpha,
                                     const rocblas_half* const A[],
                                     rocblas_int               ld_a,
                                     const rocblas_half* const B[],
                                     rocblas_int               ld_b,
                                     const rocblas_half*       beta,
                                     rocblas_half* const       C[],
                                     rocblas_int               ld_c,
                                     rocblas_int               b_c)
try
{
    return rocblas_gemm_batched_impl<rocblas_half>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, b_c);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_sgemm_batched(rocblas_handle     handle,
                                     rocblas_operation  trans_a,
                                     rocblas_operation  trans_b,
                                     rocblas_int        m,
                                     rocblas_int        n,
                                     rocblas_int        k,
                                     const float*       alpha,
                                     const float* const A[],
                                     rocblas_int        ld_a,
                                     const float* const B[],
                                     rocblas_int        ld_b,
                                     const float*       beta,
                                     float* const       C[],
                                     rocblas_int        ld_c,
                                     rocblas_int        b_c)
try
{
    return rocblas_gemm_batched_impl<float>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, b_c);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dgemm_batched(rocblas_handle      handle,
                                     rocblas_operation   trans_a,
                                     rocblas_operation   trans_b,
                                     rocblas_int         m,
                                     rocblas_int         n,
                                     rocblas_int         k,
                                     const double*       alpha,
                                     const double* const A[],
                                     rocblas_int         ld_a,
                                     const double* const B[],
                                     rocblas_int         ld_b,
                                     const double*       beta,
                                     double* const       C[],
                                     rocblas_int         ld_c,
                                     rocblas_int         b_c)
try
{
    return rocblas_gemm_batched_impl<double>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, b_c);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cgemm_batched(rocblas_handle                     handle,
                                     rocblas_operation                  trans_a,
                                     rocblas_operation                  trans_b,
                                     rocblas_int                        m,
                                     rocblas_int                        n,
                                     rocblas_int                        k,
                                     const rocblas_float_complex*       alpha,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        ld_a,
                                     const rocblas_float_complex* const B[],
                                     rocblas_int                        ld_b,
                                     const rocblas_float_complex*       beta,
                                     rocblas_float_complex* const       C[],
                                     rocblas_int                        ld_c,
                                     rocblas_int                        b_c)
try
{
    return rocblas_gemm_batched_impl<rocblas_float_complex>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, b_c);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zgemm_batched(rocblas_handle                      handle,
                                     rocblas_operation                   trans_a,
                                     rocblas_operation                   trans_b,
                                     rocblas_int                         m,
                                     rocblas_int                         n,
                                     rocblas_int                         k,
                                     const rocblas_double_complex*       alpha,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         ld_a,
                                     const rocblas_double_complex* const B[],
                                     rocblas_int                         ld_b,
                                     const rocblas_double_complex*       beta,
                                     rocblas_double_complex* const       C[],
                                     rocblas_int                         ld_c,
                                     rocblas_int                         b_c)
try
{
    return rocblas_gemm_batched_impl<rocblas_double_complex>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, b_c);
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Batched GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_batched_kernel_name(rocblas_handle      handle,
                                                 rocblas_operation   trans_a,
                                                 rocblas_operation   trans_b,
                                                 rocblas_int         m,
                                                 rocblas_int         n,
                                                 rocblas_int         k,
                                                 const rocblas_half* alpha,
                                                 const rocblas_half* A[],
                                                 rocblas_int         ld_a,
                                                 const rocblas_half* B[],
                                                 rocblas_int         ld_b,
                                                 const rocblas_half* beta,
                                                 rocblas_half*       C[],
                                                 rocblas_int         ld_c,
                                                 rocblas_int         b_c)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_sgemm_batched_kernel_name(rocblas_handle    handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 rocblas_int       k,
                                                 const float*      alpha,
                                                 const float*      A[],
                                                 rocblas_int       ld_a,
                                                 const float*      B[],
                                                 rocblas_int       ld_b,
                                                 const float*      beta,
                                                 float*            C[],
                                                 rocblas_int       ld_c,
                                                 rocblas_int       b_c)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_dgemm_batched_kernel_name(rocblas_handle    handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 rocblas_int       k,
                                                 const double*     alpha,
                                                 const double*     A[],
                                                 rocblas_int       ld_a,
                                                 const double*     B[],
                                                 rocblas_int       ld_b,
                                                 const double*     beta,
                                                 double*           C[],
                                                 rocblas_int       ld_c,
                                                 rocblas_int       b_c)
{
    return rocblas_status_not_implemented;
}

} // extern "C"
