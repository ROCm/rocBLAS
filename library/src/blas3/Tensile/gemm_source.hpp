/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"

namespace
{
    // large index support is not needed for lda, ldb, ldc as this kernel is only intended for small m, n, k
    // general alpha, beta, m, n, k
    template <typename T,
              int  DIM_M,
              int  DIM_N,
              int  BLK_M,
              int  BLK_N,
              int  BLK_K,
              int  DIM_M_A,
              int  DIM_N_A,
              int  DIM_M_B,
              int  DIM_N_B,
              bool BETA_EQ_ZERO,
              char TRANS_A,
              char TRANS_B,
              typename TConstPtr,
              typename TPtr>
    __attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N))) ROCBLAS_KERNEL void
        gemm_batched_general_kernel(rocblas_int    M,
                                    rocblas_int    N,
                                    rocblas_int    K,
                                    const T        alpha,
                                    TConstPtr*     dA_input,
                                    rocblas_int    lda,
                                    rocblas_stride a_st_or_of,
                                    TConstPtr*     dB_input,
                                    rocblas_int    ldb,
                                    rocblas_stride b_st_or_of,
                                    const T        beta,
                                    TPtr*          dC_input,
                                    rocblas_int    ldc,
                                    rocblas_stride c_st_or_of,
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

        auto* dA = load_ptr_batch(dA_input, blz, a_st_or_of);
        auto* dB = load_ptr_batch(dB_input, blz, b_st_or_of);
        auto* dC = load_ptr_batch(dC_input, blz, c_st_or_of);

        __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
        __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
        T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

        int a_i_offset = thxA + BLK_M * blx;
        int a_j_offset = thyA;
        int b_i_offset = thxB;
        int b_j_offset = thyB + BLK_N * bly;

        for(int n = 0; n < BLK_N / DIM_N; ++n)
            for(int m = 0; m < BLK_M / DIM_M; ++m)
                rC[n][m] = 0.0;

        int kk = 0;
        for(; kk < K; kk += BLK_K)
        {
            for(int n = 0; n < BLK_K; n += DIM_N_A)
            {
                for(int m = 0; m < BLK_M; m += DIM_M_A)
                {
                    int i = m + a_i_offset;
                    int j = n + kk + a_j_offset;
                    if(i < M && j < K)
                    {
                        if(TRANS_A == 'N')
                        {
                            sA[n + thyA][m + thxA] = dA[i + j * lda];
                        }
                        else if(TRANS_A == 'T')
                        {
                            sA[n + thyA][m + thxA] = dA[i * lda + j];
                        }
                        else if(TRANS_A == 'C')
                        {
                            sA[n + thyA][m + thxA] = conj(dA[i * lda + j]);
                        }
                    }
                    else
                    {
                        sA[n + thyA][m + thxA] = 0.0;
                    }
                }
            }

            for(int n = 0; n < BLK_N; n += DIM_N_B)
            {
                for(int m = 0; m < BLK_K; m += DIM_M_B)
                {
                    int i = m + kk + b_i_offset;
                    int j = n + b_j_offset;
                    if(i < K && j < N)
                    {
                        if(TRANS_B == 'N')
                        {
                            sB[n + thyB][m + thxB] = dB[i + j * ldb];
                        }
                        else if(TRANS_B == 'T')
                        {
                            sB[n + thyB][m + thxB] = dB[i * ldb + j];
                        }
                        else if(TRANS_B == 'C')
                        {
                            sB[n + thyB][m + thxB] = conj(dB[i * ldb + j]);
                        }
                    }
                    else
                    {
                        sB[n + thyB][m + thxB] = 0;
                    }
                }
            }

            __syncthreads();

            for(int k = 0; k < BLK_K; ++k)
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                        rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

            __syncthreads();
        }

        for(int n = 0; n < BLK_N / DIM_N; ++n)
        {
            for(int m = 0; m < BLK_M / DIM_M; ++m)
            {
                int coord_dCm = blx * BLK_M + m * DIM_M + thx;
                int coord_dCn = bly * BLK_N + n * DIM_N + thy;
                if(coord_dCn < N && coord_dCm < M)
                {
                    if(BETA_EQ_ZERO)
                    {
                        dC[coord_dCn * ldc + coord_dCm] = alpha * rC[n][m];
                    }
                    else
                    {
                        dC[coord_dCn * ldc + coord_dCm]
                            = alpha * rC[n][m] + beta * dC[coord_dCn * ldc + coord_dCm];
                    }
                }
            }
        }
    }

    // large index support is not needed for lda, ldb, ldc as this kernel is only intended for small m, n, k
    // general alpha, beta, restricted m, n, k
    template <typename T,
              int  DIM_M,
              int  DIM_N,
              int  BLK_M,
              int  BLK_N,
              int  BLK_K,
              int  DIM_M_A,
              int  DIM_N_A,
              int  DIM_M_B,
              int  DIM_N_B,
              bool BETA_EQ_ZERO,
              char TRANS_A,
              char TRANS_B,
              typename TConstPtr,
              typename TPtr>
    __attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N))) ROCBLAS_KERNEL void
        gemm_batched_kernel(rocblas_int    M,
                            rocblas_int    N,
                            rocblas_int    K,
                            const T        alpha,
                            TConstPtr*     dA_input,
                            rocblas_int    lda,
                            rocblas_stride a_st_or_of,
                            TConstPtr*     dB_input,
                            rocblas_int    ldb,
                            rocblas_stride b_st_or_of,
                            const T        beta,
                            TPtr*          dC_input,
                            rocblas_int    ldc,
                            rocblas_stride c_st_or_of,
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

        auto* dA = load_ptr_batch(dA_input, blz, a_st_or_of);
        auto* dB = load_ptr_batch(dB_input, blz, b_st_or_of);
        auto* dC = load_ptr_batch(dC_input, blz, c_st_or_of);

        __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
        __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
        T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

        for(int n = 0; n < BLK_N / DIM_N; ++n)
            for(int m = 0; m < BLK_M / DIM_M; ++m)
                rC[n][m] = 0.0;

        size_t coord_A, coord_B;
        if(TRANS_A == 'N')
            coord_A = (thxA + blx * BLK_M) + (thyA)*lda;
        else if(TRANS_A == 'T' || TRANS_A == 'C')
            coord_A = (thxA + blx * BLK_M) * lda + (thyA);

        if(TRANS_B == 'N')
            coord_B = thxB + (bly * BLK_N + thyB) * ldb;
        else if(TRANS_B == 'T' || TRANS_B == 'C')
            coord_B = thxB * ldb + (bly * BLK_N + thyB);

        int kk = 0;
        for(; kk < K; kk += BLK_K)
        {
            for(int n = 0; n < BLK_K; n += DIM_N_A)
                for(int m = 0; m < BLK_M; m += DIM_M_A)
                    if(TRANS_A == 'N')
                    {
                        sA[n + thyA][m + thxA] = dA[coord_A + m + n * lda];
                    }
                    else if(TRANS_A == 'T')
                    {
                        sA[n + thyA][m + thxA] = dA[coord_A + m * lda + n];
                    }
                    else if(TRANS_A == 'C')
                    {
                        sA[n + thyA][m + thxA] = conj(dA[coord_A + m * lda + n]);
                    }

            for(int n = 0; n < BLK_N; n += DIM_N_B)
                for(int m = 0; m < BLK_K; m += DIM_M_B)
                    if(TRANS_B == 'N')
                    {
                        sB[n + thyB][m + thxB] = dB[coord_B + m + n * ldb];
                    }
                    else if(TRANS_B == 'T')
                    {
                        sB[n + thyB][m + thxB] = dB[coord_B + m * ldb + n];
                    }
                    else if(TRANS_B == 'C')
                    {
                        sB[n + thyB][m + thxB] = conj(dB[coord_B + m * ldb + n]);
                    }

            __syncthreads();

            for(int k = 0; k < BLK_K; ++k)
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                        rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

            __syncthreads();

            if(TRANS_A == 'N')
                coord_A += BLK_K * lda;
            else if(TRANS_A == 'T' || TRANS_A == 'C')
                coord_A += BLK_K;

            if(TRANS_B == 'N')
                coord_B += BLK_K;
            else if(TRANS_B == 'T' || TRANS_B == 'C')
                coord_B += BLK_K * ldb;
        }

        for(int n = 0; n < BLK_N / DIM_N; ++n)
        {
            for(int m = 0; m < BLK_M / DIM_M; ++m)
            {
                int coord_dCm = blx * BLK_M + m * DIM_M + thx;
                int coord_dCn = bly * BLK_N + n * DIM_N + thy;

                if(BETA_EQ_ZERO)
                {
                    dC[coord_dCn * ldc + coord_dCm] = alpha * rC[n][m];
                }
                else
                {
                    dC[coord_dCn * ldc + coord_dCm]
                        = alpha * rC[n][m] + beta * dC[coord_dCn * ldc + coord_dCm];
                }
            }
        }
    }

    // large index support is not needed for lda, ldb, ldc as this kernel is only intended for small m, n, k
    // templated alpha, beta, restricted m, n, k
    template <typename T,
              int  DIM_M,
              int  DIM_N,
              int  BLK_M,
              int  BLK_N,
              int  BLK_K,
              int  DIM_M_A,
              int  DIM_N_A,
              int  DIM_M_B,
              int  DIM_N_B,
              int  alpha,
              int  beta,
              char TRANS_A,
              char TRANS_B,
              typename TConstPtr,
              typename TPtr>
    __attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N))) ROCBLAS_KERNEL void
        gemm_batched_kernel(rocblas_int    M,
                            rocblas_int    N,
                            rocblas_int    K,
                            TConstPtr*     dA_input,
                            rocblas_int    lda,
                            rocblas_stride a_st_or_of,
                            TConstPtr*     dB_input,
                            rocblas_int    ldb,
                            rocblas_stride b_st_or_of,
                            TPtr*          dC_input,
                            rocblas_int    ldc,
                            rocblas_stride c_st_or_of,
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

        auto* dA = load_ptr_batch(dA_input, blz, a_st_or_of);
        auto* dB = load_ptr_batch(dB_input, blz, b_st_or_of);
        auto* dC = load_ptr_batch(dC_input, blz, c_st_or_of);

        __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
        __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
        T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

        size_t coord_A, coord_B;
        if(TRANS_A == 'N')
            coord_A = (blx * BLK_M + thxA) + thyA * lda;
        else if(TRANS_A == 'T' || TRANS_A == 'C')
            coord_A = (blx * BLK_M + thxA) * lda + thyA;
        if(TRANS_B == 'N')
            coord_B = (bly * BLK_N + thyB) * ldb + thxB;
        else if(TRANS_B == 'T' || TRANS_B == 'C')
            coord_B = (bly * BLK_N + thyB) + thxB * ldb;

        for(int n = 0; n < BLK_N / DIM_N; ++n)
            for(int m = 0; m < BLK_M / DIM_M; ++m)
                rC[n][m] = 0.0;

        int kk = 0;
        for(; kk < K; kk += BLK_K)
        {
            for(int n = 0; n < BLK_K; n += DIM_N_A)
                for(int m = 0; m < BLK_M; m += DIM_M_A)
                    if(TRANS_A == 'N')
                        sA[n + thyA][m + thxA] = dA[coord_A + (n * lda + m)];
                    else if(TRANS_A == 'T')
                        sA[n + thyA][m + thxA] = dA[coord_A + (n + m * lda)];
                    else if(TRANS_A == 'C')
                        sA[n + thyA][m + thxA] = conj(dA[coord_A + (n + m * lda)]);

            for(int n = 0; n < BLK_N; n += DIM_N_B)
                for(int m = 0; m < BLK_K; m += DIM_M_B)
                    if(TRANS_B == 'N')
                        sB[n + thyB][m + thxB] = dB[coord_B + (n * ldb + m)];
                    else if(TRANS_B == 'T')
                        sB[n + thyB][m + thxB] = dB[coord_B + (n + m * ldb)];
                    else if(TRANS_B == 'C')
                        sB[n + thyB][m + thxB] = conj(dB[coord_B + (n + m * ldb)]);

            __syncthreads();

            for(int k = 0; k < BLK_K; ++k)
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                        rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

            __syncthreads();

            if(TRANS_A == 'N')
                coord_A += BLK_K * lda;
            else if(TRANS_A == 'T' || TRANS_A == 'C')
                coord_A += BLK_K;

            if(TRANS_B == 'N')
                coord_B += BLK_K;
            else if(TRANS_B == 'T' || TRANS_B == 'C')
                coord_B += BLK_K * ldb;
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

    template <bool BATCHED, typename T, typename TConstPtr, typename TPtr>
    void gemm_source_solution(rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const T           alpha,
                              TConstPtr*        dA,
                              rocblas_int       lda,
                              rocblas_stride    stride_a,
                              rocblas_stride    offset_a,
                              TConstPtr*        dB,
                              rocblas_int       ldb,
                              rocblas_stride    stride_b,
                              rocblas_stride    offset_b,
                              const T           beta,
                              TPtr*             dC,
                              rocblas_int       ldc,
                              rocblas_stride    stride_c,
                              rocblas_stride    offset_c,
                              rocblas_int       batch_count,
                              hipStream_t       stream)
    {
        // gemm has same behavior for alpha == 0 and k == 0. Special code is needed
        // for alpha == 0, no special code is needed for k == 0. It is more efficient
        // setting k = 0 than adding extra code to a kernel to handle alpha == 0
        if(alpha == 0)
            k = 0;

        TConstPtr*     dA_krn;
        TConstPtr*     dB_krn;
        TPtr*          dC_krn;
        rocblas_stride a_st_or_of;
        rocblas_stride b_st_or_of;
        rocblas_stride c_st_or_of;

        if(BATCHED)
        {
            dA_krn     = dA;
            dB_krn     = dB;
            dC_krn     = dC;
            a_st_or_of = offset_a;
            b_st_or_of = offset_b;
            c_st_or_of = offset_c;
        }
        else
        {
            dA_krn     = dA + offset_a;
            dB_krn     = dB + offset_b;
            dC_krn     = dC + offset_c;
            a_st_or_of = stride_a;
            b_st_or_of = stride_b;
            c_st_or_of = stride_c;
        }

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
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(alpha == 1.0 && beta == -1.0)
            {
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                else if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(alpha == 1.0 && beta == 0.0)
            {
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(alpha == -1.0 && beta == 0.0)
            {
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(beta == 0)
            {
                // general alpha; beta == 0
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T' , 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else
            {
                // general alpha, beta
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
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
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'N', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'T', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 1, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(alpha == 1.0 && beta == -1.0)
            {
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'N', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'T', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, -1, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(alpha == 1.0 && beta == 0.0)
            {
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'N', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'T', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 1, 0, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(alpha == -1.0 && beta == 0.0)
            {
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'N', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'T', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, -1, 0, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else if(beta == 0)
            {
                // general alpha; beta == 0
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'C' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'N' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'T' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'C' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'C' >),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else
            {
                // general alpha, beta
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
        }
        else
        {
            const int dim_m = 16;
            const int dim_n = 16;
            const int blk_m = 32;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_m, dim_n, 1);
            dim3      dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batch_count);
            if(beta == 0)
            {
                // general m, n, k, alpha; beta == 0
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true,'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, true, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
            else
            {
                // general m, n, k, alpha, beta
                // clang-format off
                if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'N'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'C', 'T'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'N', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                    hipLaunchKernelGGL((gemm_batched_general_kernel
                    <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, false, 'T', 'C'>),
                    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of,
                    dB_krn, ldb, b_st_or_of, beta, dC_krn, ldc, c_st_or_of, batch_count);
                // clang-format on
            }
        }
    }
}
