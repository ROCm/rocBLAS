/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

template <bool HERK, typename T, typename U>
ROCBLAS_KERNEL_ILF void syr2k_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n && from <= to)
    {
        auto& e = C[ty * size_t(ldc) + tx];
        e       = beta ? beta * e : 0;
        if(HERK && from == to)
            e = std::real(e);
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, bool HERK, typename U, typename V, typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
syr2k_scale_kernel(bool           upper,
                   rocblas_int    n,
                   rocblas_int    k,
                   U              alpha_host_device,
                   V              beta_host_device,
                   W              CP_array,
                   rocblas_int    ldc,
                   rocblas_stride c_st_or_of)
{
    auto beta = load_scalar(beta_host_device);

    if((!HERK && beta == 1))
        return;
    else if(HERK)
    {
        // for herk, if alpha != 0 we need imaginary clear on diagonal
        auto alpha = load_scalar(alpha_host_device);
        if(beta == 1 && (k == 0 || alpha == 0))
            return;
    }

    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, c_st_or_of);
    syr2k_scale_device<HERK>(upper, n, beta, C, ldc);
}

template <typename T,
          int  DIM,
          bool BETA_EQ_ZERO,
          bool HERK,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM* DIM)
syrkx_herkx_small_kernel(rocblas_int    N,
                         rocblas_int    K,
                         const T        alpha,
                         TConstPtr*     dA_array,
                         rocblas_int    lda,
                         rocblas_stride stride_a,
                         TConstPtr*     dB_array,
                         rocblas_int    ldb,
                         rocblas_stride stride_b,
                         const T        beta,
                         TPtr*          dC_array,
                         rocblas_int    ldc,
                         rocblas_stride stride_c,
                         rocblas_int    batch_count)
{
    int thx = threadIdx.x; // thread's m position
    int thy = threadIdx.y; // thread's n position
    int blx = blockIdx.x; // block's m position
    int bly = blockIdx.y; // block's n position
    int blz = blockIdx.z; // block's matrix in the batch

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);

    __shared__ T sA[DIM][DIM]; // shared memory for A
    __shared__ T sB[DIM][DIM]; // shared memory for B
    T            rC = 0; // register for C

    int i1 = thx + blx * DIM;
    int i2 = thy + bly * DIM;
    int i3_a;
    int i3_b;

    for(int kk = 0; kk < K; kk += DIM)
    {
        i3_a = kk + thy;
        if(i1 < N && i3_a < K)
        {
            if(TRANS == 'N')
                sA[thy][thx] = dA[i1 + i3_a * size_t(lda)];
            if(TRANS == 'T')
                sA[thy][thx] = dA[i3_a + i1 * size_t(lda)];
            if(TRANS == 'C')
                sA[thy][thx] = conj_if_true<HERK>(dA[i3_a + i1 * size_t(lda)]);
        }
        else
        {
            sA[thy][thx] = 0.0;
        }

        i3_b = kk + thx;
        if(i2 < N && i3_b < K)
        {
            if(TRANS == 'C')
                sB[thy][thx] = dB[i3_b + i2 * size_t(ldb)];
            if(TRANS == 'T')
                sB[thy][thx] = dB[i3_b + i2 * size_t(ldb)];
            if(TRANS == 'N')
                sB[thy][thx] = conj_if_true<HERK>(dB[i2 + i3_b * size_t(ldb)]);
        }
        else
        {
            sB[thy][thx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < DIM; ++k)
            rC += sA[k][thx] * sB[thy][k];

        __syncthreads();
    }

    if((UPLO == 'L' && i2 <= i1 && i1 < N) || (UPLO == 'U' && i1 <= i2 && i2 < N))
    {
        if(BETA_EQ_ZERO)
            dC[i1 + i2 * size_t(ldc)] = alpha * rC;
        else
            dC[i1 + i2 * size_t(ldc)] = alpha * rC + beta * dC[i1 + i2 * size_t(ldc)];

        // Zero out imaginary part of diagonal if herk
        if(HERK && i1 == i2)
            dC[i1 + i2 * size_t(ldc)] = std::real(dC[i1 + i2 * size_t(ldc)]);
    }
}

// N and K must be multiples of DIM
template <typename T,
          int  DIM,
          bool BETA_EQ_ZERO,
          bool HERK,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM* DIM)
syrkx_herkx_small_restrict_kernel(rocblas_int    N,
                                  rocblas_int    K,
                                  const T        alpha,
                                  TConstPtr*     dA_array,
                                  rocblas_int    lda,
                                  rocblas_stride stride_a,
                                  TConstPtr*     dB_array,
                                  rocblas_int    ldb,
                                  rocblas_stride stride_b,
                                  const T        beta,
                                  TPtr*          dC_array,
                                  rocblas_int    ldc,
                                  rocblas_stride stride_c,
                                  rocblas_int    batch_count)
{
    int thx = threadIdx.x; // thread's m position
    int thy = threadIdx.y; // thread's n position
    int blx = blockIdx.x; // block's m position
    int bly = blockIdx.y; // block's n position
    int blz = blockIdx.z; // block's matrix in the batch

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);

    __shared__ T sA[DIM][DIM]; // shared memory for A
    __shared__ T sB[DIM][DIM]; // shared memory for B
    T            rC = 0; // register for C

    int i1 = thx + blx * DIM;
    int i2 = thy + bly * DIM;
    int i3_a;
    int i3_b;

    int kk = 0;
    for(; kk < K; kk += DIM)
    {
        i3_a = kk + thy;
        if(TRANS == 'N')
            sA[thy][thx] = dA[i1 + i3_a * size_t(lda)];
        if(TRANS == 'T')
            sA[thy][thx] = dA[i3_a + i1 * size_t(lda)];
        if(TRANS == 'C')
            sA[thy][thx] = conj_if_true<HERK>(dA[i3_a + i1 * size_t(lda)]);

        i3_b = kk + thx;
        if(TRANS == 'C')
            sB[thy][thx] = dB[i3_b + i2 * size_t(ldb)];
        if(TRANS == 'T')
            sB[thy][thx] = dB[i3_b + i2 * size_t(ldb)];
        if(TRANS == 'N')
            sB[thy][thx] = conj_if_true<HERK>(dB[i2 + i3_b * size_t(ldb)]);

        __syncthreads();

        for(int k = 0; k < DIM; ++k)
            rC += sA[k][thx] * sB[thy][k];

        __syncthreads();
    }

    if((UPLO == 'L' && i2 <= i1) || (UPLO == 'U' && i1 <= i2))
    {
        if(BETA_EQ_ZERO)
            dC[i1 + i2 * size_t(ldc)] = alpha * rC;
        else
            dC[i1 + i2 * size_t(ldc)] = alpha * rC + beta * dC[i1 + i2 * size_t(ldc)];

        // Zero out imaginary part of diagonal if herk
        if(HERK && i1 == i2)
            dC[i1 + i2 * size_t(ldc)] = std::real(dC[i1 + i2 * size_t(ldc)]);
    }
}

// general alpha, beta, m, n, k
template <typename T,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          bool BETA_EQ_ZERO,
          bool HERK,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
syrkx_herkx_general_kernel(rocblas_int    N,
                           rocblas_int    K,
                           const T        alpha,
                           TConstPtr*     dA_array,
                           rocblas_int    lda,
                           rocblas_stride stride_a,
                           TConstPtr*     dB_array,
                           rocblas_int    ldb,
                           rocblas_stride stride_b,
                           const T        beta,
                           TPtr*          dC_array,
                           rocblas_int    ldc,
                           rocblas_stride stride_c,
                           rocblas_int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);

    __shared__ T sA[BLK_K][BLK_N]; // shared memory for A
    __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
    T            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

    int a_i_offset = thxA + BLK_N * blx;
    int a_j_offset = thyA;
    int b_i_offset = thxB;
    int b_j_offset = thyB + BLK_N * bly;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_N / DIM_N; ++m)
            rC[n][m] = 0.0;

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        int i = a_i_offset;
        int j = kk + a_j_offset;
        if(i < N && j < K)
        {
            if(TRANS == 'N')
                sA[thyA][thxA] = dA[i + j * size_t(lda)];
            if(TRANS == 'T')
                sA[thyA][thxA] = dA[i * size_t(lda) + j];
            if(TRANS == 'C')
                sA[thyA][thxA] = conj_if_true<HERK>(dA[i * size_t(lda) + j]);
        }
        else
        {
            sA[thyA][thxA] = 0.0;
        }
        i = kk + b_i_offset;
        j = b_j_offset;
        if(i < K && j < N)
        {
            if(TRANS == 'C')
                sB[thyB][thxB] = dB[i + j * size_t(ldb)];
            if(TRANS == 'T')
                sB[thyB][thxB] = dB[i + j * size_t(ldb)];
            if(TRANS == 'N')
                sB[thyB][thxB] = conj_if_true<HERK>(dB[i * size_t(ldb) + j]);
        }
        else
        {
            sB[thyB][thxB] = 0;
        }

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_N / DIM_N; ++m)
                    rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_N / DIM_N; ++m)
        {
            int coord_dCm = blx * BLK_N + m * DIM_N + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;
            if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
               || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
            {
                if(BETA_EQ_ZERO)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = alpha * rC[n][m];
                else
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = alpha * rC[n][m] + beta * dC[coord_dCn * size_t(ldc) + coord_dCm];

                // Zero out imaginary part of diagonal if herk
                if(HERK && coord_dCn == coord_dCm)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = std::real(dC[coord_dCn * size_t(ldc) + coord_dCm]);
            }
        }
    }
}

// large index support is not needed for lda, ldb, ldc as this kernel is only intended for small m, n, k
// general alpha, beta, restricted n, k
template <typename T,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          bool BETA_EQ_ZERO,
          bool HERK,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
syrkx_herkx_restricted_kernel(rocblas_int    N,
                              rocblas_int    K,
                              const T        alpha,
                              TConstPtr*     dA_array,
                              rocblas_int    lda,
                              rocblas_stride stride_a,
                              TConstPtr*     dB_array,
                              rocblas_int    ldb,
                              rocblas_stride stride_b,
                              const T        beta,
                              TPtr*          dC_array,
                              rocblas_int    ldc,
                              rocblas_stride stride_c,
                              rocblas_int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);

    __shared__ T sA[BLK_K][BLK_N]; // shared memory for A
    __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
    T            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_N / DIM_N; ++m)
            rC[n][m] = 0.0;

    size_t coord_A, coord_B;
    if(TRANS == 'N')
        coord_A = (thxA + blx * BLK_N) + (thyA)*size_t(lda);
    if(TRANS == 'T')
        coord_A = (thxA + blx * BLK_N) * size_t(lda) + (thyA);
    if(TRANS == 'C')
        coord_A = (thxA + blx * BLK_N) * size_t(lda) + (thyA);

    if(TRANS == 'C')
        coord_B = thxB + (bly * BLK_N + thyB) * size_t(ldb);
    if(TRANS == 'T')
        coord_B = thxB + (bly * BLK_N + thyB) * size_t(ldb);
    if(TRANS == 'N')
        coord_B = thxB * size_t(ldb) + (bly * BLK_N + thyB);

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        sA[thyA][thxA] = conj_if_true < HERK && TRANS == 'C' > (dA[coord_A]);
        sB[thyB][thxB] = conj_if_true < HERK && TRANS == 'N' > (dB[coord_B]);

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_N / DIM_N; ++m)
                    rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();

        if(TRANS == 'N')
            coord_A += BLK_K * size_t(lda);
        if(TRANS == 'T')
            coord_A += BLK_K;
        if(TRANS == 'C')
            coord_A += BLK_K;

        if(TRANS == 'C')
            coord_B += BLK_K;
        if(TRANS == 'T')
            coord_B += BLK_K;
        if(TRANS == 'N')
            coord_B += BLK_K * size_t(ldb);
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_N / DIM_N; ++m)
        {
            int coord_dCm = blx * BLK_N + m * DIM_N + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;
            if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
               || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
            {
                if(BETA_EQ_ZERO)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = alpha * rC[n][m];
                else
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = alpha * rC[n][m] + beta * dC[coord_dCn * size_t(ldc) + coord_dCm];

                // Zero out imaginary part of diagonal if herk
                if(HERK && coord_dCn == coord_dCm)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = std::real(dC[coord_dCn * size_t(ldc) + coord_dCm]);
            }
        }
    }
}

// templated alpha, beta, restricted n, k
template <typename T,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          int  alpha,
          int  beta,
          bool HERK,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
syrkx_herkx_restricted_kernel(rocblas_int    N,
                              rocblas_int    K,
                              TConstPtr*     dA_array,
                              rocblas_int    lda,
                              rocblas_stride stride_a,
                              TConstPtr*     dB_array,
                              rocblas_int    ldb,
                              rocblas_stride stride_b,
                              TPtr*          dC_array,
                              rocblas_int    ldc,
                              rocblas_stride stride_c,
                              rocblas_int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    auto* dA = load_ptr_batch(dA_array, blz, 0, stride_a);
    auto* dB = load_ptr_batch(dB_array, blz, 0, stride_b);
    auto* dC = load_ptr_batch(dC_array, blz, 0, stride_c);

    __shared__ T sA[BLK_K][BLK_N]; // shared memory for A
    __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
    T            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

    size_t coord_A, coord_B;
    if(TRANS == 'N')
        coord_A = (blx * BLK_N + thxA) + thyA * size_t(lda);
    if(TRANS == 'T')
        coord_A = (blx * BLK_N + thxA) * size_t(lda) + thyA;
    if(TRANS == 'C')
        coord_A = (blx * BLK_N + thxA) * size_t(lda) + thyA;
    if(TRANS == 'C')
        coord_B = (bly * BLK_N + thyB) * size_t(ldb) + thxB;
    if(TRANS == 'T')
        coord_B = (bly * BLK_N + thyB) * size_t(ldb) + thxB;
    if(TRANS == 'N')
        coord_B = (bly * BLK_N + thyB) + thxB * size_t(ldb);

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_N / DIM_N; ++m)
            rC[n][m] = 0.0;

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        sA[thyA][thxA] = conj_if_true < HERK && TRANS == 'C' > (dA[coord_A]);
        sB[thyB][thxB] = conj_if_true < HERK && TRANS == 'N' > (dB[coord_B]);

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_N / DIM_N; ++m)
                    rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();

        if(TRANS == 'N')
            coord_A += BLK_K * size_t(lda);
        if(TRANS == 'T')
            coord_A += BLK_K;
        if(TRANS == 'C')
            coord_A += BLK_K;

        if(TRANS == 'C')
            coord_B += BLK_K;
        if(TRANS == 'T')
            coord_B += BLK_K;
        if(TRANS == 'N')
            coord_B += BLK_K * size_t(ldb);
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_N / DIM_N; ++m)
        {
            int coord_dCm = blx * BLK_N + m * DIM_N + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;

            if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
               || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
            {
                if(alpha == 1 && beta == 1)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] += rC[n][m];
                if(alpha == 1 && beta == -1)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = -dC[coord_dCn * size_t(ldc) + coord_dCm] + rC[n][m];
                if(alpha == -1 && beta == 0)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = -rC[n][m];
                if(alpha == 1 && beta == 0)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = rC[n][m];

                // Zero out imaginary part of diagonal if herk
                if(HERK && coord_dCn == coord_dCm)
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = std::real(dC[coord_dCn * size_t(ldc) + coord_dCm]);
            }
        }
    }
}

template <bool HERK, typename T, typename TConstPtr, typename TPtr>
void syrkx_herkx_dispatch(rocblas_fill      uplo,
                          rocblas_operation trans,
                          rocblas_int       n,
                          rocblas_int       k,
                          const T           alpha,
                          TConstPtr*        dA_array,
                          rocblas_int       lda,
                          rocblas_stride    stride_a,
                          TConstPtr*        dB_array,
                          rocblas_int       ldb,
                          rocblas_stride    stride_b,
                          const T           beta,
                          TPtr*             dC_array,
                          rocblas_int       ldc,
                          rocblas_stride    stride_c,
                          rocblas_int       batch_count,
                          hipStream_t       stream)
{
    // syrkx has same behavior for alpha == 0 and k == 0. Special code is needed
    // for alpha == 0, no special code is needed for k == 0. It is more efficient
    // setting k = 0 than adding extra code to a kernel to handle alpha == 0
    if(alpha == 0)
        k = 0;

    // clang-format off
    if((n % 32 == 0) && (k % 8 == 0))
    {
        // Can also be used with:
        // n is mult of 64, k is mult of 4
        // const int dim_n = 16;
        // const int blk_n = 64;
        // const int blk_k = 4;

        // n is mult of 32, k is mult of 8
        const int dim_n = 16;
        const int blk_n = 32;
        const int blk_k = 8;
        dim3      dimBlock(dim_n, dim_n, 1);
        dim3      dimGrid(n / blk_n, n / blk_n, batch_count);
        if(alpha == 1.0 && beta == 1.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(alpha == 1.0 && beta == -1.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(alpha == 1.0 && beta == 0.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(alpha == -1.0 && beta == 0.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                    <T, dim_n, blk_n, blk_k, -1, 0, HERK, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(beta == 0)
        {
            // general alpha; beta == 0
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
    }
    else if(n % 16 == 0 && k % 16 == 0)
    {
        // n is mult of 16
        const int dim = 16;
        dim3      dimBlock(dim, dim, 1);
        dim3      dimGrid(n / dim, n / dim, batch_count);
        if(beta == 0)
        {
            // general n, k, alpha; beta == 0
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, true, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, true, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, true, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, true, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, true, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, true, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general n, k, alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, false, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, false, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, false, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, false, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, false, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_restrict_kernel
                <T, dim, false, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
    }
    else if(n % 16 == 0)
    {
        // can also be used with
        // n is mult of 8
        // const int dim = 8;

        // n is mult of 16
        const int dim = 16;
        dim3      dimBlock(dim, dim, 1);
        dim3      dimGrid(n / dim, n / dim, batch_count);
        if(beta == 0)
        {
            // general n, k, alpha; beta == 0
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, true, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, true, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, true, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, true, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, true, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, true, HERK, 'N', 'U'>),
            dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general n, k, alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, false, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, false, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, false, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, false, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, false, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_small_kernel
                <T, dim, false, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
    }
    else
    {
        const int dim_n = 16;
        const int blk_n = 32;
        const int blk_k = 8;
        dim3      dimBlock(dim_n, dim_n, 1);
        dim3      dimGrid(((n - 1) / blk_n) + 1, ((n - 1) / blk_n) + 1, batch_count);
        if(beta == 0)
        {
            // general n, k, alpha; beta == 0
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general n, k, alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'C', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_conjugate_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'C', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_herkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, HERK, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
    }
    // clang-format on
}

/**
  * kernel
  */
template <bool TWOK, bool HERM, bool trans, rocblas_int TILE_NK, typename T, typename U>
ROCBLAS_KERNEL_ILF void syr2k_her2k_mult_add_device(bool        upper,
                                                    rocblas_int n,
                                                    rocblas_int k,
                                                    U           alpha,
                                                    const T* __restrict__ A,
                                                    rocblas_int lda,
                                                    const T* __restrict__ B,
                                                    rocblas_int ldb,
                                                    T* __restrict__ C,
                                                    rocblas_int ldc)
{
    // if !alpha this function isn't called

    __shared__ T atile[TILE_NK][TILE_NK];
    __shared__ T btile[TILE_NK][TILE_NK];

    int col_pos = blockIdx.y * TILE_NK;
    int row_pos = blockIdx.x * TILE_NK;

    int tilefrom = upper ? row_pos : col_pos;
    int tileto   = upper ? col_pos : row_pos;
    if(tilefrom > tileto)
    {
        // any overlap of tile and output
        return;
    }

    int ab_rows = !trans ? n : k;
    int ab_cols = !trans ? k : n;

    int row = row_pos + threadIdx.x;
    int col = col_pos + threadIdx.y;

    int from = upper ? row : col;
    int to   = upper ? col : row;

    for(int k_pos = 0; k_pos < k; k_pos += TILE_NK)
    {
        // tiling over dimension K

        int row_loc, col_loc;
        int r, c;

        // first matrix mult: alpha*op(A)*op(B)^T
        // when HERM ^H instead of ^T

        // fetch tile of matrix A
        row_loc = row_pos + threadIdx.x;
        col_loc = k_pos + threadIdx.y;
        r       = trans ? col_loc : row_loc; // trans A = A^T, else A = A
        c       = trans ? row_loc : col_loc;

        atile[threadIdx.x][threadIdx.y]
            = (r < ab_rows && c < ab_cols)
                  ? (HERM && trans ? conj(A[c * size_t(lda) + r]) : A[c * size_t(lda) + r])
                  : 0;

        // fetch tile of matrix B
        row_loc = k_pos + threadIdx.x;
        col_loc = col_pos + threadIdx.y;
        r       = trans ? row_loc : col_loc; // trans B = B, else B = B^T
        c       = trans ? col_loc : row_loc;

        btile[threadIdx.x][threadIdx.y]
            = (c < ab_cols && r < ab_rows)
                  ? (HERM && !trans ? conj(B[c * size_t(ldb) + r]) : B[c * size_t(ldb) + r])
                  : 0;

        __syncthreads();

        // n x n symmetric/Hermitian output, tile zero where invalid
        if(row < n && col < n && from <= to)
        {
            T sum = T(0);
            for(int ki = 0; ki < TILE_NK; ++ki)
            {
                sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
            }
            C[col * size_t(ldc) + row] += alpha * sum;
        }

        __syncthreads();

        // second matrix mult: alpha*op(B)*op(A)^T, if HERM conj(alpha) and ^H
        if(TWOK)
        {
            // fetch tile of matrix B  into tileA
            row_loc = row_pos + threadIdx.x;
            col_loc = k_pos + threadIdx.y;
            r       = trans ? col_loc : row_loc; // trans B = B^T, else B = B
            c       = trans ? row_loc : col_loc;

            atile[threadIdx.x][threadIdx.y]
                = (r < ab_rows && c < ab_cols)
                      ? (HERM && trans ? conj(B[c * size_t(ldb) + r]) : B[c * size_t(ldb) + r])
                      : 0;

            // fetch tile of matrix A into tileB
            row_loc = k_pos + threadIdx.x;
            col_loc = col_pos + threadIdx.y;
            r       = trans ? row_loc : col_loc; // trans A = A, else A = A^T
            c       = trans ? col_loc : row_loc;

            btile[threadIdx.x][threadIdx.y]
                = (c < ab_cols && r < ab_rows)
                      ? (HERM && !trans ? conj(A[c * size_t(lda) + r]) : A[c * size_t(lda) + r])
                      : 0;

            __syncthreads();

            // n x n symmetric/Hermitian output, tile zero where invalid
            if(row < n && col < n && from <= to)
            {
                T sum = T(0);
                for(int ki = 0; ki < TILE_NK; ++ki)
                {
                    sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
                }
                C[col * size_t(ldc) + row] += (HERM ? conj(alpha) : alpha) * sum;
            }

            __syncthreads();
        }

    } // k_pos

    if(!TWOK && HERM && row == col && row < n)
    {
        // zero imaginary for cases when A*B aren't true Hermitian
        C[col * size_t(ldc) + row] = std::real(C[col * size_t(ldc) + row]);
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        TWOK,
          bool        HERM,
          bool        TRANS,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_XYT* DIM_XYT)
syr2k_her2k_kernel(bool           upper,
                   rocblas_int    n,
                   rocblas_int    k,
                   TScal          alpha_host_device,
                   TConstPtr      AP_array,
                   rocblas_int    lda,
                   rocblas_stride a_st_or_of,
                   TConstPtr      BP_array,
                   rocblas_int    ldb,
                   rocblas_stride b_st_or_of,
                   TPtr           CP_array,
                   rocblas_int    ldc,
                   rocblas_stride c_st_or_of)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    auto A = load_ptr_batch(AP_array, hipBlockIdx_z, a_st_or_of);
    auto B = load_ptr_batch(BP_array, hipBlockIdx_z, b_st_or_of);
    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, c_st_or_of);

    // compute matrix multiplies and accumulate on the fly into C
    // when HERM does ^H in place of ^T
    syr2k_her2k_mult_add_device<TWOK, HERM, TRANS, DIM_XYT>(
        upper, n, k, alpha, A, lda, B, ldb, C, ldc);
}

template <bool TWOK, bool HERM, rocblas_int DIM_XYT, typename T, typename TConstPtr, typename TPtr>
void syr2k_her2k_dispatch(rocblas_fill      uplo,
                          rocblas_operation trans,
                          rocblas_int       n,
                          rocblas_int       k,
                          const T           alpha,
                          TConstPtr*        dA,
                          rocblas_int       lda,
                          rocblas_stride    stride_a,
                          TConstPtr*        dB,
                          rocblas_int       ldb,
                          rocblas_stride    stride_b,
                          TPtr*             dC,
                          rocblas_int       ldc,
                          rocblas_stride    stride_c,
                          rocblas_int       batch_count,
                          hipStream_t       stream)
{
    rocblas_int bx = (n - 1) / (DIM_XYT) + 1;
    rocblas_int by = (n - 1) / (DIM_XYT) + 1;
    dim3        syr2k_grid(bx, by, batch_count);
    dim3        syr2k_threads(DIM_XYT, DIM_XYT);

    if(trans == rocblas_operation_none)
        hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, HERM, false, DIM_XYT>),
                           syr2k_grid,
                           syr2k_threads,
                           0,
                           stream,
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           alpha,
                           dA,
                           lda,
                           stride_a,
                           dB,
                           ldb,
                           stride_b,
                           dC,
                           ldc,
                           stride_c);
    else
        hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, HERM, true, DIM_XYT>),
                           syr2k_grid,
                           syr2k_threads,
                           0,
                           stream,
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           alpha,
                           dA,
                           lda,
                           stride_a,
                           dB,
                           ldb,
                           stride_b,
                           dC,
                           ldc,
                           stride_c);
}
