/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "Tensile/gemm.hpp"
#include "definitions.hpp"
#include "rocblas_syrkx.hpp"

template <typename T,
          int  DIM,
          bool BETA_EQ_ZERO,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM* DIM) void syrkx_small_kernel(rocblas_int    N,
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
        }
        else
        {
            sA[thy][thx] = 0.0;
        }

        i3_b = kk + thx;
        if(i2 < N && i3_b < K)
        {
            if(TRANS == 'T')
                sB[thy][thx] = dB[i3_b + i2 * size_t(ldb)];
            if(TRANS == 'N')
                sB[thy][thx] = dB[i2 + i3_b * size_t(ldb)];
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
    }
}

// N and K must be multiples of DIM
template <typename T,
          int  DIM,
          bool BETA_EQ_ZERO,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM* DIM) void syrkx_small_restrict_kernel(rocblas_int    N,
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
                                                                            rocblas_int batch_count)
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

        i3_b = kk + thx;
        if(TRANS == 'T')
            sB[thy][thx] = dB[i3_b + i2 * size_t(ldb)];
        if(TRANS == 'N')
            sB[thy][thx] = dB[i2 + i3_b * size_t(ldb)];

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
    }
}

// general alpha, beta, m, n, k
template <typename T,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          bool BETA_EQ_ZERO,
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_N* DIM_N) void syrkx_general_kernel(rocblas_int    N,
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
        }
        else
        {
            sA[thyA][thxA] = 0.0;
        }
        i = kk + b_i_offset;
        j = b_j_offset;
        if(i < K && j < N)
        {
            if(TRANS == 'T')
                sB[thyB][thxB] = dB[i + j * size_t(ldb)];
            if(TRANS == 'N')
                sB[thyB][thxB] = dB[i * size_t(ldb) + j];
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
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_N* DIM_N) void syrkx_restricted_kernel(rocblas_int    N,
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
                                                                            rocblas_int batch_count)
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

    if(TRANS == 'T')
        coord_B = thxB + (bly * BLK_N + thyB) * size_t(ldb);
    if(TRANS == 'N')
        coord_B = thxB * size_t(ldb) + (bly * BLK_N + thyB);

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        sA[thyA][thxA] = dA[coord_A];
        sB[thyB][thxB] = dB[coord_B];

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
          char TRANS,
          char UPLO,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_N* DIM_N) void syrkx_restricted_kernel(rocblas_int    N,
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
                                                                            rocblas_int batch_count)
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
        sA[thyA][thxA] = dA[coord_A];
        sB[thyB][thxB] = dB[coord_B];

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
            }
        }
    }
}

template <typename T, typename TConstPtr, typename TPtr>
void syrkx_dispatch(rocblas_fill      uplo,
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
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 1, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(alpha == 1.0 && beta == -1.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, -1, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(alpha == 1.0 && beta == 0.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, 1, 0, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(alpha == -1.0 && beta == 0.0)
        {
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, -1, 0, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                    <T, dim_n, blk_n, blk_k, -1, 0, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, dA_array, lda, stride_a, dB_array, ldb, stride_b, dC_array, ldc, stride_c, batch_count);
        }
        else if(beta == 0)
        {
            // general alpha; beta == 0
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, true, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_restricted_kernel
                <T, dim_n, blk_n, blk_k, false, 'N', 'U'>),
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
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, true, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, true,'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, true, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, true,'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general n, k, alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, false, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, false, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, false, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_restrict_kernel
                <T, dim, false, 'N', 'U'>),
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
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, true, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, true,'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, true, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, true,'N', 'U'>),
            dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general n, k, alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, false, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, false, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, false, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_small_kernel
                <T, dim, false, 'N', 'U'>),
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
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, true,'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, true, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, true,'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
        else
        {
            // general n, k, alpha, beta
            if((rocblas_operation_transpose == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, 'T', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_lower == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, 'N', 'L'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_transpose == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, 'T', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
            else if((rocblas_operation_none == trans) && (rocblas_fill_upper == uplo))
                hipLaunchKernelGGL((syrkx_general_kernel
                <T, dim_n, blk_n, blk_k, false, 'N', 'U'>),
                dimGrid, dimBlock, 0, stream, n, k, alpha, dA_array, lda, stride_a, dB_array, ldb, stride_b, beta, dC_array, ldc, stride_c, batch_count);
        }
    }
    // clang-format on
}

#define OFFSET_A(i1) offset_a + i1* rocblas_stride(a_s1)
#define OFFSET_B(i1) offset_b + i1* rocblas_stride(b_s1)
#define OFFSET_C(i1, i2) offset_c + i1* rocblas_stride(c_s1) + i2* rocblas_stride(c_s2)

template <int MIN_NB, bool BATCHED, typename T, typename TScal, typename TPtr, typename TConstPtr>
rocblas_status rocblas_syrkx_template(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      TScal*            alpha,
                                      TConstPtr*        da,
                                      rocblas_int       lda,
                                      TConstPtr*        db,
                                      rocblas_int       ldb,
                                      TScal*            beta,
                                      TPtr*             dc,
                                      rocblas_int       ldc)
{
    static constexpr rocblas_stride offset_c = 0, offset_a = 0, offset_b = 0;
    static constexpr rocblas_int    batch_count = 1;
    static constexpr rocblas_stride stride_c = 0, stride_a = 0, stride_b = 0;

    rocblas_stride a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_stride b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_stride c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    hipStream_t stream = handle->get_stream();

    // call syrkx_dispatch with batch_count = n_nb for n_nb diagonal blocks
    // clang-format off
    syrkx_dispatch<T>( uplo, trans, nb, k, *alpha,
                       da, lda, nb * a_s1,
                       db, ldb, nb * b_s1, *beta,
                       dc, ldc, nb * (c_s1 + c_s2), n_nb, stream);
    // clang-format on

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;
        // call syrkx_dispatch for one remainder diagonal block of size n_diag
        // clang-format off
        syrkx_dispatch<T>( uplo, trans, n_diag, k, *alpha,
                          &(da[i_diag * a_s1]),          lda, stride_a,
                          &(db[i_diag * b_s1]),          ldb, stride_b, *beta,
                          &(dc[i_diag * (c_s1 + c_s2)]), ldc, stride_c, batch_count, stream);
        // clang-format on
    }

    rocblas_operation trans_a
        = rocblas_operation_none == trans ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation trans_b
        = rocblas_operation_none == trans ? rocblas_operation_transpose : rocblas_operation_none;

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, then for each iteration of nb,i_start loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }

        // call gemm with batch_count = n_nb for n_nb square blocks of size nb x nb
        if(rocblas_fill_lower == uplo)
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                 handle, trans_a, trans_b, nb, nb, k, alpha,
                 da, OFFSET_A(i_start),    lda, stride * a_s1,
                 db, OFFSET_B(0),          ldb, stride * b_s1,          beta,
                 dc, OFFSET_C(i_start, 0), ldc, stride * (c_s1 + c_s2), n_nb   )));
            // clang-format on
        }
        else
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                 handle, trans_a, trans_b, nb, nb, k, alpha,
                 da, OFFSET_A(0),          lda, stride * a_s1,
                 db, OFFSET_B(i_start),    ldb, stride * b_s1,          beta,
                 dc, OFFSET_C(0, i_start), ldc, stride * (c_s1 + c_s2), n_nb)));
            // clang-format on
        }

        // call gemm for remainder block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_stride i1 = i_start + n_nb * stride;
            rocblas_stride i2 = i1 - nb;
            rocblas_stride n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, n1, nb, k, alpha,
                     da, OFFSET_A(i1),     lda, stride_a,
                     db, OFFSET_B(i2),     ldb, stride_b, beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, n1, k, alpha,
                     da, OFFSET_A(i2),     lda, stride_a,
                     db, OFFSET_B(i1),     ldb, stride_b, beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}

template <int MIN_NB, bool BATCHED, typename T, typename TScal, typename TPtr, typename TConstPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal*            alpha,
                                    TConstPtr*        da,
                                    rocblas_stride    offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    TConstPtr*        db,
                                    rocblas_stride    offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    TScal*            beta,
                                    TPtr*             dc,
                                    rocblas_stride    offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count)
{
    static constexpr bool TWOK = false;

    if(BATCHED == false && batch_count == 1)
    {
        return rocblas_syrkx_template<MIN_NB, BATCHED, T>(handle,
                                                          uplo,
                                                          trans,
                                                          n,
                                                          k,
                                                          alpha,
                                                          &(da[offset_a]),
                                                          lda,
                                                          &(db[offset_b]),
                                                          ldb,
                                                          beta,
                                                          &(dc[offset_c]),
                                                          ldc);
    }

    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_int c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    // n_nb diagonal blocks of size nb
    for(int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb; // diag block at c[i_diag, i_diag], size is nb
        // clang-format off
        rocblas_internal_syr2k_template<BATCHED, TWOK>(
              handle, uplo, trans, nb, k, alpha,
              da, OFFSET_A(i_diag),         lda, stride_a,
              db, OFFSET_B(i_diag),         ldb, stride_b, beta,
              dc, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;
        // clang-format off
        rocblas_internal_syr2k_template<BATCHED, TWOK>(
              handle, uplo, trans, n_diag, k, alpha,
              da, OFFSET_A(i_diag),         lda, stride_a,
              db, OFFSET_B(i_diag),         ldb, stride_b, beta,
              dc, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    rocblas_operation trans_a
        = rocblas_operation_none == trans ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation trans_b
        = rocblas_operation_none == trans ? rocblas_operation_transpose : rocblas_operation_none;

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, and each iteration of the outer loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }
        // n_nb gemm blocks of size nb x nb
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, nb, k, alpha,
                     da, OFFSET_A(i1),     lda, stride_a,
                     db, OFFSET_B(i2),     ldb, stride_b, beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, nb, k, alpha,
                     da, OFFSET_A(i2),     lda, stride_a,
                     db, OFFSET_B(i1),     ldb, stride_b, beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }

        // remainder gemm block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, n1, nb, k, alpha,
                     da, OFFSET_A(i1),     lda, stride_a,
                     db, OFFSET_B(i2),     ldb, stride_b, beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, n1, k, alpha,
                     da, OFFSET_A(i2),     lda, stride_a,
                     db, OFFSET_B(i1),     ldb, stride_b, beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}
#undef OFFSET_A
#undef OFFSET_B
#undef OFFSET_C

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrkx*.cpp

// clang-format off
#ifdef INSTANTIATE_SYRKX_TEMPLATE
#error INSTANTIATE_SYRKX_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRKX_TEMPLATE(MIN_NB_, BATCHED_, T_, TScal_, TPtr_, TConstPtr_)        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syrkx_template    \
                                   <MIN_NB_, BATCHED_, T_, TScal_, TPtr_, TConstPtr_>       \
                                   (rocblas_handle    handle,                               \
                                    rocblas_fill      uplo,                                 \
                                    rocblas_operation trans,                                \
                                    rocblas_int       n,                                    \
                                    rocblas_int       k,                                    \
                                    TScal_ *          alpha,                                \
                                    TConstPtr_ *      da,                                   \
                                    rocblas_stride    offset_a,                             \
                                    rocblas_int       lda,                                  \
                                    rocblas_stride    stride_a,                             \
                                    TConstPtr_ *      db,                                   \
                                    rocblas_stride    offset_b,                             \
                                    rocblas_int       ldb,                                  \
                                    rocblas_stride    stride_b,                             \
                                    TScal_ *          beta,                                 \
                                    TPtr_ *           dc,                                   \
                                    rocblas_stride    offset_c,                             \
                                    rocblas_int       ldc,                                  \
                                    rocblas_stride    stride_c,                             \
                                    rocblas_int       batch_count);

// instantiate for rocblas_Xsyrkx and rocblas_Xsyrkx_strided_batched
INSTANTIATE_SYRKX_TEMPLATE(16, false,  float,  float const,  float,  float const)
INSTANTIATE_SYRKX_TEMPLATE(32, false, double, double const, double, double const)
INSTANTIATE_SYRKX_TEMPLATE(16, false, double, double const, double, double const)
INSTANTIATE_SYRKX_TEMPLATE( 8, false,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)
INSTANTIATE_SYRKX_TEMPLATE(32, false,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)
INSTANTIATE_SYRKX_TEMPLATE( 8, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex, rocblas_double_complex const)
INSTANTIATE_SYRKX_TEMPLATE(32, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex, rocblas_double_complex const)

// instantiate for rocblas_Xsyrkx_batched
INSTANTIATE_SYRKX_TEMPLATE(16,  true,  float,  float const,  float* const,  float const* const)
INSTANTIATE_SYRKX_TEMPLATE(16,  true, double, double const, double* const, double const* const)
INSTANTIATE_SYRKX_TEMPLATE( 8,  true,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex* const,  rocblas_float_complex const* const)
INSTANTIATE_SYRKX_TEMPLATE( 8,  true, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex* const, rocblas_double_complex const* const)

#undef INSTANTIATE_SYRKX_TEMPLATE
// clang-format on
