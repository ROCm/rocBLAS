#include "definitions.hpp"
#include "handle.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_gemmt.hpp"
#include "utility.hpp"

template <int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          char TRANSA,
          char TRANSB,
          char UPLO,
          bool HERM_A,
          bool HERM_B,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
rocblas_internal_gemmt_kernel(rocblas_int    N,
                              rocblas_int    K,
                              TScal          alpha_device_host,
                              TConstPtr      dA_array,
                              rocblas_int    lda,
                              rocblas_stride stride_a,
                              TConstPtr      dB_array,
                              rocblas_int    ldb,
                              rocblas_stride stride_b,
                              TScal          beta_device_host,
                              TPtr           dC_array,
                              rocblas_int    ldc,
                              rocblas_stride stride_c,
                              rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_device_host);
    auto beta  = load_scalar(beta_device_host);

    if(beta == 1 && (K == 0 || alpha == 0))
        return;

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

    if(alpha != T(0))
    {
        for(int kk = 0; kk < K; kk += BLK_K)
        {
            int i = a_i_offset;
            int j = kk + a_j_offset;
            if(i < N && j < K)
            {
                if(TRANSA == 'N')
                    sA[thyA][thxA] = dA[i + j * size_t(lda)];
                if(TRANSA == 'T')
                    sA[thyA][thxA] = dA[i * size_t(lda) + j];
                if(TRANSA == 'C')
                    sA[thyA][thxA] = conj_if_true<HERM_A>(dA[i * size_t(lda) + j]);
            }
            else
            {
                sA[thyA][thxA] = 0.0;
            }
            i = kk + b_i_offset;
            j = b_j_offset;
            if(i < K && j < N)
            {
                if(TRANSB == 'N')
                    sB[thyB][thxB] = dB[i + j * size_t(ldb)];
                if(TRANSB == 'T')
                    sB[thyB][thxB] = dB[i * size_t(ldb) + j];
                if(TRANSB == 'C')
                    sB[thyB][thxB] = conj_if_true<HERM_B>(dB[i * size_t(ldb) + j]);
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
                if(beta == 0)
                    dC[coord_dCn * size_t(ldc) + coord_dCm] = alpha * rC[n][m];
                else
                    dC[coord_dCn * size_t(ldc) + coord_dCm]
                        = alpha * rC[n][m] + beta * dC[coord_dCn * size_t(ldc) + coord_dCm];
            }
        }
    }
}

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemmt_template(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const TScal*      alpha,
                                               TConstPtr         dA,
                                               rocblas_int       lda,
                                               rocblas_stride    stride_a,
                                               TConstPtr         dB,
                                               rocblas_int       ldb,
                                               rocblas_stride    stride_b,
                                               const TScal*      beta,
                                               TPtr              dC,
                                               rocblas_int       ldc,
                                               rocblas_stride    stride_c,
                                               rocblas_int       batch_count)
{
    hipStream_t stream = handle->get_stream();

    constexpr bool is_complex
        = std::is_same_v<TScal,
                         rocblas_float_complex> || std::is_same_v<TScal, rocblas_double_complex>;

    const int dim_n = 16;
    const int blk_n = 32;
    const int blk_k = 8;
    dim3      dimBlock(dim_n, dim_n);
    dim3      dimGrid(((n - 1) / blk_n) + 1, ((n - 1) / blk_n) + 1, batch_count);

#define ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha_, beta_)                                             \
    dimGrid, dimBlock, 0, stream, n, k, alpha_, dA, lda, stride_a, dB, ldb, stride_b, beta_, dC, \
        ldc, stride_c, batch_count
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(uplo == rocblas_fill_upper)
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'N',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'T',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'C',
                                                                  'U',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'N',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'T',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'C',
                                                                  'U',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'N',
                                                                  'U',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'T',
                                                                  'U',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'C',
                                                                  'U',
                                                                  is_complex,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
        }
        else
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'N',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'T',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'C',
                                                                  'L',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'N',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'T',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'C',
                                                                  'L',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'N',
                                                                  'L',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'T',
                                                                  'L',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'C',
                                                                  'L',
                                                                  is_complex,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(alpha, beta));
        }
    }
    //pointer mode host
    else
    {
        if(uplo == rocblas_fill_upper)
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'N',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'T',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'C',
                                                                  'U',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'N',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'T',
                                                                  'U',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'C',
                                                                  'U',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'N',
                                                                  'U',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'T',
                                                                  'U',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'C',
                                                                  'U',
                                                                  is_complex,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
        }
        else
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'N',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'T',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'N',
                                                                  'C',
                                                                  'L',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'N',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'T',
                                                                  'L',
                                                                  false,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'T',
                                                                  'C',
                                                                  'L',
                                                                  false,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'N',
                                                                  'L',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'T',
                                                                  'L',
                                                                  is_complex,
                                                                  false,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_internal_gemmt_kernel<dim_n,
                                                                  blk_n,
                                                                  blk_k,
                                                                  'C',
                                                                  'C',
                                                                  'L',
                                                                  is_complex,
                                                                  is_complex,
                                                                  TScal>),
                                   ROCBLAS_INTERNAL_GEMMT_PARAMS(*alpha, *beta));
        }
    }
#undef ROCBLAS_INTERNAL_GEMMT_PARAMS

    return rocblas_status_success;
}

#ifdef INSTANTIATE_GEMMT_TEMPLATE
#error INSTANTIATE_GEMMT_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMMT_TEMPLATE(TScal_, TConstPtr_, TPtr_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                   \
        rocblas_internal_gemmt_template<TScal_, TConstPtr_, TPtr_>(rocblas_handle    handle,   \
                                                                   rocblas_fill      uplo,     \
                                                                   rocblas_operation transA,   \
                                                                   rocblas_operation transB,   \
                                                                   rocblas_int       n,        \
                                                                   rocblas_int       k,        \
                                                                   const TScal_*     alpha,    \
                                                                   TConstPtr_        dA_in,    \
                                                                   rocblas_int       lda,      \
                                                                   rocblas_stride    stride_a, \
                                                                   TConstPtr_        dB_in,    \
                                                                   rocblas_int       ldb,      \
                                                                   rocblas_stride    stride_b, \
                                                                   const TScal_*     beta,     \
                                                                   TPtr_             dC_in,    \
                                                                   rocblas_int       ldc,      \
                                                                   rocblas_stride    stride_c, \
                                                                   rocblas_int       batch_count);

INSTANTIATE_GEMMT_TEMPLATE(float, const float*, float*)
INSTANTIATE_GEMMT_TEMPLATE(double, const double*, double*)
INSTANTIATE_GEMMT_TEMPLATE(rocblas_float_complex,
                           const rocblas_float_complex*,
                           rocblas_float_complex*)
INSTANTIATE_GEMMT_TEMPLATE(rocblas_double_complex,
                           const rocblas_double_complex*,
                           rocblas_double_complex*)
INSTANTIATE_GEMMT_TEMPLATE(float, const float* const*, float* const*)
INSTANTIATE_GEMMT_TEMPLATE(double, const double* const*, double* const*)
INSTANTIATE_GEMMT_TEMPLATE(rocblas_float_complex,
                           const rocblas_float_complex* const*,
                           rocblas_float_complex* const*)
INSTANTIATE_GEMMT_TEMPLATE(rocblas_double_complex,
                           const rocblas_double_complex* const*,
                           rocblas_double_complex* const*)

#undef INSTANTIATE_GEMMT_TEMPLATE
