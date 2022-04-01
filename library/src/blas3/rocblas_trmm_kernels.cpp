/* ************************************************************************
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "Tensile/gemm.hpp"
#include "definitions.hpp"
#include "rocblas_trmm.hpp"

//-- Innovative Computing Laboratory
//  -- Electrical Engineering and Computer Science Department
//  -- University of Tennessee
//  -- (C) Copyright 2009-2020
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the University of Tennessee, Knoxville nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors
//  ``as is'' and any express or implied warranties, including, but not
//  limited to, the implied warranties of merchantability and fitness for
//  a particular purpose are disclaimed. In no event shall the copyright
//  holders or contributors be liable for any direct, indirect, incidental,
//  special, exemplary, or consequential damages (including, but not
//  limited to, procurement of substitute goods or services; loss of use,
//  data, or profits; or business interruption) however caused and on any
//  theory of liability, whether in contract, strict liability, or tort
//  (including negligence or otherwise) arising in any way out of the use
//  of this software, even if advised of the possibility of such damage.

rocblas_int rocblas_get_trmm_recursive_nb(rocblas_int n);

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                     rocblas_int    n,
                                     TScal          alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     TPtr           Aa,
                                     rocblas_int    lda,
                                     rocblas_stride a_st_or_of)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);

    if(tx < m && ty < n && alpha == 0)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, a_st_or_of);

        A[tx + size_t(lda) * ty] = 0;
    }
}

template <typename TScal, typename TPtr>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TPtr           A,
                                                      rocblas_int    lda,
                                                      rocblas_stride a_st_or_of,
                                                      rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    static constexpr int GEMV_DIM_X = 16;
    static constexpr int GEMV_DIM_Y = 16;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 grid(blocksX, blocksY, batch_count);
    dim3 threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL((set_matrix_zero_if_alpha_zero_kernel<GEMV_DIM_X, GEMV_DIM_Y>),
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           alpha,
                           stride_alpha,
                           A,
                           lda,
                           a_st_or_of);
    else
        hipLaunchKernelGGL((set_matrix_zero_if_alpha_zero_kernel<GEMV_DIM_X, GEMV_DIM_Y>),
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           *alpha,
                           stride_alpha,
                           A,
                           lda,
                           a_st_or_of);
    return rocblas_status_success;
}

// left, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_lNx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        rocblas_int      lda,
                        rocblas_stride   a_st_or_of,
                        TPtr*            B_arg,
                        rocblas_int      ldb,
                        rocblas_stride   b_st_or_of)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, a_st_or_of);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, b_st_or_of);

    const int nblocks = (n + NB - 1) / NB;
    const int nn      = (bx < nblocks - 1) ? NB : n - (nblocks - 1) * NB;
    B += bx * NB * size_t(ldb);

    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    // initialize sA and sB to zero
    sA[ty * NB + tx] = 0;
    sB[ty * NB + tx] = 0;

    // load A and B
    if(ty < m && tx < m)
        sA[ty * NB + tx] = A[ty * size_t(lda) + tx];
    if(ty < nn && tx < m)
        sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

    // handle diag
    if(diag == rocblas_diagonal_unit)
    {
        if(ty == tx)
            sA[ty * NB + tx] = 1.0;
    }

    // handle uplo
    if(uplo == rocblas_fill_upper)
    {
        if(tx > ty)
            sA[ty * NB + tx] = 0.0;
    }
    else
    {
        if(tx < ty)
            sA[ty * NB + tx] = 0.0;
    }
    __syncthreads();

    T accumulator = 0;
#pragma unroll
    for(int i = 0; i < NB; i++)
        accumulator += sA[i * NB + tx] * sB[ty * NB + i];
    accumulator *= alpha;
    if(ty < nn && tx < m)
        B[ty * size_t(ldb) + tx] = accumulator;
}

// left, Trans|ConjTrans
template <const int NB, bool CONJA, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_lTx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        rocblas_int      lda,
                        rocblas_stride   a_st_or_of,
                        TPtr*            B_arg,
                        rocblas_int      ldb,
                        rocblas_stride   b_st_or_of)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, a_st_or_of);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, b_st_or_of);

    const int nblocks = (n + NB - 1) / NB;
    const int nn      = (bx < nblocks - 1) ? NB : n - (nblocks - 1) * NB;
    B += bx * NB * size_t(ldb);

    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    // init sA and sB to zero
    sA[ty * NB + tx] = 0.0;
    sB[ty * NB + tx] = 0.0;
    __syncthreads(); // needed because sA will be stored as transposed

    // load A and B
    if(ty < m && tx < m)
    {
        if(CONJA)
        {
            sA[tx * NB + ty] = conj(A[ty * size_t(lda) + tx]);
        }
        else
        {
            sA[tx * NB + ty] = A[ty * size_t(lda) + tx];
        }
    }
    if(ty < nn && tx < m)
        sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

    // handle diag
    if(diag == rocblas_diagonal_unit)
    {
        if(ty == tx)
            sA[ty * NB + tx] = 1.0;
    }

    // handle uplo
    __syncthreads();
    if(uplo == rocblas_fill_lower)
    {
        if(tx > ty)
            sA[ty * NB + tx] = 0.0;
    }
    else
    {
        if(tx < ty)
            sA[ty * NB + tx] = 0.0;
    }
    __syncthreads();

    T accumulator = 0.0;
#pragma unroll
    for(int i = 0; i < NB; i++)
        accumulator += sA[i * NB + tx] * sB[ty * NB + i];
    accumulator *= alpha;

    // write B
    if(ty < nn && tx < m)
        B[ty * size_t(ldb) + tx] = accumulator;
}

// right NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_rNx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        rocblas_int      lda,
                        rocblas_stride   a_st_or_of,
                        TPtr*            B_arg,
                        rocblas_int      ldb,
                        rocblas_stride   b_st_or_of)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, a_st_or_of);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, b_st_or_of);

    const int nblocks = (m + NB - 1) / NB;
    const int mm      = (bx < nblocks - 1) ? NB : m - (nblocks - 1) * NB;
    B += bx * NB;

    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    // init sA and sB to zero
    sA[ty * NB + tx] = 0.0;
    sB[ty * NB + tx] = 0.0;

    // load A and B
    if(ty < n && tx < n)
        sA[ty * NB + tx] = A[ty * size_t(lda) + tx];
    if(ty < n && tx < mm)
        sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

    // handle diag
    if(diag == rocblas_diagonal_unit)
    {
        if(ty == tx)
            sA[ty * NB + tx] = 1.0;
    }

    // handle uplo
    if(uplo == rocblas_fill_upper)
    {
        if(tx > ty)
            sA[ty * NB + tx] = 0.0;
    }
    else
    {
        if(tx < ty)
            sA[ty * NB + tx] = 0.0;
    }
    __syncthreads();

    T accumulator = 0.0;
#pragma unroll
    for(int i = 0; i < NB; i++)
        accumulator += sB[i * NB + tx] * sA[ty * NB + i];
    accumulator *= alpha;
    // write B
    if(ty < n && tx < mm)
        B[ty * size_t(ldb) + tx] = accumulator;
}

// right, transpose_and_conjugate_transpose
template <const int NB, bool CONJA, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_rTx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        rocblas_int      lda,
                        rocblas_stride   a_st_or_of,
                        TPtr*            B_arg,
                        rocblas_int      ldb,
                        rocblas_stride   b_st_or_of)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, a_st_or_of);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, b_st_or_of);

    const int nblocks = (m + NB - 1) / NB;
    const int mm      = (bx < nblocks - 1) ? NB : m - (nblocks - 1) * NB;
    B += bx * NB;

    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    // init sA and sB to zero
    sA[ty * NB + tx] = 0.0;
    sB[ty * NB + tx] = 0.0;

    // load A and B
    if(ty < n && tx < n)
    {
        if(CONJA)
        {
            sA[ty * NB + tx] = conj(A[ty * size_t(lda) + tx]);
        }
        else
        {
            sA[ty * NB + tx] = A[ty * size_t(lda) + tx];
        }
    }
    if(ty < n && tx < mm)
        sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

    // handle diag
    if(diag == rocblas_diagonal_unit)
    {
        if(ty == tx)
            sA[ty * NB + tx] = 1.0;
    }

    // handle uplo
    if(uplo == rocblas_fill_upper)
    {
        if(tx > ty)
            sA[ty * NB + tx] = 0.0;
    }
    else
    {
        if(tx < ty)
            sA[ty * NB + tx] = 0.0;
    }
    __syncthreads();

    T accumulator = 0.0;
#pragma unroll
    for(int i = 0; i < NB; i++)
        accumulator += sB[i * NB + tx] * sA[i * NB + ty];
    accumulator *= alpha;
    // write B
    if(ty < n && tx < mm)
        B[ty * size_t(ldb) + tx] = accumulator;
}

// clang-format off
// left, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status trmm_template_lNx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, rocblas_int lda, rocblas_stride a_st_or_of,
                       TPtr*            dB, rocblas_int ldb, rocblas_stride b_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((n + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_lNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);
    else
        hipLaunchKernelGGL((rocblas_trmm_lNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);

    return rocblas_status_success;
}

// left, Trans|ConjTrans
template <const int NB, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status trmm_template_lTx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, rocblas_int lda, rocblas_stride a_st_or_of,
                       TPtr*            dB, rocblas_int ldb, rocblas_stride b_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((n + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_lTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);
    else
        hipLaunchKernelGGL((rocblas_trmm_lTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);

    return rocblas_status_success;
}

// right, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status trmm_template_rNx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, rocblas_int lda, rocblas_stride a_st_or_of,
                       TPtr*            dB, rocblas_int ldb, rocblas_stride b_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((m + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_rNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);
    else
        hipLaunchKernelGGL((rocblas_trmm_rNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);

    return rocblas_status_success;
}

// right, Trans|ConjTrans
template <const int NB, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status trmm_template_rTx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, rocblas_int lda, rocblas_stride a_st_or_of,
                       TPtr*            dB, rocblas_int ldb, rocblas_stride b_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((m + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_rTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);
    else
        hipLaunchKernelGGL((rocblas_trmm_rTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of);

    return rocblas_status_success;
}

template <bool BATCHED, int STOPPING_NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_trmm_small(rocblas_handle    handle,
                        rocblas_side      side,
                        rocblas_fill      uplo,
                        rocblas_operation trans_a,
                        rocblas_diagonal  diag,
                        rocblas_int       m,
                        rocblas_int       n,
                        TScal             alpha,
                        rocblas_stride    stride_alpha,
                        TConstPtr         dA, rocblas_stride offset_a, rocblas_int lda, rocblas_stride stride_a,
                        TPtr              dB, rocblas_stride offset_b, rocblas_int ldb, rocblas_stride stride_b,
                        rocblas_int       batch_count)
{
    TConstPtr      dA_krn;
    TPtr           dB_krn;
    rocblas_stride a_st_or_of;
    rocblas_stride b_st_or_of;

    if(BATCHED)
    {
        dA_krn     = dA;
        dB_krn     = dB;
        a_st_or_of = offset_a;
        b_st_or_of = offset_b;
    }
    else
    {
        dB_krn     = dB + offset_b;
        dA_krn     = dA + offset_a;
        a_st_or_of = stride_a;
        b_st_or_of = stride_b;
    }

    rocblas_int shape = -1;
    if(side == rocblas_side_left)
    {
        if     (trans_a == rocblas_operation_none)                shape = 0;      // lNx     left, NoTrans
        else if(trans_a == rocblas_operation_transpose)           shape = 1;      // lTx     left, Transpose
        else if(trans_a == rocblas_operation_conjugate_transpose) shape = 2;      // lCx     left, ConjTrans
    }
    else
    {
        if     (trans_a == rocblas_operation_none)                shape = 3;      // rNx     right, NoTrans
        else if(trans_a == rocblas_operation_transpose)           shape = 4;      // rTx     right, Transpose }
        else if(trans_a == rocblas_operation_conjugate_transpose) shape = 5;      // rCx     right, ConjTrans
    }

    if (shape == 0) // lNx, left, NoTrans
        return trmm_template_lNx<STOPPING_NB, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA_krn, lda, a_st_or_of,
                                               dB_krn, ldb, b_st_or_of, batch_count);
    else if (shape == 1) // lTx, left, Transpose
        return trmm_template_lTx<STOPPING_NB, false, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA_krn, lda, a_st_or_of,
                                               dB_krn, ldb, b_st_or_of, batch_count);
    else if (shape == 2) // lCx, left, ConjTrans
        return trmm_template_lTx<STOPPING_NB, true, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA_krn, lda, a_st_or_of,
                                               dB_krn, ldb, b_st_or_of, batch_count);
    else if (shape == 3) // rNx, right, NoTrans
        return trmm_template_rNx<STOPPING_NB, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA_krn, lda, a_st_or_of,
                                               dB_krn, ldb, b_st_or_of, batch_count);
    else if (shape == 4) // rTx, right, Transpose
        return trmm_template_rTx<STOPPING_NB, false, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA_krn, lda, a_st_or_of,
                                               dB_krn, ldb, b_st_or_of, batch_count);
    else if (shape == 5) // rCx, right, ConjTrans
        return trmm_template_rTx<STOPPING_NB, true, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA_krn, lda, a_st_or_of,
                                               dB_krn, ldb, b_st_or_of, batch_count);
    else
        return rocblas_status_internal_error;
}

rocblas_int inline trmm_get_shape(rocblas_side side, rocblas_fill uplo, rocblas_operation trans_a)
{
    rocblas_int shape = -1;
    if(side == rocblas_side_left)
    {
        if(trans_a == rocblas_operation_none)
        {
            if (uplo == rocblas_fill_lower) shape = 0; else shape = 1; // lNL, lNU
        }
        else
        {
            if (uplo == rocblas_fill_lower) shape = 2; else shape = 3; // lTL | lCL, lTU | lCU
        }
    }
    else
    {
        if(trans_a == rocblas_operation_none)
        {
            if (uplo == rocblas_fill_lower) shape = 4; else shape = 5; // rNL, rNU
        }
        else
        {
            if (uplo == rocblas_fill_lower) shape = 6; else shape = 7; // rTL | rCL, rTU | rCU
        }
    }

    return shape;
}

// right, transpose_and_conjugate_transpose
template <typename T,
          const int NB,
          const int THR_DIM,
          bool LEFT,
          bool UPPER,
          bool TRANSPOSE,
          bool CONJ,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(NB* NB) rocblas_trmm_outofplace_kernel(rocblas_diagonal diag,
                                                                             int              m,
                                                                             int              n,
                                                                             TScal            alpha_device_host,
                                                                             rocblas_stride   stride_alpha,
                                                                             TConstPtr*       A_arg,
                                                                             rocblas_stride   offset_a,
                                                                             rocblas_int      lda,
                                                                             rocblas_stride   stride_a,
                                                                             TConstPtr*       B_arg,
                                                                             rocblas_stride   offset_b,
                                                                             rocblas_int      ldb,
                                                                             rocblas_stride   stride_b,
                                                                             TPtr*            C_arg,
                                                                             rocblas_stride   offset_c,
                                                                             rocblas_int      ldc,
                                                                             rocblas_stride   stride_c,
                                                                             rocblas_int      batch_count)
{
    constexpr bool ITER_UPPER = (UPPER && !TRANSPOSE) || (!UPPER && TRANSPOSE);
    constexpr rocblas_int DIM = NB / THR_DIM;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    auto A = load_ptr_batch(A_arg, hipBlockIdx_z, offset_a, stride_a);
    auto B = load_ptr_batch(B_arg, hipBlockIdx_z, offset_b, stride_b);
    auto C = load_ptr_batch(C_arg, hipBlockIdx_z, offset_c, stride_c);

    if(alpha == 0)
        return;

    const rocblas_int k = LEFT ? m : n;

    const rocblas_int tx = threadIdx.x;
    const rocblas_int ty = threadIdx.y;
    const rocblas_int bx = blockIdx.x;
    const rocblas_int by = blockIdx.y;

    __shared__ T sA[NB][NB];
    __shared__ T sB[NB][NB];

    T rC[THR_DIM][THR_DIM];

    // A and B offset by blocks and threads
    // For LEFT,  bx is block row of A. For upper triangular, we need to offset the column
    //            as well since the first (lower) portion doesn't need to be accessed.
    // For RIGHT, by is block column of A. For lower triangular, we need to offset the row
    //            as well since the first (upper) portion doesn't need to be accessed.
    //
    rocblas_int A_col_offset = LEFT ? (ITER_UPPER ? bx * NB + ty : ty)
                                    : (ITER_UPPER ? by * NB + ty : ty + NB * by);
    rocblas_int A_row_offset = LEFT ? bx * NB + tx
                                    : (ITER_UPPER ? tx : by * NB + tx);
    rocblas_int B_row_offset = LEFT ? (ITER_UPPER ? NB * bx + tx : tx)
                                    : bx * NB + tx;
    rocblas_int B_col_offset = LEFT ? by * NB + ty
                                    : (ITER_UPPER ? ty : ty + NB * by);

    const T* dA = A + (TRANSPOSE ? A_col_offset + A_row_offset * size_t(lda) : A_row_offset + A_col_offset * size_t(lda));
    const T* dB = B + B_row_offset + B_col_offset * size_t(ldb);

    // zero out result matrix
    for(rocblas_int i = 0; i < THR_DIM; i++)
    {
        for(rocblas_int j = 0; j < THR_DIM; j++)
        {
            rC[i][j] = 0.0;
        }
    }

    // full blocks of A. bx is the block row which is equal to
    // the number of full blocks in that block row (for lower triangular)
    const rocblas_int full_blocks = LEFT ? NB * bx : NB * by;

    // Iterate through the blocks. If we iterate up the triangular matrix on the left, we use the inverse of what is calculated above,
    // otherwise we add a BLK for the triangular block.
    rocblas_int block_iter_end = ((ITER_UPPER && LEFT) || (!ITER_UPPER && !LEFT)) ? k - full_blocks : full_blocks + NB;
    for(rocblas_int blk_iter = 0; blk_iter < block_iter_end; blk_iter += NB)
    {
        // store A in shared memory
        for(rocblas_int i = 0; i < NB; i += DIM)
        {
            for(rocblas_int j = 0; j < NB; j += DIM)
            {
                // Check if the A index is within the bounds of the matrix, is on a diagonal, and is within the triangular section.
                size_t A_idx = TRANSPOSE ? j * size_t(lda) + i : i * size_t(lda) + j;
                bool in_diag = diag == rocblas_diagonal_unit && j + A_row_offset == i + A_col_offset;
                bool in_size = j + A_row_offset < k && i + A_col_offset < k;
                bool in_bounds = in_size && (UPPER ? (TRANSPOSE ? (j + A_row_offset >= i + A_col_offset)
                                                                : (j + A_row_offset <= i + A_col_offset))
                                                   : (TRANSPOSE ? (j + A_row_offset <= i + A_col_offset)
                                                                : (j + A_row_offset >= i + A_col_offset)));

                if(in_bounds && !in_diag)
                    sA[i + ty][j + tx] = CONJ ? conj(dA[A_idx]) : dA[A_idx];
                else if(in_diag)
                    sA[i + ty][j + tx] = 1;
                else
                    sA[i + ty][j + tx] = 0;
            }
        }

        // store B in shared memory
        for(rocblas_int i = 0; i < NB; i += DIM)
        {
            for(rocblas_int j = 0; j < NB; j += DIM)
            {
                if(i + B_col_offset < n && j + B_row_offset < m)
                    sB[i + ty][j + tx] = dB[j + i * size_t(ldb)];
                else
                    sB[i + ty][j + tx] = 0;
            }
        }

        __syncthreads();

        // multiply C = AB
        for(rocblas_int i = 0; i < NB; i++)
        {
            for(rocblas_int jn = 0; jn < THR_DIM; jn++)
            {
                for(rocblas_int jm = 0; jm < THR_DIM; jm++)
                {
                    if(LEFT)
                        rC[jn][jm] += sA[i][jm * DIM + tx] * sB[jn * DIM + ty][i];
                    else
                        rC[jn][jm] += sB[i][jm * DIM + tx] * sA[jn * DIM + ty][i];
                }
            }
        }

        // Iterate to next block column of A to multiply
        // For transpose, we iterate down the row of memory, effectively
        // iterating across the column of the transposed matrix
        if(LEFT)
        {
            dA += TRANSPOSE ? NB : NB * size_t(lda);
            A_col_offset += NB;
            dB += NB;
            B_row_offset += NB;
        }
        else
        {
            dA += !TRANSPOSE ? NB : NB * size_t(lda);
            A_row_offset += NB;
            dB += NB * size_t(ldb);
            B_col_offset += NB;
        }

        __syncthreads();
    }

    // store the C matrix
    for(rocblas_int jn = 0; jn < THR_DIM; jn++)
    {
        rocblas_int c_idxn = by * NB + jn * DIM + ty;
        for(rocblas_int jm = 0; jm < THR_DIM; jm++)
        {
            rocblas_int c_idxm = bx * NB + jm * DIM + tx;
            if(c_idxm < m && c_idxn < n)
            {
                C[c_idxn * size_t(ldc) + c_idxm] = alpha * rC[jn][jm];
            }
        }
    }
}

template<typename T, rocblas_int NB, bool LEFT, bool UPPER, bool TRANSPOSE, bool CONJ, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status trmm_outofplace(rocblas_handle   handle,
                                 rocblas_diagonal diag,
                                 rocblas_int      m,
                                 rocblas_int      n,
                                 TScal*           alpha,
                                 rocblas_stride   stride_alpha,
                                 TConstPtr*       dA, rocblas_stride offset_a, rocblas_int lda, rocblas_stride stride_a,
                                 TConstPtr*       dB, rocblas_stride offset_b, rocblas_int ldb, rocblas_stride stride_b,
                                 TPtr*            dC, rocblas_stride offset_c, rocblas_int lddc, rocblas_stride stride_c,
                                 rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();
    const rocblas_int THR_DIM = 2;
    dim3 threads(NB/THR_DIM, NB/THR_DIM, 1);

    rocblas_int blkx = (m - 1) / NB + 1;
    rocblas_int blky = (n - 1) / NB + 1;
    dim3 grid(blkx, blky, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL((rocblas_trmm_outofplace_kernel<T, NB, THR_DIM, LEFT, UPPER, TRANSPOSE, CONJ>), grid, threads, 0, rocblas_stream,
                            diag, m, n, alpha, stride_alpha,
                            dA, offset_a, lda, stride_a,
                            dB, offset_b, ldb, stride_b,
                            dC, offset_c, lddc, stride_c, batch_count);
    }
    else
    {
        hipLaunchKernelGGL((rocblas_trmm_outofplace_kernel<T, NB, THR_DIM, LEFT, UPPER, TRANSPOSE, CONJ>), grid, threads, 0, rocblas_stream,
                            diag, m, n, *alpha, stride_alpha,
                            dA, offset_a, lda, stride_a,
                            dB, offset_b, ldb, stride_b,
                            dC, offset_c, lddc, stride_c, batch_count);
    }

    return rocblas_status_success;
}

template <int NB, bool BATCHED, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trmm_outofplace_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation trans_a,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal*            alpha,
                                     rocblas_stride    stride_alpha,
                                     TConstPtr*        dA,
                                     rocblas_stride    offset_a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TConstPtr*        dB,
                                     rocblas_stride    offset_b,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_b,
                                     TPtr*             dC,
                                     rocblas_stride    offset_c,
                                     rocblas_int       lddc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count)
{
    rocblas_int shape = trmm_get_shape(side, uplo, trans_a);

    if(shape == 0) // lNl Left, NoTrans, Lower
    {
        trmm_outofplace<T, NB, true, false, false, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 1) // lNU Left, NoTrans, Upper
    {
        trmm_outofplace<T, NB, true, true, false, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 2) // lTL Left, Trans, Lower
    {
        trmm_outofplace<T, NB, true, false, true, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 3) // lTU Left, Trans, Upper
    {
        trmm_outofplace<T, NB, true, true, true, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 4) // rNL Right, NoTrans, Lower
    {
        trmm_outofplace<T, NB, false, false, false, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 5) // rNU Right, NoTrans, Upper
    {
        trmm_outofplace<T, NB, false, true, false, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 6) // rTL Right, Trans, Lower
    {
        trmm_outofplace<T, NB, false, false, true, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }
    else if(shape == 7) // rTU Right, Trans, Upper
    {
        trmm_outofplace<T, NB, false, true, true, CONJ>(handle, diag, m, n, alpha, stride_alpha,
                        dA, offset_a, lda, stride_a,
                        dB, offset_b, ldb, stride_b,
                        dC, offset_c, lddc, stride_c, batch_count);
    }

    return rocblas_status_success;
}

template <int STOPPING_NB, bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trmm_recursive_inplace_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation trans_a,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal*            alpha,
                                     rocblas_stride    stride_alpha,
                                     TConstPtr*        dA,
                                     rocblas_stride    offset_a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TPtr*             dB,
                                     rocblas_stride    offset_b,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_b,
                                     rocblas_int       batch_count)
{

#define CALC_OFFSET_A(i, j) offset_a + i + j * rocblas_stride(lda)
#define CALC_OFFSET_B(i, j) offset_b + i + j * rocblas_stride(ldb)

    const T one = 1.0;

    rocblas_int nrow_a = (side == rocblas_side_left ? m : n);
    // stopping condition
    if(nrow_a <= STOPPING_NB)
    {
        return rocblas_trmm_small<BATCHED, STOPPING_NB, T>
		                 (handle, side, uplo, trans_a, diag,
                                  m, n, alpha, stride_alpha,
                                  dA, offset_a, lda, stride_a,
                                  dB, offset_b, ldb, stride_b, batch_count);
    }

    rocblas_status status = rocblas_status_success;

    rocblas_int shape = trmm_get_shape(side, uplo, trans_a);

    if (shape == 0) // lNl    left, NoTrans, Lower
    {
        const int m1 = rocblas_get_trmm_recursive_nb(m);
        const int m2 = m - m1;

         RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), lda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), ldb, stride_b, batch_count)));

         RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, rocblas_operation_none,
                                     m2, n, m1, alpha,
                                     dA, CALC_OFFSET_A(m1, 0), lda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0, 0), ldb, stride_b, &one,
                                     dB, CALC_OFFSET_B(m1, 0), ldb, stride_b, batch_count)));

         RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));
    }
    else if (shape == 1) // lNU  left, NoTrans, Upper
    {
        const int m2 = rocblas_get_trmm_recursive_nb(m);
        const int m1 = m - m2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));


        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, rocblas_operation_none,
                                     m1, n, m2, alpha,
                                     dA, CALC_OFFSET_A( 0, m1), lda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B(m1,  0), ldb, stride_b, &one,
                                     dB, CALC_OFFSET_B( 0,  0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), lda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), ldb, stride_b, batch_count)));
    }
    else if (shape == 2) // lTL | lCL    left, Trans|ConjTrans, Lower
    {
        const int m2 = rocblas_get_trmm_recursive_nb(m);
        const int m1 = m - m2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, trans_a, rocblas_operation_none,
                                     m1, n, m2, alpha,
                                     dA, CALC_OFFSET_A(m1, 0), lda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B(m1, 0), ldb, stride_b, &one,
                                     dB, CALC_OFFSET_B( 0, 0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), lda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), ldb, stride_b, batch_count)));
    }
    else if (shape == 3) // lTU | lCU     left, Trans|ConjTrans, Upper
    {
        const int m1 = rocblas_get_trmm_recursive_nb(m);
        const int m2 = m - m1;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), lda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, trans_a, rocblas_operation_none,
                                     m2, n, m1, alpha,
                                     dA, CALC_OFFSET_A( 0, m1), lda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0,  0), ldb, stride_b, &one,
                                     dB, CALC_OFFSET_B(m1,  0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));
    }
    else if (shape == 4) // rNL       right, NoTrans, Lower
    {
        const int n2 = rocblas_get_trmm_recursive_nb(n);
        const int n1 = n - n2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n1, n2, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0, n1), ldb, stride_b,
                                     dA, CALC_OFFSET_A(n1,  0), lda, stride_a, &one,
                                     dB, CALC_OFFSET_B( 0,  0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), lda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), ldb, stride_b, batch_count)));
    }
    else if (shape == 5) // rNU       right, NoTrans, Upper
    {
        const int n1 = rocblas_get_trmm_recursive_nb(n);
        const int n2 = n - n1;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), lda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n2, n1, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B(0,  0), ldb, stride_b,
                                     dA, CALC_OFFSET_A(0, n1), lda, stride_a, &one,
                                     dB, CALC_OFFSET_B(0, n1), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));
    }
    else if (shape == 6) // rTL | rCL      right, Trans|ConjTrans, Lower
    {
        const int n1 = rocblas_get_trmm_recursive_nb(n);
        const int n2 = n - n1;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), lda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n2, n1, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0,  0), ldb, stride_b,
                                     dA, CALC_OFFSET_A(n1,  0), lda, stride_a, &one,
                                     dB, CALC_OFFSET_B( 0, n1), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));
    }
    else if (shape == 7) // rTU | rCU      right, Trans|ConjTrans, Upper
    {
        const int n2 = rocblas_get_trmm_recursive_nb(n);
        const int n1 = n - n2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), lda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n1, n2, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B(0, n1), ldb, stride_b,
                                     dA, CALC_OFFSET_A(0, n1), lda, stride_a, &one,
                                     dB, CALC_OFFSET_B(0,  0), ldb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_inplace_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), lda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), ldb, stride_b, batch_count)));
    }
    else
    {
        status = rocblas_status_internal_error;
    }
    return status;
}

// clang-format on

template <int NB, bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_template(rocblas_handle    handle,
                                   rocblas_side      side,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_diagonal  diag,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   TScal*            alpha,
                                   rocblas_stride    stride_alpha,
                                   TConstPtr*        dA,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   TConstPtr*        dB,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   TPtr*             dC,
                                   rocblas_stride    offset_c,
                                   rocblas_int       lddc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    const bool in_place = dB == dC;

    if(in_place)
    {
        return rocblas_internal_trmm_recursive_inplace_template<NB, BATCHED, T>(
            handle,
            side,
            uplo,
            trans_a,
            diag,
            m,
            n,
            alpha,
            stride_alpha,
            dA,
            rocblas_stride(offset_a),
            lda,
            stride_a,
            dC,
            rocblas_stride(offset_c),
            lddc,
            stride_c,
            batch_count);
    }
    else if(trans_a == rocblas_operation_conjugate_transpose)
    {
        return rocblas_internal_trmm_outofplace_template<NB, BATCHED, true, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m,
                                                                               n,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               offset_a,
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               offset_b,
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               offset_c,
                                                                               lddc,
                                                                               stride_c,
                                                                               batch_count);
    }
    else
    {
        return rocblas_internal_trmm_outofplace_template<NB, BATCHED, false, T>(handle,
                                                                                side,
                                                                                uplo,
                                                                                trans_a,
                                                                                diag,
                                                                                m,
                                                                                n,
                                                                                alpha,
                                                                                stride_alpha,
                                                                                dA,
                                                                                offset_a,
                                                                                lda,
                                                                                stride_a,
                                                                                dB,
                                                                                offset_b,
                                                                                ldb,
                                                                                stride_b,
                                                                                dC,
                                                                                offset_c,
                                                                                lddc,
                                                                                stride_c,
                                                                                batch_count);
    }
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files trmm*.cpp

// clang-format off
#ifdef INSTANTIATE_TRMM_TEMPLATE
#error INSTANTIATE_TRMM_TEMPLATE already defined
#endif

#define INSTANTIATE_TRMM_TEMPLATE(NB_, BATCHED_, T_, TScal_, TConstPtr_, TPtr_)          \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trmm_template  \
                                  <NB_, BATCHED_, T_, TScal_, TConstPtr_, TPtr_>         \
                                  (rocblas_handle    handle,                             \
                                   rocblas_side      side,                               \
                                   rocblas_fill      uplo,                               \
                                   rocblas_operation trans_a,                            \
                                   rocblas_diagonal  diag,                               \
                                   rocblas_int       m,                                  \
                                   rocblas_int       n,                                  \
                                   TScal_*           alpha,                              \
                                   rocblas_stride    stride_alpha,                       \
                                   TConstPtr_*       dA,                                 \
                                   rocblas_stride    offset_a,                           \
                                   rocblas_int       lda,                                \
                                   rocblas_stride    stride_a,                           \
                                   TConstPtr_*       dB,                                 \
                                   rocblas_stride    offset_b,                           \
                                   rocblas_int       ldb,                               \
                                   rocblas_stride    stride_b,                           \
                                   TPtr_*            dC,                                 \
                                   rocblas_stride    offset_c,                           \
                                   rocblas_int       lddc,                               \
                                   rocblas_stride    stride_c,                           \
                                   rocblas_int       batch_count);

// instantiate for rocblas_Xtrmm and rocblas_Xtrmm_strided_batched
INSTANTIATE_TRMM_TEMPLATE(32, false,  float,  float const,  float const,  float)
INSTANTIATE_TRMM_TEMPLATE(32, false, double, double const, double const, double)
INSTANTIATE_TRMM_TEMPLATE(32, false, rocblas_float_complex, rocblas_float_complex const, rocblas_float_complex const, rocblas_float_complex)
INSTANTIATE_TRMM_TEMPLATE(16, false, rocblas_float_complex, rocblas_float_complex const, rocblas_float_complex const, rocblas_float_complex)
INSTANTIATE_TRMM_TEMPLATE(32, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex const, rocblas_double_complex)
INSTANTIATE_TRMM_TEMPLATE(16, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex const, rocblas_double_complex)

// instantiate for rocblas_Xtrmm_batched
INSTANTIATE_TRMM_TEMPLATE(32, true,  float,  float const,  float const* const,  float* const)
INSTANTIATE_TRMM_TEMPLATE(32, true, double, double const, double const* const, double* const)
INSTANTIATE_TRMM_TEMPLATE(32, true, rocblas_float_complex, rocblas_float_complex const, rocblas_float_complex const* const, rocblas_float_complex* const)
INSTANTIATE_TRMM_TEMPLATE(16, true, rocblas_float_complex, rocblas_float_complex const, rocblas_float_complex const* const, rocblas_float_complex* const)
INSTANTIATE_TRMM_TEMPLATE(32, true, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex const* const, rocblas_double_complex* const)
INSTANTIATE_TRMM_TEMPLATE(16, true, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex const* const, rocblas_double_complex* const)

#undef INSTANTIATE_TRMM_TEMPLATE

#ifdef INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE
#error INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE already defined
#endif

#define INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(TScal_, TPtr_)     \
template rocblas_status set_matrix_zero_if_alpha_zero_template  \
                        <TScal_, TPtr_>                         \
                        (rocblas_handle handle,                 \
                         rocblas_int    m,                      \
                         rocblas_int    n,                      \
                         TScal_         alpha,                  \
                         rocblas_stride stride_alpha,           \
                         TPtr_          A,                      \
                         rocblas_int    lda,                    \
                         rocblas_stride a_st_or_of,             \
                         rocblas_int    batch_count);

INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( float const*,  float* const*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( float const*,  float*)

INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(double const*, double*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(double const*, double* const*)

INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex* const*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex* const*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex*)

#undef INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE

// clang-format on
