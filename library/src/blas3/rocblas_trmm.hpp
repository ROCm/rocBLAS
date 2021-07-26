/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#ifdef BUILD_WITH_TENSILE
#include "Tensile/gemm_tensile.hpp"
#else
#include "Tensile/gemm_source.hpp"
#endif

#include "Tensile/gemm.hpp"
#include "definitions.hpp"

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

template <typename TScal, typename TPtr, typename T_lda>
ROCBLAS_KERNEL void set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                                         rocblas_int    n,
                                                         TScal          alpha_device_host,
                                                         rocblas_stride stride_alpha,
                                                         TPtr           Aa,
                                                         T_lda          offsetA,
                                                         T_lda          lda,
                                                         rocblas_stride strideA)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);

    if(tx < m && ty < n && alpha == 0)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offsetA, strideA);

        A[tx + lda * ty] = 0;
    }
}

template <typename TScal, typename TPtr, typename T_lda>
rocblas_status set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TPtr           A,
                                                      T_lda          offsetA,
                                                      T_lda          lda,
                                                      rocblas_stride strideA,
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
        hipLaunchKernelGGL(set_matrix_zero_if_alpha_zero_kernel,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           alpha,
                           stride_alpha,
                           A,
                           offsetA,
                           lda,
                           strideA);
    else
        hipLaunchKernelGGL(set_matrix_zero_if_alpha_zero_kernel,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           *alpha,
                           stride_alpha,
                           A,
                           offsetA,
                           lda,
                           strideA);
    return rocblas_status_success;
}

// left, NoTrans
template <const int NB,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr,
          typename T_lda>
ROCBLAS_KERNEL __launch_bounds__(NB* NB) void rocblas_trmm_lNx_kernel(rocblas_fill     uplo,
                                                                      rocblas_diagonal diag,
                                                                      int              m,
                                                                      int   n, // m must be <= NB
                                                                      TScal alpha_device_host,
                                                                      rocblas_stride stride_alpha,
                                                                      TConstPtr*     A_arg,
                                                                      T_lda          offset_a,
                                                                      T_lda          ldda,
                                                                      rocblas_stride stride_a,
                                                                      TPtr*          B_arg,
                                                                      T_lda          offset_b,
                                                                      T_lda          lddb,
                                                                      rocblas_stride stride_b)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, offset_a, stride_a);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, offset_b, stride_b);

    const int nblocks = (n + NB - 1) / NB;
    const int nn      = (bx < nblocks - 1) ? NB : n - (nblocks - 1) * NB;
    B += bx * NB * lddb;

    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    // initialize sA and sB to zero
    sA[ty * NB + tx] = 0;
    sB[ty * NB + tx] = 0;

    // load A and B
    if(ty < m && tx < m)
        sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < nn && tx < m)
        sB[ty * NB + tx] = B[ty * lddb + tx];

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
        B[ty * lddb + tx] = accumulator;
}

// left, Trans|ConjTrans
template <const int NB,
          bool      CONJA,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr,
          typename T_lda>
ROCBLAS_KERNEL __launch_bounds__(NB* NB) void rocblas_trmm_lTx_kernel(rocblas_fill     uplo,
                                                                      rocblas_diagonal diag,
                                                                      int              m,
                                                                      int   n, // m must be <= NB
                                                                      TScal alpha_device_host,
                                                                      rocblas_stride stride_alpha,
                                                                      TConstPtr*     A_arg,
                                                                      T_lda          offset_a,
                                                                      T_lda          ldda,
                                                                      rocblas_stride stride_a,
                                                                      TPtr*          B_arg,
                                                                      T_lda          offset_b,
                                                                      T_lda          lddb,
                                                                      rocblas_stride stride_b)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, offset_a, stride_a);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, offset_b, stride_b);

    const int nblocks = (n + NB - 1) / NB;
    const int nn      = (bx < nblocks - 1) ? NB : n - (nblocks - 1) * NB;
    B += bx * NB * lddb;

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
            sA[tx * NB + ty] = conj(A[ty * ldda + tx]);
        }
        else
        {
            sA[tx * NB + ty] = A[ty * ldda + tx];
        }
    }
    if(ty < nn && tx < m)
        sB[ty * NB + tx] = B[ty * lddb + tx];

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
        B[ty * lddb + tx] = accumulator;
}

// right NoTrans
template <const int NB,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr,
          typename T_lda>
ROCBLAS_KERNEL __launch_bounds__(NB* NB) void rocblas_trmm_rNx_kernel(rocblas_fill     uplo,
                                                                      rocblas_diagonal diag,
                                                                      int              m,
                                                                      int   n, // m must be <= NB
                                                                      TScal alpha_device_host,
                                                                      rocblas_stride stride_alpha,
                                                                      TConstPtr*     A_arg,
                                                                      T_lda          offset_a,
                                                                      T_lda          ldda,
                                                                      rocblas_stride stride_a,
                                                                      TPtr*          B_arg,
                                                                      T_lda          offset_b,
                                                                      T_lda          lddb,
                                                                      rocblas_stride stride_b)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, offset_a, stride_a);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, offset_b, stride_b);

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
        sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < n && tx < mm)
        sB[ty * NB + tx] = B[ty * lddb + tx];

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
        B[ty * lddb + tx] = accumulator;
}

// right, transpose_and_conjugate_transpose
template <const int NB,
          bool      CONJA,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr,
          typename T_lda>
ROCBLAS_KERNEL __launch_bounds__(NB* NB) void rocblas_trmm_rTx_kernel(rocblas_fill     uplo,
                                                                      rocblas_diagonal diag,
                                                                      int              m,
                                                                      int   n, // m must be <= NB
                                                                      TScal alpha_device_host,
                                                                      rocblas_stride stride_alpha,
                                                                      TConstPtr*     A_arg,
                                                                      T_lda          offset_a,
                                                                      T_lda          ldda,
                                                                      rocblas_stride stride_a,
                                                                      TPtr*          B_arg,
                                                                      T_lda          offset_b,
                                                                      T_lda          lddb,
                                                                      rocblas_stride stride_b)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    T alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(alpha == 0)
        return;
    auto* A = load_ptr_batch(A_arg, hipBlockIdx_z, offset_a, stride_a);
    auto* B = load_ptr_batch(B_arg, hipBlockIdx_z, offset_b, stride_b);

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
            sA[ty * NB + tx] = conj(A[ty * ldda + tx]);
        }
        else
        {
            sA[ty * NB + tx] = A[ty * ldda + tx];
        }
    }
    if(ty < n && tx < mm)
        sB[ty * NB + tx] = B[ty * lddb + tx];

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
        B[ty * lddb + tx] = accumulator;
}

// clang-format off
// left, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr, typename T_lda>
rocblas_status trmm_template_lNx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, T_lda offset_a, T_lda ldda, rocblas_stride stride_a,
                       TPtr*            dB, T_lda offset_b, T_lda lddb, rocblas_stride stride_b,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((n + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_lNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);
    else
        hipLaunchKernelGGL((rocblas_trmm_lNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);

    return rocblas_status_success;
}

// left, Trans|ConjTrans
template <const int NB, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr, typename T_lda>
rocblas_status trmm_template_lTx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, T_lda offset_a, T_lda ldda, rocblas_stride stride_a,
                       TPtr*            dB, T_lda offset_b, T_lda lddb, rocblas_stride stride_b,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((n + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_lTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);
    else
        hipLaunchKernelGGL((rocblas_trmm_lTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);

    return rocblas_status_success;
}

// right, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr, typename T_lda>
rocblas_status trmm_template_rNx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, T_lda offset_a, T_lda ldda, rocblas_stride stride_a,
                       TPtr*            dB, T_lda offset_b, T_lda lddb, rocblas_stride stride_b,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((m + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_rNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);
    else
        hipLaunchKernelGGL((rocblas_trmm_rNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);

    return rocblas_status_success;
}

// right, Trans|ConjTrans
template <const int NB, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr, typename T_lda>
rocblas_status trmm_template_rTx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, T_lda offset_a, T_lda ldda, rocblas_stride stride_a,
                       TPtr*            dB, T_lda offset_b, T_lda lddb, rocblas_stride stride_b,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();

    dim3 threads(NB, NB, 1);
    dim3 grid((m + NB - 1) / NB, 1, batch_count);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_trmm_rTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);
    else
        hipLaunchKernelGGL((rocblas_trmm_rTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, offset_a, ldda, stride_a,
                           dB, offset_b, lddb, stride_b);

    return rocblas_status_success;
}

template <int STOPPING_NB, typename T, typename TScal, typename TConstPtr, typename TPtr, typename T_lda>
rocblas_status rocblas_trmm_small(rocblas_handle    handle,
                        rocblas_side      side,
                        rocblas_fill      uplo,
                        rocblas_operation trans_a,
                        rocblas_diagonal  diag,
                        rocblas_int       m,
                        rocblas_int       n,
                        TScal*            alpha,
                        rocblas_stride    stride_alpha,
                        TConstPtr*        dA, T_lda offset_a, T_lda ldda, rocblas_stride stride_a,
                        TPtr*             dB, T_lda offset_b, T_lda lddb, rocblas_stride stride_b,
                        rocblas_int       batch_count)
{
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
                                               dA, offset_a, ldda, stride_a,
                                               dB, offset_b, lddb, stride_b, batch_count);
    else if (shape == 1) // lTx, left, Transpose
        return trmm_template_lTx<STOPPING_NB, false, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA, offset_a, ldda, stride_a,
                                               dB, offset_b, lddb, stride_b, batch_count);
    else if (shape == 2) // lCx, left, ConjTrans
        return trmm_template_lTx<STOPPING_NB, true, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA, offset_a, ldda, stride_a,
                                               dB, offset_b, lddb, stride_b, batch_count);
    else if (shape == 3) // rNx, right, NoTrans
        return trmm_template_rNx<STOPPING_NB, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA, offset_a, ldda, stride_a,
                                               dB, offset_b, lddb, stride_b, batch_count);
    else if (shape == 4) // rTx, right, Transpose
        return trmm_template_rTx<STOPPING_NB, false, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA, offset_a, ldda, stride_a,
                                               dB, offset_b, lddb, stride_b, batch_count);
    else if (shape == 5) // rCx, right, ConjTrans
        return trmm_template_rTx<STOPPING_NB, true, T>(handle, uplo, diag,
                                               m, n, alpha, stride_alpha,
                                               dA, offset_a, ldda, stride_a,
                                               dB, offset_b, lddb, stride_b, batch_count);
    else
        return rocblas_status_internal_error;
}

template <int STOPPING_NB, bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr, typename T_lda>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trmm_recursive_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation trans_a,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal*            alpha,
                                     rocblas_stride    stride_alpha,
                                     TConstPtr*        dA,
                                     T_lda             offset_a,
                                     T_lda             ldda,
                                     rocblas_stride    stride_a,
                                     TPtr*             dB,
                                     T_lda             offset_b,
                                     T_lda             lddb,
                                     rocblas_stride    stride_b,
                                     rocblas_int       batch_count)
{

#define CALC_OFFSET_A(i, j) offset_a + i + j* ldda
#define CALC_OFFSET_B(i, j) offset_b + i + j* lddb

    const T one = 1.0;

    rocblas_int nrow_a = (side == rocblas_side_left ? m : n);
    // stopping condition
    if(nrow_a <= STOPPING_NB)
    {
        return rocblas_trmm_small<STOPPING_NB, T>(handle, side, uplo, trans_a, diag,
                                                  m, n, alpha, stride_alpha,
                                                  dA, offset_a, ldda, stride_a,
                                                  dB, offset_b, lddb, stride_b, batch_count);
    }

    rocblas_status status = rocblas_status_success;

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

    if (shape == 0) // lNl    left, NoTrans, Lower
    {
        const int m1 = rocblas_get_trmm_recursive_nb(m);
        const int m2 = m - m1;

         RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), ldda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), lddb, stride_b, batch_count)));

         RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, rocblas_operation_none,
                                     m2, n, m1, alpha,
                                     dA, CALC_OFFSET_A(m1, 0), ldda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0, 0), lddb, stride_b, &one,
                                     dB, CALC_OFFSET_B(m1, 0), lddb, stride_b, batch_count)));

         RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));
    }
    else if (shape == 1) // lNU  left, NoTrans, Upper
    {
        const int m2 = rocblas_get_trmm_recursive_nb(m);
        const int m1 = m - m2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));


        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, rocblas_operation_none,
                                     m1, n, m2, alpha,
                                     dA, CALC_OFFSET_A( 0, m1), ldda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B(m1,  0), lddb, stride_b, &one,
                                     dB, CALC_OFFSET_B( 0,  0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), ldda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), lddb, stride_b, batch_count)));
    }
    else if (shape == 2) // lTL | lCL    left, Trans|ConjTrans, Lower
    {
        const int m2 = rocblas_get_trmm_recursive_nb(m);
        const int m1 = m - m2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, trans_a, rocblas_operation_none,
                                     m1, n, m2, alpha,
                                     dA, CALC_OFFSET_A(m1, 0), ldda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B(m1, 0), lddb, stride_b, &one,
                                     dB, CALC_OFFSET_B( 0, 0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), ldda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), lddb, stride_b, batch_count)));
    }
    else if (shape == 3) // lTU | lCU     left, Trans|ConjTrans, Upper
    {
        const int m1 = rocblas_get_trmm_recursive_nb(m);
        const int m2 = m - m1;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m2, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(m1, m1), ldda, stride_a,
                                     dB, CALC_OFFSET_B(m1,  0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, trans_a, rocblas_operation_none,
                                     m2, n, m1, alpha,
                                     dA, CALC_OFFSET_A( 0, m1), ldda, stride_a,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0,  0), lddb, stride_b, &one,
                                     dB, CALC_OFFSET_B(m1,  0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m1, n, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));
    }
    else if (shape == 4) // rNL       right, NoTrans, Lower
    {
        const int n2 = rocblas_get_trmm_recursive_nb(n);
        const int n1 = n - n2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n1, n2, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0, n1), lddb, stride_b,
                                     dA, CALC_OFFSET_A(n1,  0), ldda, stride_a, &one,
                                     dB, CALC_OFFSET_B( 0,  0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), ldda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), lddb, stride_b, batch_count)));
    }
    else if (shape == 5) // rNU       right, NoTrans, Upper
    {
        const int n1 = rocblas_get_trmm_recursive_nb(n);
        const int n2 = n - n1;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), ldda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n2, n1, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B(0,  0), lddb, stride_b,
                                     dA, CALC_OFFSET_A(0, n1), ldda, stride_a, &one,
                                     dB, CALC_OFFSET_B(0, n1), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));
    }
    else if (shape == 6) // rTL | rCL      right, Trans|ConjTrans, Lower
    {
        const int n1 = rocblas_get_trmm_recursive_nb(n);
        const int n2 = n - n1;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), ldda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n2, n1, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B( 0,  0), lddb, stride_b,
                                     dA, CALC_OFFSET_A(n1,  0), ldda, stride_a, &one,
                                     dB, CALC_OFFSET_B( 0, n1), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));
    }
    else if (shape == 7) // rTU | rCU      right, Trans|ConjTrans, Upper
    {
        const int n2 = rocblas_get_trmm_recursive_nb(n);
        const int n1 = n - n2;

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n1, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(0, 0), ldda, stride_a,
                                     dB, CALC_OFFSET_B(0, 0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_template<BATCHED, T>(handle, rocblas_operation_none, trans_a,
                                     m, n1, n2, alpha,
                        (TConstPtr*) dB, CALC_OFFSET_B(0, n1), lddb, stride_b,
                                     dA, CALC_OFFSET_A(0, n1), ldda, stride_a, &one,
                                     dB, CALC_OFFSET_B(0,  0), lddb, stride_b, batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle, side, uplo, trans_a, diag,
                                     m, n2, alpha, stride_alpha,
                                     dA, CALC_OFFSET_A(n1, n1), ldda, stride_a,
                                     dB, CALC_OFFSET_B( 0, n1), lddb, stride_b, batch_count)));
    }
    else
    {
        status = rocblas_status_internal_error;
    }
    return status;
}
// clang-format on
