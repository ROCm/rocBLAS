/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_SYRK2K_HPP__
#define __ROCBLAS_SYRK2K_HPP__
#pragma once
#include "handle.h"

template <typename T, typename U>
__device__ void syr2k_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n && from <= to)
        C[ty * ldc + tx] *= beta;
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename U, typename V>
__global__ void syr2k_scale_kernel(bool           upper,
                                   rocblas_int    n,
                                   U              beta_host_device,
                                   V              CP_array,
                                   ptrdiff_t      shift_c,
                                   rocblas_int    ldc,
                                   rocblas_stride stride_c)
{
    auto beta = load_scalar(beta_host_device);
    if(beta == 1)
        return;

    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    syr2k_scale_device(upper, n, beta, C, ldc);
}

/** helper for complex support */
template <typename T>
__forceinline__ __device__ void syr2k_her2k_zero_imaginary(T&)
{
}

template <typename T>
__forceinline__ __device__ void syr2k_her2k_zero_imaginary(rocblas_complex_num<T>& a)
{
    a.imag(0);
}

/**
  * kernel
  */
template <bool TWOK, bool HERM, bool trans, rocblas_int TILE_NK, typename T, typename U>
__device__ void syr2k_her2k_mult_add_device(bool        upper,
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

        int row_loc, col_loc, k_loc;
        int r, c;

        // first matrix mult: alpha*op(A)*op(B)^T
        // when HERM ^H instead of ^T

        // fetch tile of matrix A
        row_loc = row_pos + threadIdx.x;
        col_loc = k_pos + threadIdx.y;
        r       = trans ? col_loc : row_loc; // trans A = A^T, else A = A
        c       = trans ? row_loc : col_loc;

        atile[threadIdx.x][threadIdx.y]
            = (r < ab_rows && c < ab_cols) ? (HERM && trans ? conj(A[c * lda + r]) : A[c * lda + r])
                                           : 0;

        // fetch tile of matrix B
        row_loc = k_pos + threadIdx.x;
        col_loc = col_pos + threadIdx.y;
        r       = trans ? row_loc : col_loc; // trans B = B, else B = B^T
        c       = trans ? col_loc : row_loc;

        btile[threadIdx.x][threadIdx.y]
            = (c < ab_cols && r < ab_rows)
                  ? (HERM && !trans ? conj(B[c * ldb + r]) : B[c * ldb + r])
                  : 0;

        __syncthreads();

        // n x n symmetric/hermitian output, tile zero where invalid
        if(row < n && col < n && from <= to)
        {
            T sum = T(0);
            for(int ki = 0; ki < TILE_NK; ++ki)
            {
                sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
            }
            C[col * ldc + row] += alpha * sum;
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
                      ? (HERM && trans ? conj(B[c * ldb + r]) : B[c * ldb + r])
                      : 0;

            // fetch tile of matrix A into tileB
            row_loc = k_pos + threadIdx.x;
            col_loc = col_pos + threadIdx.y;
            r       = trans ? row_loc : col_loc; // trans A = A, else A = A^T
            c       = trans ? col_loc : row_loc;

            btile[threadIdx.x][threadIdx.y]
                = (c < ab_cols && r < ab_rows)
                      ? (HERM && !trans ? conj(A[c * lda + r]) : A[c * lda + r])
                      : 0;

            __syncthreads();

            // n x n symmetric/hermitian output, tile zero where invalid
            if(row < n && col < n && from <= to)
            {
                T sum = T(0);
                for(int ki = 0; ki < TILE_NK; ++ki)
                {
                    sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
                }
                C[col * ldc + row] += (HERM ? conj(alpha) : alpha) * sum;
            }

            __syncthreads();
        }

    } // k_pos

    if(!TWOK && HERM && row == col && row < n)
    {
        // zero imaginary for cases when A*B aren't true Hermetian
        syr2k_her2k_zero_imaginary(C[col * ldc + row]);
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
__global__ void syr2k_her2k_kernel(bool              upper,
                                   rocblas_operation trans,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha_host_device,
                                   TConstPtr         AP_array,
                                   ptrdiff_t         shift_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   TConstPtr         BP_array,
                                   ptrdiff_t         shift_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   TPtr              CP_array,
                                   ptrdiff_t         shift_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    auto A = load_ptr_batch(AP_array, hipBlockIdx_z, shift_a, stride_a);
    auto B = load_ptr_batch(BP_array, hipBlockIdx_z, shift_b, stride_b);
    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);

    // compute matrix multiplies and accumulate on the fly into C
    // when HERM does ^H in place of ^T
    syr2k_her2k_mult_add_device<TWOK, HERM, TRANS, DIM_XYT>(
        upper, n, k, alpha, A, lda, B, ldb, C, ldc);
}

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syr2k_arg_check(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TScal             alpha,
                                              TConstPtr         AP,
                                              rocblas_int       offsetA,
                                              rocblas_int       lda,
                                              rocblas_stride    strideA,
                                              TConstPtr         BP,
                                              rocblas_int       offsetB,
                                              rocblas_int       ldb,
                                              rocblas_stride    strideB,
                                              TScal             beta,
                                              TPtr              CP,
                                              rocblas_int       offsetC,
                                              rocblas_int       ldc,
                                              rocblas_stride    strideC,
                                              rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n
       || (trans == rocblas_operation_none && (lda < n || ldb < n))
       || (trans != rocblas_operation_none && (lda < k || ldb < k)))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && (!AP || !BP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool TWOK, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_syr2k_template(rocblas_handle    handle,
                                                              rocblas_fill      uplo,
                                                              rocblas_operation trans,
                                                              rocblas_int       n,
                                                              rocblas_int       k,
                                                              TScal             alpha,
                                                              TConstPtr         AP,
                                                              rocblas_int       offsetA,
                                                              rocblas_int       lda,
                                                              rocblas_stride    strideA,
                                                              TConstPtr         BP,
                                                              rocblas_int       offsetB,
                                                              rocblas_int       ldb,
                                                              rocblas_stride    strideB,
                                                              TScal             beta,
                                                              TPtr              CP,
                                                              rocblas_int       offsetC,
                                                              rocblas_int       ldc,
                                                              rocblas_stride    strideC,
                                                              rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    static constexpr int syr2k_SCALE_DIM_X = 128;
    static constexpr int syr2k_SCALE_DIM_Y = 8;
    rocblas_int          gx                = (n - 1) / (syr2k_SCALE_DIM_X) + 1;
    rocblas_int          gy                = (n - 1) / (syr2k_SCALE_DIM_Y) + 1;
    dim3                 syr2k_scale_grid(gx, gy, batch_count);
    dim3                 syr2k_scale_threads(syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y);

    static constexpr int syr2k_DIM_XY = 32;
    rocblas_int          bx           = (n - 1) / (syr2k_DIM_XY) + 1;
    rocblas_int          by           = (n - 1) / (syr2k_DIM_XY) + 1;
    dim3                 syr2k_grid(bx, by, batch_count);
    dim3                 syr2k_threads(syr2k_DIM_XY, syr2k_DIM_XY);

    // Launch a herk kernel for syr2k.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if(*beta == 1 && (*alpha == 0 || k == 0))
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }

    return rocblas_status_success;
}

#endif
