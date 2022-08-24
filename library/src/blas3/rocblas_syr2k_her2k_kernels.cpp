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

#include "handle.hpp"
#include "rocblas_syr2k_her2k.hpp"

template <typename T, typename U>
ROCBLAS_KERNEL_ILF void syr2k_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n && from <= to)
        C[ty * size_t(ldc) + tx] = beta ? beta * C[ty * size_t(ldc) + tx] : 0;
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename U, typename V>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
syr2k_scale_kernel(bool           upper,
                   rocblas_int    n,
                   U              beta_host_device,
                   V              CP_array,
                   rocblas_int    ldc,
                   rocblas_stride c_st_or_of)
{
    auto beta = load_scalar(beta_host_device);
    if(beta == 1)
        return;

    auto C = load_ptr_batch(CP_array, blockIdx.z, c_st_or_of);
    syr2k_scale_device(upper, n, beta, C, ldc);
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename U, typename V, typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
her2k_scale_kernel(bool           upper,
                   rocblas_int    n,
                   rocblas_int    k,
                   U              alpha_host_device,
                   V              beta_host_device,
                   W              CP_array,
                   rocblas_int    ldc,
                   rocblas_stride c_st_or_of)
{
    auto alpha = load_scalar(alpha_host_device);
    auto beta  = load_scalar(beta_host_device);

    if(beta == 1 && (k == 0 || alpha == 0)) // if alpha not zero we need imaginary clear on diagonal
        return;

    auto C = load_ptr_batch(CP_array, blockIdx.z, c_st_or_of);
    herk_scale_device(upper, n, beta, C, ldc);
}

/** helper for complex support */
template <typename T>
ROCBLAS_KERNEL_ILF void syr2k_her2k_zero_imaginary(T&)
{
}

template <typename T>
ROCBLAS_KERNEL_ILF void syr2k_her2k_zero_imaginary(rocblas_complex_num<T>& a)
{
    a.imag(0);
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
        syr2k_her2k_zero_imaginary(C[col * size_t(ldc) + row]);
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
syr2k_her2k_kernel(bool              upper,
                   rocblas_operation trans,
                   rocblas_int       n,
                   rocblas_int       k,
                   TScal             alpha_host_device,
                   TConstPtr         AP_array,
                   rocblas_int       lda,
                   rocblas_stride    a_st_or_of,
                   TConstPtr         BP_array,
                   rocblas_int       ldb,
                   rocblas_stride    b_st_or_of,
                   TPtr              CP_array,
                   rocblas_int       ldc,
                   rocblas_stride    c_st_or_of)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    auto A = load_ptr_batch(AP_array, blockIdx.z, a_st_or_of);
    auto B = load_ptr_batch(BP_array, blockIdx.z, b_st_or_of);
    auto C = load_ptr_batch(CP_array, blockIdx.z, c_st_or_of);

    // compute matrix multiplies and accumulate on the fly into C
    // when HERM does ^H in place of ^T
    syr2k_her2k_mult_add_device<TWOK, HERM, TRANS, DIM_XYT>(
        upper, n, k, alpha, A, lda, B, ldb, C, ldc);
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool BATCHED, bool TWOK, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal             alpha,
                                    TConstPtr         AP,
                                    rocblas_stride    offsetA,
                                    rocblas_int       lda,
                                    rocblas_stride    strideA,
                                    TConstPtr         BP,
                                    rocblas_stride    offsetB,
                                    rocblas_int       ldb,
                                    rocblas_stride    strideB,
                                    TScal             beta,
                                    TPtr              CP,
                                    rocblas_stride    offsetC,
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

    TPtr           CP_krn;
    TConstPtr      BP_krn;
    TConstPtr      AP_krn;
    rocblas_stride a_st_or_of;
    rocblas_stride b_st_or_of;
    rocblas_stride c_st_or_of;

    if(BATCHED)
    {
        CP_krn     = CP;
        BP_krn     = BP;
        AP_krn     = AP;
        a_st_or_of = offsetA;
        b_st_or_of = offsetB;
        c_st_or_of = offsetC;
    }
    else
    {
        CP_krn     = CP + offsetC;
        BP_krn     = BP + offsetB;
        AP_krn     = AP + offsetA;
        a_st_or_of = strideA;
        b_st_or_of = strideB;
        c_st_or_of = strideC;
    }

    // Launch a herk kernel for syr2k.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel<syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y>),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           beta,
                           CP_krn,
                           ldc,
                           c_st_or_of);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
    }
    else
    {
        if(*beta == 1 && (k == 0 || *alpha == 0))
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel<syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y>),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           *beta,
                           CP_krn,
                           ldc,
                           c_st_or_of);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
    }

    return rocblas_status_success;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool BATCHED,
          bool TWOK,
          typename TScal,
          typename TConstPtr,
          typename UScal,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal             alpha,
                                    TConstPtr         AP,
                                    rocblas_stride    offsetA,
                                    rocblas_int       lda,
                                    rocblas_stride    strideA,
                                    TConstPtr         BP,
                                    rocblas_stride    offsetB,
                                    rocblas_int       ldb,
                                    rocblas_stride    strideB,
                                    UScal             beta,
                                    TPtr              CP,
                                    rocblas_stride    offsetC,
                                    rocblas_int       ldc,
                                    rocblas_stride    strideC,
                                    rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    static constexpr int her2k_SCALE_DIM_X = 128;
    static constexpr int her2k_SCALE_DIM_Y = 8;
    rocblas_int          gx                = (n - 1) / (her2k_SCALE_DIM_X) + 1;
    rocblas_int          gy                = (n - 1) / (her2k_SCALE_DIM_Y) + 1;
    dim3                 her2k_scale_grid(gx, gy, batch_count);
    dim3                 her2k_scale_threads(her2k_SCALE_DIM_X, her2k_SCALE_DIM_Y);

    // Uses a syrk kernel in Hermitian mode
    static constexpr int  SYRK_DIM_XY = 32;
    rocblas_int           bx          = (n - 1) / (SYRK_DIM_XY) + 1;
    rocblas_int           by          = (n - 1) / (SYRK_DIM_XY) + 1;
    dim3                  syrk_grid(bx, by, batch_count);
    dim3                  syrk_threads(SYRK_DIM_XY, SYRK_DIM_XY);
    static constexpr bool Hermitian = true;

    TPtr           CP_krn;
    TConstPtr      BP_krn;
    TConstPtr      AP_krn;
    rocblas_stride a_st_or_of;
    rocblas_stride b_st_or_of;
    rocblas_stride c_st_or_of;

    if(BATCHED)
    {
        CP_krn     = CP;
        BP_krn     = BP;
        AP_krn     = AP;
        a_st_or_of = offsetA;
        b_st_or_of = offsetB;
        c_st_or_of = offsetC;
    }
    else
    {
        CP_krn     = CP + offsetC;
        BP_krn     = BP + offsetB;
        AP_krn     = AP + offsetA;
        a_st_or_of = strideA;
        b_st_or_of = strideB;
        c_st_or_of = strideC;
    }

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // scale C so we can use directly for output without work buffer, zeros diag imaginary
        hipLaunchKernelGGL((her2k_scale_kernel<her2k_SCALE_DIM_X, her2k_SCALE_DIM_Y>),
                           her2k_scale_grid,
                           her2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           alpha,
                           beta,
                           CP_krn,
                           ldc,
                           c_st_or_of);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
    }
    else
    {
        if(*beta == 1 && (k == 0 || *alpha == 0))
            return rocblas_status_success;

        // scale C so we can use directly for output without work buffer, zeros diag imaginary
        hipLaunchKernelGGL((her2k_scale_kernel<her2k_SCALE_DIM_X, her2k_SCALE_DIM_Y>),
                           her2k_scale_grid,
                           her2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           *alpha,
                           *beta,
                           CP_krn,
                           ldc,
                           c_st_or_of);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
    }

    return rocblas_status_success;
}

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her2k_syr2k_check_numerics(const char*       function_name,
                                                  rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation trans,
                                                  rocblas_int       n,
                                                  rocblas_int       k,
                                                  TConstPtr         A,
                                                  rocblas_int       lda,
                                                  rocblas_stride    strideA,
                                                  TConstPtr         B,
                                                  rocblas_int       ldb,
                                                  rocblas_stride    strideB,
                                                  TPtr              C,
                                                  rocblas_int       ldc,
                                                  rocblas_stride    strideC,
                                                  rocblas_int       batch_count,
                                                  const int         check_numerics,
                                                  bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;

    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              n,
                                                              k,
                                                              A,
                                                              0,
                                                              lda,
                                                              strideA,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              n,
                                                              k,
                                                              B,
                                                              0,
                                                              ldb,
                                                              strideB,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status = rocblas_internal_check_numerics_matrix_template(
        function_name,
        handle,
        rocblas_operation_none,
        uplo,
        HERM ? rocblas_client_hermitian_matrix : rocblas_client_symmetric_matrix,
        n,
        n,
        C,
        0,
        ldc,
        strideC,
        batch_count,
        check_numerics,
        is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syr2k*.cpp or her2k*.cpp

// clang-format off

#ifdef INSTANTIATE_SYR2K_TEMPLATE
#error INSTANTIATE_SYR2K_TEMPLATE already defined
#endif

#define INSTANTIATE_SYR2K_TEMPLATE(BATCHED, TWOK_, TScal_, TConstPtr_, TPtr_)            \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syr2k_template \
                                <BATCHED, TWOK_, TScal_, TConstPtr_, TPtr_>              \
				(rocblas_handle    handle,                               \
                                 rocblas_fill      uplo,                                 \
                                 rocblas_operation trans,                                \
                                 rocblas_int       n,                                    \
                                 rocblas_int       k,                                    \
                                 TScal_            alpha,                                \
                                 TConstPtr_        AP,                                   \
                                 rocblas_stride    offsetA,                              \
                                 rocblas_int       lda,                                  \
                                 rocblas_stride    strideA,                              \
                                 TConstPtr_        BP,                                   \
                                 rocblas_stride    offsetB,                              \
                                 rocblas_int       ldb,                                  \
                                 rocblas_stride    strideB,                              \
                                 TScal_            beta,                                 \
                                 TPtr_             CP,                                   \
                                 rocblas_stride    offsetC,                              \
                                 rocblas_int       ldc,                                  \
                                 rocblas_stride    strideC,                              \
                                 rocblas_int       batch_count);

INSTANTIATE_SYR2K_TEMPLATE(false, true, float const*, float const*, float*)
INSTANTIATE_SYR2K_TEMPLATE(false, true, double const*, double const*, double*)
INSTANTIATE_SYR2K_TEMPLATE(false, true, rocblas_float_complex const*, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_TEMPLATE(false, true, rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYR2K_TEMPLATE( true, true, float const*, float const* const*, float* const*)
INSTANTIATE_SYR2K_TEMPLATE( true, true, double const*, double const* const*, double* const*)
INSTANTIATE_SYR2K_TEMPLATE( true, true, rocblas_float_complex const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_TEMPLATE( true, true, rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_SYR2K_TEMPLATE(false, false, float const*, float const*, float*)
INSTANTIATE_SYR2K_TEMPLATE(false, false, double const*, double const*, double*)
INSTANTIATE_SYR2K_TEMPLATE(false, false, rocblas_float_complex const*, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_TEMPLATE(false, false, rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYR2K_TEMPLATE( true, false, float const*, float const* const*, float* const*)
INSTANTIATE_SYR2K_TEMPLATE( true, false, double const*, double const* const*, double* const*)
INSTANTIATE_SYR2K_TEMPLATE( true, false, rocblas_float_complex const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_TEMPLATE( true, false, rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYR2K_TEMPLATE

#ifdef INSTANTIATE_HER2K_TEMPLATE
#error INSTANTIATE_HER2K_TEMPLATE already defined
#endif

#define INSTANTIATE_HER2K_TEMPLATE(BATCHED_, TWOK_, TScal_, TConstPtr_, UScal_, TPtr_) \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                               \
rocblas_internal_her2k_template<BATCHED_, TWOK_, TScal_, TConstPtr_, UScal_, TPtr_>    \
                               (rocblas_handle    handle,                              \
                                rocblas_fill      uplo,                                \
                                rocblas_operation trans,                               \
                                rocblas_int       n,                                   \
                                rocblas_int       k,                                   \
                                TScal_            alpha,                               \
                                TConstPtr_        AP,                                  \
                                rocblas_stride    offsetA,                             \
                                rocblas_int       lda,                                 \
                                rocblas_stride    strideA,                             \
                                TConstPtr_        BP,                                  \
                                rocblas_stride    offsetB,                             \
                                rocblas_int       ldb,                                 \
                                rocblas_stride    strideB,                             \
                                UScal_            beta,                                \
                                TPtr_             CP,                                  \
                                rocblas_stride    offsetC,                             \
                                rocblas_int       ldc,                                 \
                                rocblas_stride    strideC,                             \
                                rocblas_int       batch_count);

INSTANTIATE_HER2K_TEMPLATE(false, true, rocblas_float_complex const*, rocblas_float_complex const*, float const*, rocblas_float_complex*)
INSTANTIATE_HER2K_TEMPLATE(false, true, rocblas_double_complex const*, rocblas_double_complex const*, double const*, rocblas_double_complex*)
INSTANTIATE_HER2K_TEMPLATE( true, true, rocblas_float_complex const*, rocblas_float_complex const* const*, float const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_TEMPLATE( true, true, rocblas_double_complex const*, rocblas_double_complex const* const*, double const*, rocblas_double_complex* const*)
INSTANTIATE_HER2K_TEMPLATE(false, false, rocblas_float_complex const*, rocblas_float_complex const*, float const*, rocblas_float_complex*)
INSTANTIATE_HER2K_TEMPLATE(false, false, rocblas_double_complex const*, rocblas_double_complex const*, double const*, rocblas_double_complex*)
INSTANTIATE_HER2K_TEMPLATE( true, false, rocblas_float_complex const*, rocblas_float_complex const* const*, float const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_TEMPLATE( true, false, rocblas_double_complex const*, rocblas_double_complex const* const*, double const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HER2K_TEMPLATE


#ifdef INSTANTIATE_HER2K_SYR2K_NUMERICS
#error INSTANTIATE_HER2K_SYR2K_NUMERICS already defined
#endif

#define INSTANTIATE_HER2K_SYR2K_NUMERICS(HERM_, TConstPtr_, TPtr_)                        \
template rocblas_status rocblas_her2k_syr2k_check_numerics                                \
                                  <HERM_, TConstPtr_, TPtr_>                            \
                                  (const char*       function_name,                     \
                                   rocblas_handle handle,                               \
                                   rocblas_fill   uplo,                                 \
                                   rocblas_operation trans,                             \
                                   rocblas_int    n,                                    \
                                   rocblas_int    k,                                    \
                                   TConstPtr_     A,                                    \
                                   rocblas_int    lda,                                  \
                                   rocblas_stride strideA,                              \
                                   TConstPtr_     B,                                    \
                                   rocblas_int    ldb,                                  \
                                   rocblas_stride strideB,                              \
                                   TPtr_          C,                                    \
                                   rocblas_int    ldc,                                  \
                                   rocblas_stride strideC,                              \
                                   rocblas_int    batch_count,                          \
                                   const int      check_numerics,                       \
                                   bool           is_input);

// instantiate for rocblas_Xher2k_Xsyr2k and rocblas_Xher2k_Xsyr2k_strided_batched
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, float const*, float*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, double const*, double*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true, rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xher2k_Xsyr2k_batched
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, float const* const*, float* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, double const* const*, double* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HER2K_SYR2K_NUMERICS
// clang-format on
