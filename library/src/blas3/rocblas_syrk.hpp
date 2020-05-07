/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_SYRK_HPP__
#define __ROCBLAS_SYRK_HPP__
#include "handle.h"

template <typename T, typename U>
__device__ void syrk_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n && from <= to)
    {
        C[ty * ldc + tx] *= beta;
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename U, typename V>
__global__ void syrk_scale_kernel(bool           upper,
                                  rocblas_int    n,
                                  U              beta_host_device,
                                  V              CP_array,
                                  ptrdiff_t      shift_c,
                                  rocblas_int    ldc,
                                  rocblas_stride stride_c)
{
    auto C    = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    auto beta = load_scalar(beta_host_device);

    if(beta == 1)
        return;

    syrk_scale_device(upper, n, beta, C, ldc);
}

/**
  * kernel
  */
template <bool HERM, bool TRANSA, rocblas_int TILE_NK, typename T, typename U>
__device__ void syrk_herk_mult_add_device(bool        upper,
                                          rocblas_int n,
                                          rocblas_int k,
                                          U           alpha,
                                          const T* __restrict__ A,
                                          rocblas_int lda,
                                          T* __restrict__ C,
                                          rocblas_int ldc)
{
    __shared__ T atile[TILE_NK][TILE_NK];
    __shared__ T btile[TILE_NK][TILE_NK];

    int col_pos = blockIdx.y * TILE_NK;
    int row_pos = blockIdx.x * TILE_NK;

    int tilefrom = upper ? row_pos : col_pos;
    int tileto   = upper ? col_pos : row_pos;
    if(!alpha || tilefrom > tileto)
    {
        // any overlap of tile and output
        return;
    }

    int a_cols = !TRANSA ? k : n;
    int a_rows = !TRANSA ? n : k;

    int row = row_pos + threadIdx.x;
    int col = col_pos + threadIdx.y;

    int from = upper ? row : col;
    int to   = upper ? col : row;

    for(int k_pos = 0; k_pos < k; k_pos += TILE_NK)
    {
        // tiling over dimension K

        int row_loc, col_loc, k_loc;
        int r, c;

        // fetch tile of matrix A
        row_loc = row_pos + threadIdx.x;
        col_loc = k_pos + threadIdx.y;
        r       = TRANSA ? col_loc : row_loc; // true A = A^T, false A = A
        c       = TRANSA ? row_loc : col_loc;

        atile[threadIdx.x][threadIdx.y]
            = (r < a_rows && c < a_cols) ? (HERM && TRANSA ? conj(A[c * lda + r]) : A[c * lda + r])
                                         : 0;

        // fetch tile of matrix B
        row_loc = k_pos + threadIdx.x;
        col_loc = col_pos + threadIdx.y;
        r       = TRANSA ? row_loc : col_loc; // true B = A, false B = A^T
        c       = TRANSA ? col_loc : row_loc;

        btile[threadIdx.x][threadIdx.y]
            = (c < a_cols && r < a_rows) ? (HERM && !TRANSA ? conj(A[c * lda + r]) : A[c * lda + r])
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

    } // k_pos

    // if(HERM && row == col && row < n)
    // {
    //     // zero imaginary in case of numerical drift
    //     C[col * ldc + row].y = 0;
    // }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        HERM,
          bool        TRANS,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
__global__ void syrk_herk_kernel(bool              upper,
                                 rocblas_operation transA,
                                 rocblas_int       n,
                                 rocblas_int       k,
                                 TScal             alpha_host_device,
                                 TConstPtr         AP_array,
                                 ptrdiff_t         shift_a,
                                 rocblas_int       lda,
                                 rocblas_stride    stride_a,
                                 TPtr              CP_array,
                                 ptrdiff_t         shift_c,
                                 rocblas_int       ldc,
                                 rocblas_stride    stride_c)
{

    auto A     = load_ptr_batch(AP_array, hipBlockIdx_z, shift_a, stride_a);
    auto C     = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    auto alpha = load_scalar(alpha_host_device);

    // compute A^T * A or A * A^T and accumulate on the fly into C
    // when HERM does A^H in place of A^T

    if(alpha == 0)
        return;
    syrk_herk_mult_add_device<HERM, TRANS, DIM_XYT>(upper, n, k, alpha, A, lda, C, ldc);
}

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syrk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             TScal             alpha,
                                             TConstPtr         AP,
                                             rocblas_int       offsetA,
                                             rocblas_int       lda,
                                             rocblas_stride    strideA,
                                             TScal             beta,
                                             TPtr              CP,
                                             rocblas_int       offsetC,
                                             rocblas_int       ldc,
                                             rocblas_stride    strideC,
                                             rocblas_int       batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;
    if(!n || !batch_count)
        return rocblas_status_success;
    if((k > 0 && (!AP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_syrk_template(rocblas_handle    handle,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation transA,
                                                             rocblas_int       n,
                                                             rocblas_int       k,
                                                             TScal             alpha,
                                                             TConstPtr         AP,
                                                             rocblas_int       offsetA,
                                                             rocblas_int       lda,
                                                             rocblas_stride    strideA,
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

    static constexpr int SYRK_SCALE_DIM_X = 128;
    static constexpr int SYRK_SCALE_DIM_Y = 8;
    rocblas_int          gx               = (n - 1) / (SYRK_SCALE_DIM_X) + 1;
    rocblas_int          gy               = (n - 1) / (SYRK_SCALE_DIM_Y) + 1;
    dim3                 syrk_scale_grid(gx, gy, batch_count);
    dim3                 syrk_scale_threads(SYRK_SCALE_DIM_X, SYRK_SCALE_DIM_Y);

    static constexpr int SYRK_DIM_XY = 32;
    rocblas_int          bx          = (n - 1) / (SYRK_DIM_XY) + 1;
    rocblas_int          by          = (n - 1) / (SYRK_DIM_XY) + 1;
    dim3                 syrk_grid(bx, by, batch_count);
    dim3                 syrk_threads(SYRK_DIM_XY, SYRK_DIM_XY);

    // Launch a herk kernel for syrk.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syrk_scale_kernel),
                           syrk_scale_grid,
                           syrk_scale_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if((!*alpha || k == 0) && *beta == 1)
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syrk_scale_kernel),
                           syrk_scale_grid,
                           syrk_scale_threads,
                           0,
                           handle->rocblas_stream,
                           uplo == rocblas_fill_upper,
                           n,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }

    return rocblas_status_success;
}

#endif
