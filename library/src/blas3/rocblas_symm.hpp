/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_SYMM_HPP__
#define __ROCBLAS_SYMM_HPP__

#include "handle.h"

template <typename T>
__device__ void symm_scale_device(rocblas_int m, rocblas_int n, T beta, T* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(tx < m && ty < n)
    {
        C[ty * ldc + tx] *= beta;
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename T, typename U>
__global__ void symm_scale_kernel(rocblas_int    m,
                                  rocblas_int    n,
                                  T              beta_host_device,
                                  U              CP_array,
                                  ptrdiff_t      shift_c,
                                  rocblas_int    ldc,
                                  rocblas_stride stride_c)
{
    auto beta = load_scalar(beta_host_device);
    if(beta == 1)
        return;

    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    symm_scale_device(m, n, beta, C, ldc);
}

/**
  * kernel
  */
template <bool HERM, bool RIGHT, rocblas_int TILE_NK, typename T>
__device__ void symm_hemm_mult_add_device(bool        upper,
                                          rocblas_int m,
                                          rocblas_int n,
                                          T           alpha,
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

    int row = row_pos + threadIdx.x;
    int col = col_pos + threadIdx.y;

    int from, to;

    int k_end = !RIGHT ? m : n;
    for(int k_pos = 0; k_pos < k_end; k_pos += TILE_NK)
    {
        // tiling over dimension K

        int row_loc, col_loc;
        int r, c;

        // when HERM ^H instead of ^T fetch A

        if(!RIGHT)
        {
            // premultiply: alpha*A*B

            // fetch tile of symm matrix A
            row_loc = row_pos + threadIdx.x;
            col_loc = k_pos + threadIdx.y;

            from = upper ? row_loc : col_loc;
            to   = upper ? col_loc : row_loc;

            r = from > to ? col_loc : row_loc;
            c = from > to ? row_loc : col_loc;

            if(!HERM)
            {
                atile[threadIdx.x][threadIdx.y] = (r < m && c < m) ? A[c * lda + r] : 0;
            }
            else
            {
                T e = (r < m && c < m)
                          ? (from > to ? conj(A[c * lda + r])
                                       : (from == to ? std::real(A[c * lda + r]) : A[c * lda + r]))
                          : 0;
                atile[threadIdx.x][threadIdx.y] = e;
            }

            // fetch tile of matrix B
            row_loc = k_pos + threadIdx.x;
            col_loc = col_pos + threadIdx.y;
            r       = row_loc;
            c       = col_loc;

            btile[threadIdx.x][threadIdx.y] = (r < m && c < n) ? B[c * ldb + r] : 0;

            __syncthreads();
        }
        else
        {
            // post multiply: alpha*B*A

            // fetch tile of matrix B  into tileA
            row_loc = row_pos + threadIdx.x;
            col_loc = k_pos + threadIdx.y;
            r       = row_loc;
            c       = col_loc;

            atile[threadIdx.x][threadIdx.y] = (r < m && c < n) ? B[c * ldb + r] : 0;

            // fetch tile of symm matrix A into tileB
            row_loc = k_pos + threadIdx.x;
            col_loc = col_pos + threadIdx.y;

            from = upper ? row_loc : col_loc;
            to   = upper ? col_loc : row_loc;

            r = from > to ? col_loc : row_loc;
            c = from > to ? row_loc : col_loc;

            if(!HERM)
            {
                btile[threadIdx.x][threadIdx.y] = (r < n && c < n) ? A[c * lda + r] : 0;
            }
            else
            {
                T e = (r < n && c < n)
                          ? (from > to ? conj(A[c * lda + r])
                                       : (from == to ? std::real(A[c * lda + r]) : A[c * lda + r]))
                          : 0;
                btile[threadIdx.x][threadIdx.y] = e;
            }

            __syncthreads();
        }

        // m x n output, tile zero where invalid
        if(row < m && col < n)
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
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        HERM,
          bool        RIGHT,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
__global__ void symm_hemm_kernel(bool           upper,
                                 rocblas_int    m,
                                 rocblas_int    n,
                                 TScal          alpha_host_device,
                                 TConstPtr      AP_array,
                                 ptrdiff_t      shift_a,
                                 rocblas_int    lda,
                                 rocblas_stride stride_a,
                                 TConstPtr      BP_array,
                                 ptrdiff_t      shift_b,
                                 rocblas_int    ldb,
                                 rocblas_stride stride_b,
                                 TPtr           CP_array,
                                 ptrdiff_t      shift_c,
                                 rocblas_int    ldc,
                                 rocblas_stride stride_c)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    auto A = load_ptr_batch(AP_array, hipBlockIdx_z, shift_a, stride_a);
    auto B = load_ptr_batch(BP_array, hipBlockIdx_z, shift_b, stride_b);
    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);

    // compute matrix multiplies and accumulate on the fly into C
    // when HERM does ^H in place of ^T for A fetches to symmetric empty side
    symm_hemm_mult_add_device<HERM, RIGHT, DIM_XYT>(upper, m, n, alpha, A, lda, B, ldb, C, ldc);
}

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_symm_arg_check(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      TScal          alpha,
                                      TConstPtr      AP,
                                      rocblas_int    offsetA,
                                      rocblas_int    lda,
                                      rocblas_stride strideA,
                                      TConstPtr      BP,
                                      rocblas_int    offsetB,
                                      rocblas_int    ldb,
                                      rocblas_stride strideB,
                                      TScal          beta,
                                      TPtr           CP,
                                      rocblas_int    offsetC,
                                      rocblas_int    ldc,
                                      rocblas_stride strideC,
                                      rocblas_int    batch_count)
{

    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldc < m || ldb < m
       || (side == rocblas_side_left && (lda < m)) || (side != rocblas_side_left && (lda < n)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!AP || !BP || !alpha || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool HERM, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_symm_template(rocblas_handle handle,
                                                             rocblas_side   side,
                                                             rocblas_fill   uplo,
                                                             rocblas_int    m,
                                                             rocblas_int    n,
                                                             TScal          alpha,
                                                             TConstPtr      AP,
                                                             rocblas_int    offsetA,
                                                             rocblas_int    lda,
                                                             rocblas_stride strideA,
                                                             TConstPtr      BP,
                                                             rocblas_int    offsetB,
                                                             rocblas_int    ldb,
                                                             rocblas_stride strideB,
                                                             TScal          beta,
                                                             TPtr           CP,
                                                             rocblas_int    offsetC,
                                                             rocblas_int    ldc,
                                                             rocblas_stride strideC,
                                                             rocblas_int    batch_count)
{
    // quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    static constexpr int symm_SCALE_DIM_X = 128;
    static constexpr int symm_SCALE_DIM_Y = 8;
    rocblas_int          gx               = (m - 1) / (symm_SCALE_DIM_X) + 1;
    rocblas_int          gy               = (n - 1) / (symm_SCALE_DIM_Y) + 1;
    dim3                 symm_scale_grid(gx, gy, batch_count);
    dim3                 symm_scale_threads(symm_SCALE_DIM_X, symm_SCALE_DIM_Y);

    static constexpr int symm_DIM_XY = 32;
    rocblas_int          bx          = (m - 1) / (symm_DIM_XY) + 1;
    rocblas_int          by          = (n - 1) / (symm_DIM_XY) + 1;
    dim3                 symm_grid(bx, by, batch_count);
    dim3                 symm_threads(symm_DIM_XY, symm_DIM_XY);

    // Launch a herk kernel for symm.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((symm_scale_kernel),
                           symm_scale_grid,
                           symm_scale_threads,
                           0,
                           handle->rocblas_stream,
                           m,
                           n,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(side == rocblas_side_left)
        {
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, false, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
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
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, true, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
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
        if(*beta == 1 && (*alpha == 0))
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((symm_scale_kernel),
                           symm_scale_grid,
                           symm_scale_threads,
                           0,
                           handle->rocblas_stream,
                           m,
                           n,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(side == rocblas_side_left)
        {
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, false, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
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
            hipLaunchKernelGGL((symm_hemm_kernel<HERM, true, symm_DIM_XY>),
                               symm_grid,
                               symm_threads,
                               0,
                               handle->rocblas_stream,
                               uplo == rocblas_fill_upper,
                               m,
                               n,
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
