/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

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

constexpr int rocblas_hemv_DIM_X()
{
    return 64;
}

/*! \brief rocblas_internal_hemv_kernel_workspace_size
    workspace buffer for column reductions: number of blocks * cols * batch_count

    @param[in]
    outputType To*
        Type of output values
    @param[in]
    n rocblas_int
        Number of columns
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t
    rocblas_internal_hemv_symv_kernel_workspace_size(rocblas_int n, rocblas_int batch_count = 1)
{
    auto blocks = (n - 1) / rocblas_hemv_DIM_X() + 1;
    return sizeof(To) * blocks * n * batch_count;
}

/** helper for complex support */
template <typename T>
ROCBLAS_KERNEL_ILF void hemv_zero_imaginary(T&)
{
}

template <typename T>
ROCBLAS_KERNEL_ILF void hemv_zero_imaginary(rocblas_complex_num<T>& a)
{
    a.imag(0);
}

// treats sA as 16x64 block
#define sA16(i_, j_) (sA[(i_)][(j_)]) // i.e., sA[ (i_)*(NB_X+3) + (j_) ]

// treats sA as 32x32 block
#define sA32(i_, j_) (sA[0][(i_) + bank_shift * (j_)])

/***************************************************************************/ /**
    Upper case, compute block multiply, workspace = A*x, for any size n:

                [ (A11*x1 + A12*x2 + A13*x3)     ---                 ---    ]   [ A11    A12    A13 ]   [ x1 ]
    workspace = [ (A12^H*x1)                   (A22*x2 + A23*x3)     ---    ] = [ A12^H  A22    A23 ] * [ x2 ]
                [ (A13^H*x1)                   (A23^H*x2)          (A33*x3) ]   [ A13^H  A23^H  A33 ]   [ x3 ]

    The order is different from the lower case, because
    the upper case processes a block row from the diagonal to the right, whereas
    the lower case processes a block row from the diagonal to the left.

    Uses a 64x4 thread block.
    For     diagonal tiles, covers a 64x64 tile using three 32x32 tiles (plus one gets transposed).
    For off-diagonal tiles, covers a 64x64 tile using four  64x16 tiles.
    In both cases, each thread multiplies 4 elements.

    For rows past the bottom of the matrix, the A pointer is adjusted to be the
    last valid row of A, which multiple threads will read.
    Extra rows are ignored when saving results to workspace.
    Columns past the right edge are explicitly ignored when loading.
    x values past the bottom are set to zero, thus, extra columns are zeroed
    when multiplying.
*******************************************************************************/
template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_lda,
          typename T>
ROCBLAS_KERNEL_ILF void hemvn_kernel_upper_calc(rocblas_int n,
                                                T           alpha,
                                                const T* __restrict__ A,
                                                T_lda lda,
                                                const T* __restrict__ x,
                                                ptrdiff_t incx,
                                                T* __restrict__ workspace)
{
    if(!alpha)
        return;
    // 64x4 thread block
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int blk     = blockIdx.x;
    const int blk_ind = NB_X * blk;
    const int td      = NB_X * ty + tx;

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    int partial = (blk == gridDim.x - 1 ? (n % NB_X) : 0);

    T psum, psum_t;
    T total = T{0};

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
    __shared__ T sA[quarter_NB_X][NB_X + 3]; // only needs +2, use +3 to avoid bank conflict
    __shared__ T sx_blk[NB_X]; // for x[ blk ]
    __shared__ T sx_jj[NB_X]; // for x[ jj ], which cycles over all blocks right of diag

    T rA[4];
    T psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx) * incx; // x is x(blk_ind + tx)
    if(ty == 0)
    {
        if(partial == 0 || tx < partial)
        {
            sx_blk[tx] = x[0];
        }
        else
        {
            sx_blk[tx] = T{0};
        }
    }

    // offset blocks * cols * batch
    workspace
        += size_t(hipGridDim_x) * n * hipBlockIdx_y; // workspace is workspace(0, 0, batch_count)

    // --------------------
    // move to block row
    workspace += blk * n; // workspace is workspace(0, blk)

    A += blk_ind; // A is A(blk_ind, 0)
    A += ty2 * lda + tx2; // A is A(blk_ind + tx2, ty2)

    // move to 32x32 diag block
    A += blk_ind * lda; // A is A(blk_ind + tx2, blk_ind + ty2)

    // load 32x32 diag block A(blk_ind + 0:31, blk_ind + 0:31) into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 >= partial)
        {
            A = A - tx2
                + (partial
                   - 1); // A is A(blk_ind + partial-1, blk_ind + ty2), the bottom-most valid row
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 >= partial)
        {
            A = A + tx2 - (partial - 1); // A is A(blk_ind + tx2, blk_ind + ty2)
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

// symmetrize 32x32 diag block, copying upper to lower triangle,
// as four 32x8 sections in parallel:
// columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
#pragma unroll
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j > tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 32x32 diag block, then repeat steps from first diag block
    A += half_NB_X + half_NB_X * lda; // A is A(blk_ind + NB/2 + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 diag block A[block + 0:31, block + 0:31] into sA
    if(partial)
    {
        if(tx2 + half_NB_X >= partial)
        {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j + half_NB_X < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 + half_NB_X >= partial)
        {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

// symmetrize 32x32 diag block, copying upper to lower triangle
#pragma unroll
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j > tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to off-diag 32x32 block
    A -= half_NB_X; // A is A(blk_ind + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 block of A into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 >= partial)
        {
            A = A - (tx2) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j + half_NB_X < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 >= partial)
        {
            A = A + (tx2) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

    // multiply 32x32 block (below diag)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += (IS_HEMV ? conj(sA32(ty2 + j * 8, tx2)) : sA32(ty2 + j * 8, tx2))
                * sx_blk[j * 8 + ty2];
    }
    //__syncthreads();  // no sync needed here

    // multiply transposed 32x32 block (above diag)
    psum_t = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum_t += sA32(tx2, ty2 * 4 + j) * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial sums for non-transposed 32x32 block
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // store partial sums for transposed 32x32 block
    sA32(ty2, tx2) = psum_t;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 64x64 block right of diag in block row, and
    // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
    A += half_NB_X * lda; // A is A(blk_ind + tx2, blk_ind + NB_X + ty2 )
    A -= ty2 * lda + tx2; // A is A(blk_ind,       blk_ind + NB_X       )
    A += 4 * ty * lda + tx; // A is A(blk_ind + tx,  blk_ind        + 4*ty)

    // Unlike lower case, don't adjust A here for partial # of rows.
    // Since block is right of diagonal, it must have all NB rows,
    // but can have < NB columns, dealt with when loading below.

    x -= blk_ind * incx; // x is x(tx)

    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;

    // cycle over blocks jj right of diagonal, in block row blk
    for(int jj = blk + 1; jj < gridDim.x; ++jj)
    {
        partial = (jj == gridDim.x - 1 ? (n % NB_X) : 0);

        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        if(ty == 0)
        {
            if(partial == 0 || tx < partial)
            {
                sx_jj[tx] = x[jj * NB_X * incx];
            }
            else
            {
                sx_jj[tx] = T{0};
            }
        }
        __syncthreads();

        for(int k = 0; k < 4; k++)
        {
            // load 64x16 block of A into rA, 4 elements per thread,
            // as four 64x4 sections in parallel:
            // columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
            if(partial)
            {
#pragma unroll
                for(int j = 0; j < 4; j++)
                {
                    if(4 * ty + j + k * quarter_NB_X < partial)
                    {
                        rA[j] = A[j * lda];
                    }
                    else
                    {
                        rA[j] = T{0};
                    }
                }
            }
            else
            {
#pragma unroll
                for(int j = 0; j < 4; j++)
                {
                    rA[j] = A[j * lda];
                }
            }

// 1) multiply 64x16 block A_{blk,jj} * x_jj
//    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
// 2) multiply 16x64 block A_{blk,jj} * x_blk,
//    storing each product Aji*xi to sA(j,i)
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                total
                    += rA[j] * sx_jj[quarter_NB_X * k + ty * 4 + j]; // y_blk = A_{blk,jj}   * x_jj
                sA16(ty * 4 + j, tx)
                    = (IS_HEMV ? conj(rA[j]) : rA[j]) * sx_blk[tx]; // y_jj  = A_{blk,jj}^H * x_blk
            }
            __syncthreads();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            psum_t = T{0};
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                psum_t += sA16(tx4, ty4 * 4 + j);
            }
            __syncthreads();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;

            // move right to next 64x16 block
            A += lda * quarter_NB_X; // A is A(blk_ind + tx, jj*NB_X + (k+1)*NB_X/4 + 4*ty)
        }
// already at next 64x64 block
// A is A(blk_ind + tx, (jj+1)*NB_x + 4*ty)

// store partial row sums of transposed result, y_jj
#pragma unroll
        for(int k = 0; k < 4; k++)
        {
            sA16(tx4, ty4 + quarter_NB_X * k) = psums_t[k];
        }
        __syncthreads();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        if(ty4 < 4 && (partial == 0 || tx4 + ty4 * quarter_NB_X < partial))
        {
            int ty4_nb4 = ty4 * quarter_NB_X;
            psum_t      = sA16(tx4, 0 + ty4_nb4) + sA16(tx4, 1 + ty4_nb4) + sA16(tx4, 2 + ty4_nb4)
                     + sA16(tx4, 3 + ty4_nb4) + sA16(tx4, 4 + ty4_nb4) + sA16(tx4, 5 + ty4_nb4)
                     + sA16(tx4, 6 + ty4_nb4) + sA16(tx4, 7 + ty4_nb4) + sA16(tx4, 8 + ty4_nb4)
                     + sA16(tx4, 9 + ty4_nb4) + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                     + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4) + sA16(tx4, 14 + ty4_nb4)
                     + sA16(tx4, 15 + ty4_nb4);
            workspace[jj * NB_X + tx4 + ty4_nb4]
                = psum_t; // store at workspace( jj*NB_X + tx4 + ty4*16, blk )
        }
        __syncthreads();
    }

    // store row sums
    sA16(ty, tx) = total;
    __syncthreads();

    partial = (blk == gridDim.x - 1 ? (n % NB_X) : 0);

    // sum up final total, y_blk, for row tx
    if(ty == 0 && (partial == 0 || tx < partial))
    {
        total                      = sA16(0, tx) + sA16(1, tx) + sA16(2, tx) + sA16(3, tx);
        workspace[blk * NB_X + tx] = total; // store at workspace( blk*NB_X + tx, blk )
    }
}
// end hemvn_kernel_upper_calc

/*****************************************************************************
    Upper case, sum up final results
    Each block sums one block row; each thread sums one row.

    On input (for 3 blocks):
                [ (A11*x1 + A12*x2 + A13*x3)     ---                 ---    ]
    workspace = [ (A12^H*x1)                   (A22*x2 + A23*x3)     ---    ]
                [ (A13^H*x1)                   (A23^H*x2)          (A33*x3) ]

    On output:
              [ (A11*x1 + A12*x2 + A13*x3)         ]
    y = alpha*[ (A12^H*x1) + (A22*x2 + A23*x3)     ] + beta*y
              [ (A13^H*x1) + (A23^H*x2) + (A33*x3) ]
*******************************************************************************/
template <rocblas_int NB_X, typename U, typename TPtr, typename W>
__launch_bounds__(NB_X) ROCBLAS_KERNEL
    void hemvn_kernel_upper_block_sum(rocblas_int    n,
                                      U              alpha_device_host,
                                      rocblas_stride stride_alpha,
                                      U              beta_device_host,
                                      rocblas_stride stride_beta,
                                      TPtr __restrict__ ya,
                                      ptrdiff_t      shifty,
                                      rocblas_int    incy,
                                      rocblas_stride stridey,
                                      W* __restrict__ workspace)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    auto* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    int tx      = threadIdx.x;
    int blk     = blockIdx.x;
    int blk_ind = blk * NB_X;
    int ind     = blk_ind + tx;
    if(!alpha)
    {
        if(ind < n)
            y[ind * incy] = beta ? beta * y[ind * incy] : 0;

        return;
    }

    // offset blocks * cols * batch
    workspace
        += size_t(hipGridDim_x) * n * hipBlockIdx_y; // workspace is workspace(0, 0, batch_count)

    // Don't write outside [0, ..., n)
    if(ind < n)
    {
        workspace += ind;
        W Ax = W{0};
        for(int j = 0; j <= blk; ++j)
        {
            Ax += workspace[0];
            workspace += n;
        }
        y[ind * incy] = beta ? beta * y[ind * incy] + alpha * Ax : alpha * Ax;
    }
}
// end hemvn_kernel_upper_block_sum_calc

/***************************************************************************/ /**
    Lower case, compute block multiply, workspace = A*x, for any size n:

                [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]   [ A11  A21^H  A31^H ]   [ x1 ]
    workspace = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ] = [ A21  A22    A32^H ] * [ x2 ]
                [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]   [ A31  A32    A33   ]   [ x3 ]

    Uses a 64x4 thread block.
    For     diagonal tiles, covers a 64x64 tile using three 32x32 tiles (plus one gets transposed).
    For off-diagonal tiles, covers a 64x64 tile using four  64x16 tiles.
    In both cases, each thread multiplies 4 elements.

    For rows past the bottom of the matrix, the A pointer is adjusted to be the
    last valid row of A, which multiple threads will read.
    Extra rows are ignored when saving results to workspace.
    Columns past the right edge are explicitly ignored when loading.
    x values past the bottom are set to zero, thus, extra columns are zeroed
    when multiplying.

    Previously:
                [ (A11*x1)       ---                                          ]
    workspace = [ (A21^H*x2)   (A21*x1 + A22*x2)     ---                      ]
                [ (A31^H*x3)   (A32^H*x3)          (A31*x1 + A32*x2 + A33*x3) ]
    which doesn't workspace as well because that has dimension blocks*NB by blocks,
    where blocks*NB >= n, and it can be that blocks*NB > lda, so it won't fit in
    lda*blocks space. This is why it used to need lwork = lda*(blocks + 1).
*******************************************************************************/

template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_lda,
          typename T>
ROCBLAS_KERNEL_ILF void hemvn_kernel_lower_calc(rocblas_int n,
                                                T           alpha,
                                                const T* __restrict__ A,
                                                T_lda lda,
                                                const T* __restrict__ x,
                                                ptrdiff_t incx,
                                                T* __restrict__ workspace)
{
    if(!alpha)
        return;
    // 64x4 thread block
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int blk     = blockIdx.x;
    const int blk_ind = NB_X * blk;
    const int td      = NB_X * ty + tx;

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    const int partial = (blk == gridDim.x - 1 ? (n % NB_X) : 0);

    T psum, psum_t;
    T total = T{0};

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
    __shared__ T sA[quarter_NB_X][NB_X + 3]; // only needs +2, use +3 to avoid bank conflict
    __shared__ T sx_blk[NB_X]; // for x[ blk ]
    __shared__ T sx_jj[NB_X]; // for x[ jj ], which cycles over all blocks left of diag

    T rA[4];
    T psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx) * incx; // x is x(blk_ind + tx)
    if(ty == 0)
    {
        if(partial == 0 || tx < partial)
        {
            sx_blk[tx] = x[0];
        }
        else
        {
            sx_blk[tx] = T{0};
        }
    }

    // --------------------
    // offset blocks * cols * batch
    workspace
        += size_t(hipGridDim_x) * n * hipBlockIdx_y; // workspace is workspace(0, 0, batch_count)

    // move to block row
    workspace += blk * n; // workspace is workspace(0, blk)

    A += blk_ind; // A is A(blk_ind, 0)
    A += ty2 * lda + tx2; // A is A(blk_ind + tx2, ty2)

    // move to 32x32 diag block
    A += blk_ind * lda; // A is A(blk_ind + tx2, blk_ind + ty2)

    // load 32x32 diag block A(blk_ind + 0:31, blk_ind + 0:31) into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 >= partial)
        {
            A = A - tx2
                + (partial
                   - 1); // A is A(blk_ind + partial-1, blk_ind + ty2), the bottom-most valid row
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 >= partial)
        {
            A = A + tx2 - (partial - 1); // A is A(blk_ind + tx2, blk_ind + ty2)
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

// symmetrize 32x32 diag block, copying lower to upper triangle,
// as four 32x8 sections in parallel:
// columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
#pragma unroll
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j < tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 32x32 diag block, then repeat steps from first diag block
    A += half_NB_X + half_NB_X * lda; // A is A(blk_ind + NB/2 + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 diag block A[block + 0:31, block + 0:31] into sA
    if(partial)
    {
        if(tx2 + half_NB_X >= partial)
        {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j + half_NB_X < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 + half_NB_X >= partial)
        {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

// symmetrize 32x32 diag block, copying lower to upper triangle
#pragma unroll
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j < tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to off-diag 32x32 block
    A -= half_NB_X * lda; // A is A(blk_ind + NB/2 + tx2, blk_ind + ty2)

    // load 32x32 block of A into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 + half_NB_X >= partial)
        {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 + half_NB_X >= partial)
        {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

    // multiply 32x32 block (below diag)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 + j * 8) * sx_blk[j * 8 + ty2];
    }
    //__syncthreads();  // no sync needed here

    // multiply transposed 32x32 block (above diag)
    psum_t = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum_t += (IS_HEMV ? conj(sA32(ty2 * 4 + j, tx2)) : sA32(ty2 * 4 + j, tx2))
                  * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial sums for non-transposed 32x32 block
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // store partial sums for transposed 32x32 block
    sA32(ty2, tx2) = psum_t;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to leftmost 64x64 block in block row, and
    // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
    A -= half_NB_X; // A is A(blk_ind + tx2, blk_ind + ty2)
    A -= blk_ind * lda; // A is A(blk_ind + tx2,           ty2)
    A -= ty2 * lda + tx2; // A is A(blk_ind, 0)
    A += 4 * ty * lda + tx; // A is A(blk_ind + tx, 4*ty)

    if(partial && tx >= partial)
    {
        A = A - tx + (partial - 1); // A is A(blk_ind + partial-1, 4*ty), the bottom-most valid row
    }

    x -= blk_ind * incx; // x is x(tx)

    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;

    // cycle over blocks jj left of diagonal, in block row blk
    for(int jj = 0; jj < blk; ++jj)
    {
        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        // since this block is left of diagonal, x must have all NB rows
        if(ty == 0)
        {
            sx_jj[tx] = x[jj * NB_X * incx];
        }
        __syncthreads();

        for(int k = 0; k < 4; k++)
        {
// load 64x16 block of A into rA, 4 elements per thread,
// as four 64x4 sections in parallel:
// columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
// since this block is left of diagonal, it has all NB columns,
// and block of x must have all NB rows.
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                rA[j] = A[j * lda];
            }

// 1) multiply 64x16 block A_{blk,jj} * x_jj
//    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
// 2) multiply transposed 16x64 block A_{blk,jj}^H * x_blk,
//    storing each product Aji*xi to sA(j,i)
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                total
                    += rA[j] * sx_jj[quarter_NB_X * k + ty * 4 + j]; // y_blk = A_{blk,jj}   * x_jj
                sA16(ty * 4 + j, tx)
                    = (IS_HEMV ? conj(rA[j]) : rA[j]) * sx_blk[tx]; // y_jj  = A_{blk,jj}^H * x_blk
            }
            __syncthreads();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            psum_t = T{0};
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                psum_t += sA16(tx4, ty4 * 4 + j);
            }
            __syncthreads();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;

            // move right to next 64x16 block
            A += lda
                 * quarter_NB_X; // A is A(blk_ind + tx#, jj*NB_x + (k+1)*NB_X/4 + 4*ty), # tx or partial
        }
// already at next 64x64 block
// A is A(blk_ind + tx#, (jj+1)*NB_x + 4*ty), # tx or partial

// store partial row sums of transposed result, y_jj
#pragma unroll
        for(int k = 0; k < 4; k++)
        {
            sA16(tx4, ty4 + quarter_NB_X * k) = psums_t[k];
        }
        __syncthreads();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        // since this is the transposed block above the diagonal, it must have all NB rows
        if(ty4 < 4)
        {
            int ty4_nb4 = ty4 * quarter_NB_X;
            psum_t      = sA16(tx4, 0 + ty4_nb4) + sA16(tx4, 1 + ty4_nb4) + sA16(tx4, 2 + ty4_nb4)
                     + sA16(tx4, 3 + ty4_nb4) + sA16(tx4, 4 + ty4_nb4) + sA16(tx4, 5 + ty4_nb4)
                     + sA16(tx4, 6 + ty4_nb4) + sA16(tx4, 7 + ty4_nb4) + sA16(tx4, 8 + ty4_nb4)
                     + sA16(tx4, 9 + ty4_nb4) + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                     + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4) + sA16(tx4, 14 + ty4_nb4)
                     + sA16(tx4, 15 + ty4_nb4);
            workspace[jj * NB_X + tx4 + ty4_nb4]
                = psum_t; // store at workspace( jj*NB_X + tx4 + ty4*16, blk )
        }
        __syncthreads();
    }

    // store row sums
    sA16(ty, tx) = total;
    __syncthreads();

    // sum up final total, y_blk, for row tx
    if(ty == 0 && (partial == 0 || tx < partial))
    {
        total                      = sA16(0, tx) + sA16(1, tx) + sA16(2, tx) + sA16(3, tx);
        workspace[blk * NB_X + tx] = total; // store at workspace( blk*NB_X + tx, blk )
    }
}
// end hemvn_kernel_lower_calc

/*****************************************************************************
    UPLO = rocblas_fill_lower, sum up final results
    Each block sums one block row; each thread sums one row.

    On input (for 3 blocks):
                [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]
    workspace = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ]
                [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]


    On output:
              [ (A11*x1) + (A21^H*x2) + (A31^H*x3) ]
    y = alpha*[ (A21*x1 + A22*x2)     + (A32^H*x3) ] + beta*y
              [ (A21*x1 + A22*x2 + A33*x3)         ]
*******************************************************************************/
template <rocblas_int NB_X, typename U, typename TPtr, typename W>
__launch_bounds__(NB_X) ROCBLAS_KERNEL
    void hemvn_kernel_lower_block_sum(rocblas_int    n,
                                      U              alpha_device_host,
                                      rocblas_stride stride_alpha,
                                      U              beta_device_host,
                                      rocblas_stride stride_beta,
                                      TPtr __restrict__ ya,
                                      ptrdiff_t      shifty,
                                      rocblas_int    incy,
                                      rocblas_stride stridey,
                                      W* __restrict__ workspace)
{
    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    auto* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    int tx      = threadIdx.x;
    int blk     = blockIdx.x;
    int blk_ind = blk * NB_X;
    int ind     = blk_ind + tx;
    int blocks  = gridDim.x;

    if(!alpha)
    {
        if(ind < n)
            y[ind * incy] = beta ? beta * y[ind * incy] : 0;

        return;
    }

    // offset blocks * cols * batch
    workspace
        += size_t(hipGridDim_x) * n * hipBlockIdx_y; // workspace is workspace(0, 0, batch_count)

    // Don't write outside [0, ..., n)
    if(ind < n)
    {
        workspace += ind + blk * n;
        W Ax = W{0};
        for(int j = blk; j < blocks; ++j)
        {
            Ax += workspace[0];
            workspace += n;
        }
        y[ind * incy] = beta ? beta * y[ind * incy] + alpha * Ax : alpha * Ax;
    }
}
// end hemvn_kernel_lower_block_sum_calc

template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int NB_Y,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_lda,
          typename U,
          typename V,
          typename W>
__launch_bounds__(NB_X* NB_Y) ROCBLAS_KERNEL void hemvn_kernel_upper(rocblas_int n,
                                                                     U           alpha_device_host,
                                                                     rocblas_stride stride_alpha,
                                                                     V              Aa,
                                                                     ptrdiff_t      shifta,
                                                                     T_lda          lda,
                                                                     rocblas_stride strideA,
                                                                     V              xa,
                                                                     ptrdiff_t      shiftx,
                                                                     rocblas_int    incx,
                                                                     rocblas_stride stridex,
                                                                     U beta_device_host,
                                                                     rocblas_stride stride_beta,
                                                                     W              workspace)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    if(NB_X * NB_Y != num_threads)
        return;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const auto* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const auto* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    hemvn_kernel_upper_calc<IS_HEMV, NB_X, bank_shift, half_NB_X, quarter_NB_X, T_lda>(
        n, alpha, A, lda, x, incx, workspace);
}

template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int NB_Y,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_lda,
          typename U,
          typename V,
          typename W>
__launch_bounds__(NB_X* NB_Y) ROCBLAS_KERNEL void hemvn_kernel_lower(rocblas_int n,
                                                                     U           alpha_device_host,
                                                                     rocblas_stride stride_alpha,
                                                                     V              Aa,
                                                                     ptrdiff_t      shifta,
                                                                     T_lda          lda,
                                                                     rocblas_stride strideA,
                                                                     V              xa,
                                                                     ptrdiff_t      shiftx,
                                                                     rocblas_int    incx,
                                                                     rocblas_stride stridex,
                                                                     U beta_device_host,
                                                                     rocblas_stride stride_beta,
                                                                     W              workspace)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    if(NB_X * NB_Y != num_threads)
        return;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const auto* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const auto* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    hemvn_kernel_lower_calc<IS_HEMV, NB_X, bank_shift, half_NB_X, quarter_NB_X, T_lda>(
        n, alpha, A, lda, x, incx, workspace);
}

/**
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  *  Note stride_alpha and stride_beta are only used AND only tested by rocSOLVER
  *  These strided scalar fetches are only supported for device_ptr mode
  */
template <bool IS_HEMV, typename U, typename V, typename TPtr, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_symv_template(rocblas_handle handle,
                                        rocblas_fill   uplo,
                                        rocblas_int    n,
                                        const U*       alpha,
                                        rocblas_stride stride_alpha,
                                        V              A,
                                        rocblas_int    offseta,
                                        rocblas_int    lda,
                                        rocblas_stride strideA,
                                        V              x,
                                        rocblas_int    offsetx,
                                        rocblas_int    incx,
                                        rocblas_stride stridex,
                                        const U*       beta,
                                        rocblas_stride stride_beta,
                                        TPtr           y,
                                        rocblas_int    offsety,
                                        rocblas_int    incy,
                                        rocblas_stride stridey,
                                        rocblas_int    batch_count,
                                        W              workspace)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    bool i64_indices = n * size_t(lda) > std::numeric_limits<rocblas_int>::max();

    static constexpr int HEMV_DIM_X         = rocblas_hemv_DIM_X();
    static constexpr int HEMV_DIM_Y         = 4;
    static constexpr int bank_shift         = 33;
    static constexpr int half_HEMV_DIM_X    = 32;
    static constexpr int quarter_HEMV_DIM_X = 16;
    rocblas_int          blocks             = (n - 1) / (HEMV_DIM_X) + 1;

    dim3 hemv_grid(blocks, batch_count);
    dim3 hemv_threads(HEMV_DIM_X, HEMV_DIM_Y);
    dim3 hemv_threads_sum(HEMV_DIM_X);

#define hemv_kernel_KARGS(alpha_, beta_)                                                           \
    hemv_grid, hemv_threads, 0, rocblas_stream, n, alpha_, stride_alpha, A, offseta, lda, strideA, \
        x, shiftx, incx, stridex, beta_, stride_beta, workspace

#define hemv_kernel_sum_KARGS(alpha_, beta_)                                                     \
    hemv_grid, hemv_threads_sum, 0, rocblas_stream, n, alpha_, stride_alpha, beta_, stride_beta, \
        y, shifty, incy, stridey, workspace

    if(uplo == rocblas_fill_upper)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            if(i64_indices)
            {
                hipLaunchKernelGGL((hemvn_kernel_upper<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       size_t>),
                                   hemv_kernel_KARGS(alpha, beta));

                hipLaunchKernelGGL((hemvn_kernel_upper_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(alpha, beta));
            }
            else
            {
                hipLaunchKernelGGL((hemvn_kernel_upper<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       rocblas_int>),
                                   hemv_kernel_KARGS(alpha, beta));

                hipLaunchKernelGGL((hemvn_kernel_upper_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(alpha, beta));
            }
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;
            if(i64_indices)
            {
                hipLaunchKernelGGL((hemvn_kernel_upper<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       size_t>),
                                   hemv_kernel_KARGS(*alpha, *beta));

                hipLaunchKernelGGL((hemvn_kernel_upper_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(*alpha, *beta));
            }
            else
            {
                hipLaunchKernelGGL((hemvn_kernel_upper<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       rocblas_int>),
                                   hemv_kernel_KARGS(*alpha, *beta));

                hipLaunchKernelGGL((hemvn_kernel_upper_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(*alpha, *beta));
            }
        }
    }
    else
    {
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            if(i64_indices)
            {
                hipLaunchKernelGGL((hemvn_kernel_lower<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       size_t>),
                                   hemv_kernel_KARGS(alpha, beta));

                hipLaunchKernelGGL((hemvn_kernel_lower_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(alpha, beta));
            }
            else
            {
                hipLaunchKernelGGL((hemvn_kernel_lower<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       rocblas_int>),
                                   hemv_kernel_KARGS(alpha, beta));

                hipLaunchKernelGGL((hemvn_kernel_lower_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(alpha, beta));
            }
        }
        else
        {
            if(!*alpha && *beta == 1)
                return rocblas_status_success;
            if(i64_indices)
            {
                hipLaunchKernelGGL((hemvn_kernel_lower<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       size_t>),
                                   hemv_kernel_KARGS(*alpha, *beta));

                hipLaunchKernelGGL((hemvn_kernel_lower_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(*alpha, *beta));
            }
            else
            {
                hipLaunchKernelGGL((hemvn_kernel_lower<IS_HEMV,
                                                       HEMV_DIM_X,
                                                       HEMV_DIM_Y,
                                                       bank_shift,
                                                       half_HEMV_DIM_X,
                                                       quarter_HEMV_DIM_X,
                                                       rocblas_int>),
                                   hemv_kernel_KARGS(*alpha, *beta));

                hipLaunchKernelGGL((hemvn_kernel_lower_block_sum<HEMV_DIM_X>),
                                   hemv_kernel_sum_KARGS(*alpha, *beta));
            }
        }
    }
    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_hemv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_int    offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}
