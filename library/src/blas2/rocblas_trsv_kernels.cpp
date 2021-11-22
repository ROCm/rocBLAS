/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "check_numerics_vector.hpp"
#include "rocblas_trsv.hpp"

// Copyright 2014-2021, The Science and Technology Facilities Council (STFC)
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the STFC nor the names of its contributors may be
//       used to endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL STFC BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Solve the A_21 section during A inversion
template <typename T,
          rocblas_int N,
          rocblas_int LDA,
          rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        UNIT>
void ROCBLAS_KERNEL_ILF rocblas_invert_solve_A21(const T* const __restrict__ A11,
                                                 T* const __restrict__ A21,
                                                 const T* __restrict__ A22,
                                                 T* const __restrict__ sx)
{
    const rocblas_int tid      = DIM_X * threadIdx.y + threadIdx.x;
    const rocblas_int ntid     = DIM_X * DIM_Y;
    const rocblas_int tx       = tid % N;
    const rocblas_int ty       = tid / N;
    const rocblas_int col_span = ntid / N;

    // break A_21 into sub-blocks
    for(rocblas_int i = 0; i < N; i += col_span)
    {
        rocblas_int col = i + ty;

        // skip all calculations if out-of-bounds
        // can't return early/break because of syncthreads
        bool skip = col >= N;

        T val = 0;
        if(!skip)
        {
            // Multiplying sub-matrices
            for(rocblas_int j = i; j < N; j++)
            {
                if(j + ty < N)
                    val += A21[(j + ty) * LDA + tx] * A11[col * LDA + j + ty];
            }
            val = -val;
        }

        // Update with solved value
        for(rocblas_int j = 0; j < N; j++)
        {
            if(tx == j && !skip)
            {
                if(!UNIT)
                    val *= A22[j * LDA + j];
                sx[ty] = val;
            }
            __syncthreads();
            if(tx > j && !skip)
            {
                val += A22[j * LDA + tx] * sx[ty];
            }
            __syncthreads();
        }

        // Store solved value
        if(!skip)
            A21[col * LDA + tx] = -val;
        __syncthreads();
    }
}

// Solve the A_12 section during A inversion
template <typename T,
          rocblas_int N,
          rocblas_int LDA,
          rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        UNIT>
void ROCBLAS_KERNEL_ILF rocblas_invert_solve_A12(const T* const __restrict__ A11,
                                                 T* const __restrict__ A12,
                                                 const T* __restrict__ A22,
                                                 T* const __restrict__ sx)
{
    const rocblas_int tid      = DIM_X * threadIdx.y + threadIdx.x;
    const rocblas_int ntid     = DIM_X * DIM_Y;
    const rocblas_int tx       = tid % N;
    const rocblas_int ty       = tid / N;
    const rocblas_int col_span = ntid / N;

    // break A_12 into sub-blocks
    for(rocblas_int i = N - 1; i >= 0; i -= col_span)
    {
        rocblas_int col  = i - ty;
        bool        skip = col < 0;
        T           val  = 0;
        if(!skip)
        {
            for(rocblas_int j = 0; j < N; j++)
            {
                if(j <= col)
                {
                    val += A12[(j)*LDA + tx] * A22[(col)*LDA + (j)];
                }
            }
        }

        // Substitution method to deal with A11 since it isn't yet
        // inverted (but the diagonal is)
        for(rocblas_int j = N - 1; j >= 0; j--)
        {
            if(tx == j && !skip)
            {
                if(!UNIT)
                    val *= A11[j * LDA + j];
                sx[ty] = -val;
            }

            __syncthreads();

            if(tx < j && !skip)
            {
                val -= A11[j * LDA + tx] * sx[ty];
            }
            __syncthreads();
        }

        // Store solved value
        if(!skip)
            A12[col * LDA + tx] = val;

        __syncthreads();
    }
}

template <typename T, rocblas_int LDA>
void ROCBLAS_KERNEL_ILF rocblas_trsv_transpose(const rocblas_int n,
                                               const T* const __restrict__ A,
                                               T* const __restrict__ at)
{
    if(threadIdx.y == 0 && threadIdx.x < n)
    {
        for(rocblas_int i = 0; i < n; i++)
        {
            at[i * LDA + threadIdx.x] = A[threadIdx.x * LDA + i];
        }
    }
}

template <rocblas_int n>
static constexpr bool equals_two = false;

template <>
ROCBLAS_CLANG_STATIC constexpr bool equals_two<2> = true;

// Invert a 2x2 triangular section of A
template <typename T,
          rocblas_int N,
          rocblas_int LDA,
          rocblas_int threadsx,
          rocblas_int threadsy,
          bool        UNIT,
          bool        TRANS,
          std::enable_if_t<equals_two<N>, rocblas_int> = 0>
void ROCBLAS_KERNEL_ILF rocblas_trsv_invert(T* const __restrict__ A, T* const __restrict__ sx)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        if(UNIT)
        {
            A[0]       = 1;
            A[LDA + 1] = 1;
        }
        else
        {
            // Diagonal is already stored as 1 / A[x], so A[0] and A[LDA + 1] are already done
            // A[1] = -A[1] * (1 / A[0]) * (1 / A[LDA + 1])
            A[1] = A[1] * (A[0] * A[LDA + 1]);
        }
        if(TRANS)
        {
            // For the transpose case, we can simply copy over the solved A[1]
            // to the appropriate place
            A[LDA] = A[1];
        }
    }
}

// Recursive invert to solve A^-1
template <typename T,
          rocblas_int N,
          rocblas_int LDA,
          rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        UNIT,
          bool        TRANS,
          std::enable_if_t<!equals_two<N>, rocblas_int> = 0>
void ROCBLAS_KERNEL_ILF rocblas_trsv_invert(T* const __restrict__ A, T* const __restrict__ sx)
{
    // A is broken down as:
    // A = [ A_11 0    ]
    //     [ A_21 A_22 ]

    // A^-1 can be solved as:
    // A^-1 = [ (A_11^-1)                   (0)     ]
    //        [ (-A_21 * A_11^-1 * A_22^-1) (A_22^-1)]

    // Invert A_11 section by breaking into smaller and smaller pieces
    rocblas_trsv_invert<T, N / 2, LDA, DIM_X, DIM_Y, UNIT, TRANS>(A, sx);
    __syncthreads();

    // Solve A_21 section
    rocblas_invert_solve_A21<T, N / 2, LDA, DIM_X, DIM_Y, UNIT>(
        A, &A[N / 2], &A[(LDA + 1) * N / 2], sx);

    if(TRANS)
    {
        __syncthreads();
        rocblas_trsv_transpose<T, LDA>(N / 2, &A[N / 2], &A[(N / 2) * LDA]);
    }
    __syncthreads();

    // Invert A_22 section by breaking into smaller and smaller pieces
    rocblas_trsv_invert<T, N / 2, LDA, DIM_X, DIM_Y, UNIT, TRANS>(&A[(LDA + 1) * N / 2], sx);
}

// Invert a 2x2 triangular section of A
template <typename T,
          rocblas_int N,
          rocblas_int LDA,
          rocblas_int threadsx,
          rocblas_int threadsy,
          bool        UNIT,
          bool        TRANS,
          std::enable_if_t<equals_two<N>, rocblas_int> = 0>
void ROCBLAS_KERNEL_ILF rocblas_trsv_invert_upper(T* const __restrict__ A, T* const __restrict__ sx)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        if(UNIT)
        {
            A[0]       = 1;
            A[LDA + 1] = 1;
        }
        else
        {
            // Diagonal is already stored as 1 / A[x], so A[0] and A[LDA + 1] are already done
            // A[1] = -A[1] * (1 / A[0]) * (1 / A[LDA + 1])
            A[LDA] = A[LDA] * (A[0] * A[LDA + 1]);
        }
        if(TRANS)
        {
            // For the transpose case, we can simply copy over the solved A[1]
            // to the appropriate place
            A[1] = A[LDA];
        }
    }
}

// Recursive invert to solve A^-1
template <typename T,
          rocblas_int N,
          rocblas_int LDA,
          rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        UNIT,
          bool        TRANS,
          std::enable_if_t<!equals_two<N>, rocblas_int> = 0>
void ROCBLAS_KERNEL_ILF rocblas_trsv_invert_upper(T* const __restrict__ A, T* const __restrict__ sx)
{
    // A is broken down as:
    // A = [ A_11 A_12 ]
    //     [ 0    A_22 ]

    // A^-1 can be solved as:
    // A^-1 = [ (A_11^-1) (A_11^-1 * -A_12 * A_22^-1) ]
    //        [ (0)       (A_22^-1)                   ]

    // Invert A_22 section by breaking into smaller and smaller pieces
    rocblas_trsv_invert_upper<T, N / 2, LDA, DIM_X, DIM_Y, UNIT, TRANS>(&A[(LDA + 1) * N / 2], sx);
    __syncthreads();
    __threadfence();

    // Solve A_21 section
    //                                             A11, A12,             A22
    rocblas_invert_solve_A12<T, N / 2, LDA, DIM_X, DIM_Y, UNIT>(
        A, &A[(N / 2) * LDA], &A[(LDA + 1) * N / 2], sx);

    if(TRANS)
    {
        __syncthreads();
        rocblas_trsv_transpose<T, LDA>(N / 2, &A[(N / 2) * LDA], &A[(N / 2)]);
    }
    __syncthreads();

    // Invert A_11 section by breaking into smaller and smaller pieces
    rocblas_trsv_invert_upper<T, N / 2, LDA, DIM_X, DIM_Y, UNIT, TRANS>(A, sx);
}

template <typename T, rocblas_int N, rocblas_int DIM_Y, bool UPPER>
void ROCBLAS_KERNEL_ILF rocblas_trsv_block_solve_inverse(const T* __restrict__ Ainv,
                                                         T* __restrict__ sx,
                                                         T& val,
                                                         T* const __restrict__ sum)
{
    Ainv += threadIdx.y * N + threadIdx.x;
    sx += threadIdx.y;

    if(threadIdx.y == 0)
    {
        sx[threadIdx.x] = val;
    }

    __syncthreads();

    if(!UPPER)
    {
        // Multiply Ainv's threadIdx.x row with x
        val = 0;
        for(rocblas_int i = 0; i < N; i += DIM_Y)
        {
            if(threadIdx.x >= threadIdx.y + i)
                val += Ainv[i * N] * sx[i];
        }
        sum[threadIdx.y * N + threadIdx.x] = val;

        __syncthreads();

        // Store into val
        if(threadIdx.y == 0)
        {
            for(rocblas_int i = 1; i < DIM_Y; i++)
            {
                val += sum[i * N + threadIdx.x];
            }
        }
    }
    else
    {
        val = 0;
        for(rocblas_int i = 0; i < N; i += DIM_Y)
        {
            if(threadIdx.x <= i + threadIdx.y)
            {
                val += Ainv[i * N] * sx[i];
            }
        }

        sum[threadIdx.y * N + threadIdx.x] = val;
        __syncthreads();

        if(threadIdx.y == 0)
        {
            for(rocblas_int i = 1; i < DIM_Y; i++)
                val += sum[i * N + threadIdx.x];
        }
    }
}

template <rocblas_int BLOCK, bool UNIT, typename T>
void ROCBLAS_KERNEL_ILF rocblas_trsv_block_solve_lower(const T* __restrict__ A,
                                                       rocblas_int lda,
                                                       T&          val)
{
    T __shared__ xs;

    // Iterate forwards
    for(rocblas_int i = 0; i < BLOCK; i++)
    {
        // Solve current element
        if(threadIdx.x == i && threadIdx.y == 0)
        {
            if(!UNIT)
                val *= A[i * lda + i];
            xs = val;
        }

        __syncthreads();

        // Update future elements with solved one
        if(threadIdx.x > i && threadIdx.y == 0)
        {
            val += A[i * lda + threadIdx.x] * xs;
        }

        __syncthreads();
    }
}

template <rocblas_int BLOCK, bool UNIT, typename T>
void ROCBLAS_KERNEL_ILF rocblas_trsv_block_solve_upper(const T* __restrict__ A,
                                                       rocblas_int lda,
                                                       T&          val)
{
    T __shared__ xs;

    for(rocblas_int i = BLOCK - 1; i >= 0; i--)
    {
        // Solve current element
        if(threadIdx.x == i && threadIdx.y == 0)
        {
            if(!UNIT)
                val *= A[i * lda + i];
            xs = val;
        }

        __syncthreads();

        // Update future elements with solved one
        if(threadIdx.x < i && threadIdx.y == 0)
        {
            val += A[i * lda + threadIdx.x] * xs;
        }

        __syncthreads();
    }
}

ROCBLAS_KERNEL static __launch_bounds__(1) void rocblas_trsv_init(rocblas_int* w_completed_sec)
{
    // The last block section which has been completed (for each batch)
    w_completed_sec[blockIdx.x] = -1;
}

// If defined, INV_AFTER allows for a block-inversion technique while waiting for data
// from the previous block.
// INV_AFTER defines how many block iterations to do using substitution before
// having the current block perform an inversion of it's block so we can do a
// multiply (essentially a trmv) instead of a solve
#define INV_AFTER 5

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        LOWER,
          bool        TRANS,
          bool        CONJ,
          bool        UNIT,
          typename T,
          typename ATYPE,
          typename XTYPE>
ROCBLAS_KERNEL
    __launch_bounds__(DIM_X* DIM_Y) void rocblas_trsv_device(rocblas_int    m,
                                                             ATYPE          dA,
                                                             ptrdiff_t      offset_A,
                                                             rocblas_int    lda,
                                                             rocblas_stride stride_A,
                                                             XTYPE          dx,
                                                             ptrdiff_t      offset_x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stride_x,
                                                             rocblas_int*   w_completed_sec)
{
    // If we need to start at the bottom and work upwards (backwards substitution)
    constexpr bool backwards_sub = (!LOWER && !TRANS) || (LOWER && TRANS);

    // Load appropriate pointers
    const rocblas_int batchid = blockIdx.y;
    auto* __restrict__ A      = load_ptr_batch(dA, batchid, offset_A, stride_A);
    auto* __restrict__ x      = load_ptr_batch(dx, batchid, offset_x, stride_x);

    // Storing the updated sum of x values, so we can have more than 1 thread working on each val
    T __shared__ sum[DIM_X * DIM_Y];

    // Shared memory for diagonal block of A for solve
    T __shared__ sAdiag[DIM_X * DIM_X];

    // Shared memory to access block portion of x
    T __shared__ sx[DIM_X];

    // Storing a single DIM_X * DIM_X block in registers.
    // Each thread stores DIM_X / DIM_Y elements in the same row
    T sAoff[DIM_X / DIM_Y];

    const rocblas_int num_blocks = gridDim.x;
    const ptrdiff_t   tid        = blockDim.x * threadIdx.y + threadIdx.x;
    const rocblas_int tx         = threadIdx.x;
    const rocblas_int ty         = threadIdx.y;

    // Assign to register row in each thread
    rocblas_int block_row = backwards_sub ? num_blocks - 1 - blockIdx.x : blockIdx.x;

    // If problem is not divisible into DIM_X sized sections, the last block row
    // will be smaller and must be handled differently
    const rocblas_int remainder        = m % DIM_X;
    const bool        row_is_remainder = ((m - 1) / DIM_X == block_row && remainder != 0);

    // Store square block of A beside triangular part (if not first row)
    const bool first_row = backwards_sub ? block_row == num_blocks - 1 : block_row == 0;
    if(!first_row)
    {
        const rocblas_int block_col = backwards_sub ? block_row + 1 : block_row - 1;
        const rocblas_int local_col = TRANS ? block_row * DIM_X + tx : block_col * DIM_X + ty;
        const rocblas_int local_row = TRANS ? block_col * DIM_X + ty : block_row * DIM_X + tx;
        const size_t      A_idx     = (local_row) + (local_col)*size_t(lda);

        for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
        {
            const size_t i_idx = TRANS ? i : i * size_t(lda);

            if(TRANS ? (local_row + i < m && local_col < m) : (local_row < m && local_col + i < m))
                sAoff[i / DIM_Y] = A[A_idx + i_idx];
            else
                sAoff[i / DIM_Y] = 0.0;
        }
    }

    // Storing diagonal block of A into shared memory for subtitution solve
#ifdef INV_AFTER
    bool cache_transpose = (TRANS && LOWER && num_blocks - 1 - block_row < INV_AFTER)
                           || (TRANS && !LOWER && block_row < INV_AFTER)
                           || (TRANS && row_is_remainder);
#else
    bool cache_transpose = TRANS; // works for ALL without inversion method
#endif
    if(!row_is_remainder)
    {
        rocblas_int row = tx;
        for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
        {
            const rocblas_int col    = ty + i;
            const rocblas_int sA_idx = cache_transpose ? col + DIM_X * row : col * DIM_X + row;
            const size_t      A_idx
                = (block_row * DIM_X * size_t(lda) + block_row * DIM_X) + col * size_t(lda) + row;
            const rocblas_int total_col = block_row * DIM_X + col;
            const rocblas_int total_row = block_row * DIM_X + row;

            if((row > col && LOWER) || (col > row && !LOWER))
            {
                sAdiag[sA_idx] = CONJ ? -conj(A[A_idx]) : -A[A_idx];
            }
            else if(!UNIT && row == col)
            {
                // Dividing here so we can just multiply later.
                sAdiag[sA_idx] = 1.0 / (CONJ ? conj(A[A_idx]) : A[A_idx]);
            }
            else if(col < DIM_X && row < DIM_X) // In off-triangular portion - set to 0
            {
                sAdiag[sA_idx] = 0.0;
            }
        }
    }
    else // remainder of a block
    {
        rocblas_int row = tx;
        for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
        {
            const rocblas_int col    = ty + i;
            const rocblas_int sA_idx = cache_transpose ? col + DIM_X * row : col * DIM_X + row;
            const size_t      A_idx
                = (block_row * DIM_X * size_t(lda) + block_row * DIM_X) + col * size_t(lda) + row;
            const rocblas_int total_col = block_row * DIM_X + col;
            const rocblas_int total_row = block_row * DIM_X + row;
            if(((row > col && LOWER) || (col > row && !LOWER)) && row < remainder
               && col < remainder)
            {
                sAdiag[sA_idx] = CONJ ? -conj(A[A_idx]) : -A[A_idx];
            }
            else if(!UNIT && row == col && row < remainder)
            {
                // Dividing here so we can just multiply later.
                sAdiag[sA_idx] = 1.0 / (CONJ ? conj(A[A_idx]) : A[A_idx]);
            }
            else if(col < DIM_X
                    && row < DIM_X) // In off-triangular portion or past end of remainder
            {
                sAdiag[sA_idx] = 0.0;
            }
        }
    }
    __syncthreads();

#ifdef INV_AFTER
    if(((block_row >= INV_AFTER && !backwards_sub)
        || (num_blocks - 1 - block_row >= INV_AFTER && backwards_sub))
       && !row_is_remainder)
    {
        if(LOWER)
            rocblas_trsv_invert<T, DIM_X, DIM_X, DIM_X, DIM_Y, UNIT, TRANS>(sAdiag, sum);
        else
            rocblas_trsv_invert_upper<T, DIM_X, DIM_X, DIM_X, DIM_Y, UNIT, TRANS>(sAdiag, sum);
    }
#endif
    __syncthreads();

    // Store relevant x value into register
    T val = 0;
    if(ty == 0)
    {
        if(!row_is_remainder || tx < remainder)
        {
            val = -x[(block_row * DIM_X + tx) * incx];
        }
    }

    // Once previously solved block is ready, apply this to other square blocks
    rocblas_int       col_done = -1;
    const rocblas_int iters    = backwards_sub ? num_blocks - 1 - block_row : block_row;
    for(rocblas_int block_iter = 0; block_iter < iters; block_iter++)
    {
        // For backwards substitution, we start at the bottom and propogate upwards, else we go top-to-bottom
        const rocblas_int block_col = backwards_sub ? (num_blocks - 1 - block_iter) : block_iter;

        const rocblas_int local_col = TRANS ? block_row * DIM_X + tx : block_col * DIM_X + ty;
        const rocblas_int local_row = TRANS ? block_col * DIM_X + ty : block_row * DIM_X + tx;
        const size_t      A_idx     = local_col * size_t(lda) + local_row;
        const rocblas_int x_idx     = (block_col * DIM_X) * incx;

        if(tid == 0)
        {
            // Wait until the previous column is done. Use global memory to
            // update when ready.
            if(col_done < block_iter)
            {
                while(w_completed_sec[batchid] < block_iter)
                    __threadfence();
                col_done = w_completed_sec[batchid];
            }
        }

        // Few intermittent failures without this. Needed to wait for updated x values, I guess?
        __threadfence();
        __syncthreads();

        // Store x val (of previous block) into shared memory
        if(tid < DIM_X)
        {
            if(block_col * DIM_X + tid >= m)
                sx[tid] = 0.0;
            else
                sx[tid] = x[x_idx + tid * incx];
        }

        __syncthreads();

        // Update val with result of previous block
        for(rocblas_int i = 0; i < DIM_X; i += DIM_Y)
        {
            // Use shared memory if previous col since we cached this earlier
            const size_t i_idx = TRANS ? i : i * size_t(lda);
            const bool   cached
                = !first_row
                  && (backwards_sub ? block_col == block_row + 1 : block_col == block_row - 1);
            if(TRANS ? (local_row + i < m && local_col < m) : (local_row < m && local_col + i < m))
            {
                auto A_val = cached ? sAoff[i / DIM_Y] : A[A_idx + i_idx];
                if(CONJ)
                    A_val = conj(A_val);
                val += A_val * sx[i + ty];
            }
        }
    }

    // Add "solved" x values into shared memory to be summed further
    sum[ty * DIM_X + tx] = val;
    __syncthreads();

    if(ty == 0)
    {
        // Sum DIM_Y elements into single val
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            val += sum[i * DIM_X + tx];
        }
        val = -val;

        if(row_is_remainder && tx >= remainder)
            val = 0.0; // zero out out-of-bounds
    }

    // Solve the current block.
    // It's important that we're very efficient here, as other blocks are
    // likely just waiting for the result of this block.
#ifdef INV_AFTER
    if(((block_row >= INV_AFTER && !backwards_sub)
        || (num_blocks - 1 - block_row >= INV_AFTER && backwards_sub))
       && !row_is_remainder)
    {
        rocblas_trsv_block_solve_inverse<T, DIM_X, DIM_Y, backwards_sub>(sAdiag, sx, val, sum);

        if(!row_is_remainder || tx < remainder)
        {
            if(ty == 0)
            {
                x[(block_row * DIM_X + tid) * incx] = val;
            }
        }
    }
    else // same as without inversion
    {
        // Solve the diagonal block
        if(backwards_sub)
            rocblas_trsv_block_solve_upper<DIM_X, UNIT>(sAdiag, DIM_X, val);
        else
            rocblas_trsv_block_solve_lower<DIM_X, UNIT>(sAdiag, DIM_X, val);

        // Store solved value into x
        if(!row_is_remainder || tx < remainder)
            if(ty == 0)
                x[(block_row * DIM_X + tid) * incx] = val;
    }
#else
    // Solve the diagonal block
    if(backwards_sub)
        rocblas_trsv_block_solve_upper<DIM_X, UNIT>(sAdiag, DIM_X, val);
    else
        rocblas_trsv_block_solve_lower<DIM_X, UNIT>(sAdiag, DIM_X, val);

    // Store solved value into x
    if(!row_is_remainder || tx < remainder)
        if(ty == 0)
            x[(block_row * DIM_X + tid) * incx] = val;
#endif

    // ensure solved x values are saved
    __threadfence();

    // next column is ready
    // don't need an atomic op here since there should only
    // be one block for each batch here at once
    __syncthreads(); // for windows instability
    if(tid == 0)
        w_completed_sec[batchid]++;

    __threadfence();
}

template <rocblas_int DIM_X, typename T, typename ATYPE, typename XTYPE>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsv_substitution_template(rocblas_handle    handle,
                                                rocblas_fill      uplo,
                                                rocblas_operation transA,
                                                rocblas_diagonal  diag,
                                                rocblas_int       m,
                                                ATYPE             dA,
                                                ptrdiff_t         offset_A,
                                                rocblas_int       lda,
                                                rocblas_stride    stride_A,
                                                XTYPE             dx,
                                                ptrdiff_t         offset_x,
                                                rocblas_int       incx,
                                                rocblas_stride    stride_x,
                                                rocblas_int       batch_count,
                                                rocblas_int*      w_completed_sec)
{
    if(batch_count == 0)
        return rocblas_status_success;

    // Temporarily change the thread's default device ID to the handle's device ID
    // cppcheck-suppress unreadVariable
    auto saved_device_id = handle->push_device_id();

    offset_x = incx < 0 ? offset_x + ptrdiff_t(incx) * (1 - m) : offset_x;

    constexpr rocblas_int DIM_Y  = 4;
    rocblas_int           blocks = (m + DIM_X - 1) / DIM_X;
    dim3                  threads(DIM_X, DIM_Y, 1);
    dim3                  grid(blocks, batch_count);

    // Initialize global variables
    hipLaunchKernelGGL(
        rocblas_trsv_init, dim3(batch_count), dim3(1), 0, handle->get_stream(), w_completed_sec);

#define TRSV_TEMPLATE_PARAMS                                                                    \
    grid, threads, 0, handle->get_stream(), m, dA, offset_A, lda, stride_A, dx, offset_x, incx, \
        stride_x, w_completed_sec

    // Template Parameters: DIM_X, DIM_Y, LOWER, TRANSPOSE, CONJUGATE, UNIT_DIAG, T
    if(uplo == rocblas_fill_upper)
    {
        if(diag == rocblas_diagonal_unit)
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL(
                    (rocblas_trsv_device<DIM_X, DIM_Y, false, false, false, true, T>),
                    TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, false, true, false, true, T>),
                                   TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
            {
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, false, true, true, true, T>),
                                   TRSV_TEMPLATE_PARAMS);
            }
        }
        else
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL(
                    (rocblas_trsv_device<DIM_X, DIM_Y, false, false, false, false, T>),
                    TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL(
                    (rocblas_trsv_device<DIM_X, DIM_Y, false, true, false, false, T>),
                    TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, false, true, true, false, T>),
                                   TRSV_TEMPLATE_PARAMS);
        }
    }
    else
    {
        if(diag == rocblas_diagonal_unit)
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, true, false, false, true, T>),
                                   TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, true, true, false, true, T>),
                                   TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, true, true, true, true, T>),
                                   TRSV_TEMPLATE_PARAMS);
        }
        else
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL(
                    (rocblas_trsv_device<DIM_X, DIM_Y, true, false, false, false, T>),
                    TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, true, true, false, false, T>),
                                   TRSV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((rocblas_trsv_device<DIM_X, DIM_Y, true, true, true, false, T>),
                                   TRSV_TEMPLATE_PARAMS);
        }
    }
#undef TRSV_TEMPLATE_PARAMS

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_tr_matrix_template for checking Matrix `A` which is a Triangular Matrix
template <typename T, typename U>
rocblas_status rocblas_internal_trsv_check_numerics(const char*       function_name,
                                                    rocblas_handle    handle,
                                                    rocblas_int       m,
                                                    T                 A,
                                                    rocblas_int       offset_a,
                                                    rocblas_int       lda,
                                                    rocblas_stride    stride_a,
                                                    U                 x,
                                                    rocblas_int       offset_x,
                                                    rocblas_int       inc_x,
                                                    rocblas_stride    stride_x,
                                                    rocblas_int       batch_count,
                                                    const rocblas_int check_numerics,
                                                    bool              is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          m,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *trsv*.cpp

// clang-format off

#ifdef INSTANTIATE_TRSV_NUMERICS
#error INSTANTIATE_TRSV_NUMERICS already defined
#endif

#define INSTANTIATE_TRSV_NUMERICS(T_, U_)                                             \
template rocblas_status rocblas_internal_trsv_check_numerics <T_, U_>                 \
                                                   (const char*       function_name,  \
                                                    rocblas_handle    handle,         \
                                                    rocblas_int       m,              \
                                                    T_                A,              \
                                                    rocblas_int       offset_a,       \
                                                    rocblas_int       lda,            \
                                                    rocblas_stride    stride_a,       \
                                                    U_                x,              \
                                                    rocblas_int       offset_x,       \
                                                    rocblas_int       inc_x,          \
                                                    rocblas_stride    stride_x,       \
                                                    rocblas_int       batch_count,    \
                                                    const rocblas_int check_numerics, \
                                                    bool              is_input);

INSTANTIATE_TRSV_NUMERICS(float const*, float*)
INSTANTIATE_TRSV_NUMERICS(double const*, double*)
INSTANTIATE_TRSV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TRSV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TRSV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TRSV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TRSV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TRSV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TRSV_NUMERICS

#ifdef INSTANTIATE_TRSV_TEMPLATE
#error INSTANTIATE_TRSV_TEMPLATE already defined
#endif

#define INSTANTIATE_TRSV_TEMPLATE(DIM_X_, T_, ATYPE_, XTYPE_)                                        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trsv_substitution_template \
                                               <DIM_X_, T_, ATYPE_, XTYPE_>                          \
                                               (rocblas_handle    handle,                            \
                                                rocblas_fill      uplo,                              \
                                                rocblas_operation transA,                            \
                                                rocblas_diagonal  diag,                              \
                                                rocblas_int       m,                                 \
                                                ATYPE_             dA,                               \
                                                ptrdiff_t         offset_A,                          \
                                                rocblas_int       lda,                               \
                                                rocblas_stride    stride_A,                          \
                                                XTYPE_             dx,                               \
                                                ptrdiff_t         offset_x,                          \
                                                rocblas_int       incx,                              \
                                                rocblas_stride    stride_x,                          \
                                                rocblas_int       batch_count,                       \
                                                rocblas_int*      w_completed_sec);


INSTANTIATE_TRSV_TEMPLATE(64, float, float const*, float*)
INSTANTIATE_TRSV_TEMPLATE(64, double, double const*, double*)
INSTANTIATE_TRSV_TEMPLATE(64, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TRSV_TEMPLATE(32, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TRSV_TEMPLATE(64, float, float const* const*, float* const*)
INSTANTIATE_TRSV_TEMPLATE(64, double, double const* const*, double* const*)
INSTANTIATE_TRSV_TEMPLATE(64, rocblas_float_complex, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TRSV_TEMPLATE(32, rocblas_double_complex, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TRSV_TEMPLATE

// clang-format on
