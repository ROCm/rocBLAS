/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "../blas1/rocblas_copy.hpp"

__device__ inline rocblas_int
    packed_matrix_index(bool upper, bool trans, rocblas_int n, rocblas_int row, rocblas_int col)
{
    return upper ? (trans ? ((row * (row + 1) / 2) + col) : ((col * (col + 1) / 2) + row))
                 : (trans ? (((row * (2 * n - row + 1)) / 2) + (col - row))
                          : (((col * (2 * n - col + 1)) / 2) + (row - col)));
}

// Uses forward substitution to solve Ax = b. Used for a non-transposed lower-triangular matrix
// or a transposed upper-triangular matrix.
template <bool CONJ, rocblas_int BLK_SIZE, typename T>
__device__ void tpsv_forward_substitution_calc(
    bool diag, bool trans, int n, const T* __restrict__ A, T* __restrict__ x, rocblas_int incx)
{
    __shared__ T xshared[BLK_SIZE];
    int          tx = threadIdx.x;

    // main loop - iterate forward in BLK_SIZE chunks
    for(rocblas_int i = 0; i < n; i += BLK_SIZE)
    {
        // cache x into shared memory
        if(tx + i < n)
            xshared[tx] = x[(tx + i) * incx];

        __syncthreads();

        // iterate through the current block and solve elements
        for(rocblas_int j = 0; j < BLK_SIZE; j++)
        {
            // solve element that can be solved
            if(tx == j && !diag && j + i < n)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = j + i;
                rocblas_int indexA = packed_matrix_index(trans, trans, n, rowA, colA);
                xshared[tx]        = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
            }

            __syncthreads();

            // for rest of block, subtract previous solved part
            if(tx > j && j + i < n)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = tx + i;
                rocblas_int indexA = packed_matrix_index(trans, trans, n, rowA, colA);

                // Ensure row is in range, and subtract
                if(rowA < n)
                    xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
            }
        }

        __syncthreads();

        // apply solved diagonal block to the rest of the array
        // 1. Iterate down rows
        for(rocblas_int j = BLK_SIZE + i; j < n; j += BLK_SIZE)
        {
            if(tx + j >= n)
                break;

            // 2. Sum result (across columns) to be subtracted from original value
            T val = 0;
            for(rocblas_int p = 0; p < BLK_SIZE; p++)
            {
                rocblas_int colA   = i + p;
                rocblas_int rowA   = tx + j;
                rocblas_int indexA = packed_matrix_index(trans, trans, n, rowA, colA);

                if(diag && colA == rowA)
                    val += xshared[p];
                else if(colA < n)
                    val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
            }

            x[(tx + j) * incx] -= val;
        }

        // store solved part back to global memory
        if(tx + i < n)
            x[(tx + i) * incx] = xshared[tx];

        __syncthreads();
    }
}

// Uses backward substitution to solve Ax = b. Used for a non-transposed upper-triangular matrix
// or a transposed lower-triangular matrix.
template <bool CONJ, rocblas_int BLK_SIZE, typename T>
__device__ void tpsv_backward_substitution_calc(
    bool diag, bool trans, int n, const T* __restrict__ A, T* __restrict__ x, rocblas_int incx)
{
    __shared__ T xshared[BLK_SIZE];
    int          tx = threadIdx.x;

    // main loop - Start at end of array and iterate backwards in BLK_SIZE chunks
    for(rocblas_int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
    {
        // cache x into shared memory
        if(tx + i >= 0)
            xshared[tx] = x[(tx + i) * incx];

        __syncthreads();

        // Iterate backwards through the current block to solve elements.
        for(rocblas_int j = BLK_SIZE - 1; j >= 0; j--)
        {
            // Solve the new element that can be solved
            if(tx == j && !diag && j + i >= 0)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = j + i;
                rocblas_int indexA = packed_matrix_index(!trans, trans, n, rowA, colA);
                xshared[tx]        = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
            }

            __syncthreads();

            // for rest of block, subtract previous solved part
            if(tx < j && j + i >= 0)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = tx + i;
                rocblas_int indexA = packed_matrix_index(!trans, trans, n, rowA, colA);

                // Ensure row is in range, and subtract
                if(rowA >= 0)
                    xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
            }
        }

        __syncthreads();

        // apply solved diagonal block to the rest of the array
        // 1. Iterate up rows, starting at the block above the current block
        for(rocblas_int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
        {
            if(tx + j < 0)
                break;

            // 2. Sum result (across columns) to be subtracted from the original value
            T val = 0;
            for(rocblas_int p = 0; p < BLK_SIZE; p++)
            {
                rocblas_int colA   = i + p;
                rocblas_int rowA   = tx + j;
                rocblas_int indexA = packed_matrix_index(!trans, trans, n, rowA, colA);

                if(diag && colA == rowA)
                    val += xshared[p];
                else
                    val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
            }

            x[(tx + j) * incx] -= val;
        }

        // store solved part back to global memory
        if(tx + i >= 0)
            x[(tx + i) * incx] = xshared[tx];

        __syncthreads();
    }
}

/**
     *  Calls forwards/backwards substitution kernels with appropriate arguments.
     *  Note the attribute here - apparently this is needed for group sizes > 256.
     *  Currently BLK_SIZE is set to 512, so this is required or else we get
     *  incorrect behaviour.
     *
     *  To optimize we can probably use multiple kernels (a substitution kernel and a
     *  multiplication kernel) so we can use multiple blocks instead of a single one.
     */
template <bool CONJ, rocblas_int BLK_SIZE, typename TConstPtr, typename TPtr>
__attribute__((amdgpu_flat_work_group_size(64, 1024))) __global__ void
    rocblas_tpsv_kernel(rocblas_fill      uplo,
                        rocblas_operation transA,
                        rocblas_diagonal  diag,
                        rocblas_int       n,
                        TConstPtr __restrict__ APa,
                        ptrdiff_t      shift_A,
                        rocblas_stride stride_A,
                        TPtr __restrict__ xa,
                        ptrdiff_t      shift_x,
                        rocblas_int    incx,
                        rocblas_stride stride_x)
{
    const auto* AP = load_ptr_batch(APa, hipBlockIdx_x, shift_A, stride_A);
    auto*       x  = load_ptr_batch(xa, hipBlockIdx_x, shift_x, stride_x);

    bool is_diag = diag == rocblas_diagonal_unit;

    if(transA == rocblas_operation_none)
    {
        if(uplo == rocblas_fill_upper)
            tpsv_backward_substitution_calc<false, BLK_SIZE>(is_diag, false, n, AP, x, incx);
        else
            tpsv_forward_substitution_calc<false, BLK_SIZE>(is_diag, false, n, AP, x, incx);
    }
    else if(uplo == rocblas_fill_upper)
        tpsv_forward_substitution_calc<CONJ, BLK_SIZE>(is_diag, true, n, AP, x, incx);
    else
        tpsv_backward_substitution_calc<CONJ, BLK_SIZE>(is_diag, true, n, AP, x, incx);
}

template <rocblas_int BLOCK, typename TConstPtr, typename TPtr>
rocblas_status rocblas_tpsv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       n,
                                     TConstPtr         A,
                                     rocblas_int       offset_A,
                                     rocblas_stride    stride_A,
                                     TPtr              x,
                                     rocblas_int       offset_x,
                                     rocblas_int       incx,
                                     rocblas_stride    stride_x,
                                     rocblas_int       batch_count)
{
    if(batch_count == 0 || n == 0)
        return rocblas_status_success;

    rocblas_status status = rocblas_status_success;

    // Temporarily switch to host pointer mode, restoring on return
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    ptrdiff_t shift_A = offset_A;

    dim3 grid(batch_count);
    dim3 threads(BLOCK);

    if(rocblas_operation_conjugate_transpose == transA)
    {
        hipLaunchKernelGGL((rocblas_tpsv_kernel<true, BLOCK>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           transA,
                           diag,
                           n,
                           A,
                           shift_A,
                           stride_A,
                           x,
                           shift_x,
                           incx,
                           stride_x);
    }
    else
    {
        hipLaunchKernelGGL((rocblas_tpsv_kernel<false, BLOCK>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           transA,
                           diag,
                           n,
                           A,
                           shift_A,
                           stride_A,
                           x,
                           shift_x,
                           incx,
                           stride_x);
    }

    return rocblas_status_success;
}
