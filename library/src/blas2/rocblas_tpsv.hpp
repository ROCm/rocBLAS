/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../blas1/rocblas_copy.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

namespace
{
    constexpr rocblas_int NB = 256;

    // Uses fowards elimination to solve Ax = b for a upper-triangular matrix which has been
    // transposed - therefore is a lower-triangular matrix.
    template <bool CONJ, rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvt_upper_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        int          tx = threadIdx.x;

        // main loop - iterate forward in BLK_SIZE chunks
        for(int i = 0; i < n; i += BLK_SIZE)
        {
            // cache x into shared memory
            if(tx + i < n)
                xshared[tx] = x[(tx + i) * incx];

            __syncthreads();

            // iterate through the current block and solve elements
            for(int j = 0; j < BLK_SIZE; j++)
            {
                if(j + i >= n)
                    break;

                // solve element that can be solved
                if(tx == j && !diag)
                {
                    int colA = j + i;
                    int rowA = j + i;
                    // Use upper-triangular matrix indexing
                    int indexA = (rowA * (rowA + 1) / 2) + colA;
                    // if(colA < n && rowA < n && indexA < (n * (n + 1) / 2))
                    xshared[tx] = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
                }

                __syncthreads();
                // for rest of block, subtract previous solved part
                if(tx >= j + 1)
                {
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = (rowA * (rowA + 1) / 2) + colA;
                    // if(colA < n && rowA < n && indexA < (n * (n + 1) / 2))
                    if(rowA < n)
                        xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
                }
            }

            __syncthreads();

            // apply solved diagonal block to the rest of the array
            // 1. Iterate down rows
            for(int j = BLK_SIZE; j < n - i; j += BLK_SIZE)
            {
                if(tx + i + j >= n)
                    continue;

                // 2. Sum result (across columns) to be subtracted from original value
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA   = i + p;
                    int rowA   = i + tx + j;
                    int indexA = (rowA * (rowA + 1) / 2) + colA;
                    if(colA < 0 || rowA < 0 || colA >= n || rowA >= n)
                        continue;
                    if(indexA >= 0 && indexA < (n * (n + 1) / 2))
                    {
                        if(diag && colA == rowA)
                            val += xshared[p];
                        else
                            val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
                    }
                }

                if(tx + i + j < n)
                    x[(tx + i + j) * incx] -= val;
            }

            // store solved part back to global memory
            if(tx + i < n)
                x[(tx + i) * incx] = xshared[tx];

            __syncthreads();
        }
    }

    // Uses backwards elimination to solve Ax = b for a lower-triangular matrix which has been
    // transposed - therefore is an upper-triangular matrix.
    template <bool CONJ, rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvt_lower_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        int          tx = threadIdx.x;

        // main loop - Start at end of array and iterate backwards in BLK_SIZE chunks
        for(int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
        {
            // cache x into shared memory
            if(tx + i < n && tx + i >= 0)
                xshared[tx] = x[(tx + i) * incx];

            __syncthreads();

            // Iterate backwards through the current block to solve elements.
            for(int j = BLK_SIZE - 1; j >= 0; j--)
            {
                // Solve the new element that can be solved
                if(tx == j && !diag)
                {
                    int colA = j + i;
                    int rowA = j + i;
                    // Use lower-triangular matrix indexing
                    int indexA = ((rowA * (2 * n - rowA + 1)) / 2) + (colA - rowA);
                    if(rowA >= 0 && colA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                        xshared[tx] = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
                }

                __syncthreads();

                // for rest of block, subtract previous solved part
                if(tx < j)
                {
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = ((rowA * (2 * n - rowA + 1)) / 2) + (colA - rowA);
                    if(rowA >= 0 && colA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                        xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
                }
            }

            __syncthreads();

            // apply solved diagonal block to the rest of the array
            // 1. Iterate up rows, starting at the block above the current block
            for(int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
            {
                // 2. Sum result (across columns) to be subtracted from the original value
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA = i + p;
                    int rowA = tx + j;
                    if(colA < 0 || rowA < 0 || colA >= n || rowA >= n)
                        continue;
                    int indexA = ((rowA * (2 * n - rowA + 1)) / 2) + (colA - rowA);

                    if(rowA >= 0 && colA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                    {
                        if(diag && colA == rowA)
                            val += xshared[p];
                        else
                            val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
                    }
                }

                if(tx + j < n && tx + j >= 0)
                    x[(tx + j) * incx] -= val;
            }

            // store solved part back to global memory
            if(tx + i < n && tx + i >= 0)
                x[(tx + i) * incx] = xshared[tx];

            __syncthreads();
        }
    }

    // Uses backwards elimination to solve Ax = b for a upper-triangular matrix.
    template <rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvn_upper_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        int          tx = threadIdx.x;

        // main loop - Start at end of array and iterate backwards in BLK_SIZE chunks
        for(int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
        {
            // cache x into shared memory
            if(tx + i < n && tx + i >= 0)
                xshared[tx] = x[(tx + i) * incx];

            __syncthreads();

            // Iterate backwards through the current block to solve elements.
            for(int j = BLK_SIZE - 1; j >= 0; j--)
            {
                // Solve the new element that can be solved
                if(tx == j && !diag)
                {
                    int colA   = j + i;
                    int rowA   = j + i;
                    int indexA = ((colA * (colA + 1) / 2) + rowA);
                    if(indexA >= 0 && indexA < ((n * (n + 1) / 2)))
                        xshared[tx] = xshared[tx] / A[indexA];
                }

                __syncthreads();

                // for rest of block, subtract previous solved part
                if(tx < j)
                {
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = ((colA * (colA + 1) / 2) + rowA);
                    if(indexA >= 0 && indexA < ((n * (n + 1) / 2)))
                        xshared[tx] -= A[indexA] * xshared[j];
                }
            }

            __syncthreads();

            // apply solved diagonal block to the rest of the array
            // 1. Iterate up rows, starting at the block above the current block
            for(int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
            {
                // 2. Sum result (across columns) to be subtracted from the original value
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA = i + p;
                    int rowA = tx + j;
                    if(colA < 0 || rowA < 0 || colA >= n || rowA >= n)
                        continue;
                    int indexA = ((colA * (colA + 1) / 2) + rowA);
                    if(rowA >= 0 && colA >= 0)
                    {
                        if(diag && colA == rowA)
                            val += xshared[p];
                        else if(indexA >= 0)
                            val += A[indexA] * xshared[p];
                    }
                }

                if(tx + j < n && tx + j >= 0)
                    x[(tx + j) * incx] -= val;
            }

            // store solved part back to global memory
            if(tx + i < n && tx + i >= 0)
                x[(tx + i) * incx] = xshared[tx];

            __syncthreads();
        }
    }

    // Uses forward elimination to solve Ax = b for a lower-triangular matrix.
    template <rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvn_lower_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        int          tx = threadIdx.x;

        // main loop - iterate forward in BLK_SIZE chunks.
        for(int i = 0; i < n; i += BLK_SIZE)
        {
            // cache x into shared memory
            if(tx + i < n)
                xshared[tx] = x[(tx + i) * incx];

            __syncthreads();

            // iterate through the current block and solve elements
            for(int j = 0; j < BLK_SIZE; j++)
            {
                if(j + i >= n)
                    break;

                // solve element that can be solved
                if(tx == j && !diag)
                {
                    int colA    = j + i;
                    int rowA    = j + i;
                    int indexA  = ((colA * (2 * n - colA + 1)) / 2) + (rowA - colA);
                    xshared[tx] = xshared[tx] / A[indexA];
                }

                __syncthreads();

                // for rest of block, subtract previous solved part
                if(tx >= j + 1)
                {
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = ((colA * (2 * n - colA + 1)) / 2) + (rowA - colA);
                    // TODO: check here
                    xshared[tx] -= A[indexA] * xshared[j];
                }
            }

            __syncthreads();

            // apply solved diagonal block to the rest of the array
            // 1. Iterate down rows
            for(int j = BLK_SIZE; j < n - i; j += BLK_SIZE)
            {
                if(tx + i + j >= n)
                    continue;

                // 2. Sum result (across columns) to be subtracted from original value
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA = i + p;
                    int rowA = i + tx + j;
                    if(colA < 0 || rowA < 0 || colA >= n || rowA >= n)
                        continue;
                    int indexA = ((colA * (2 * n - colA + 1)) / 2) + (rowA - colA);
                    if(diag && colA == rowA)
                        val += xshared[p];
                    else if(indexA >= 0 && indexA < (n * (n + 1) / 2))
                        val += A[indexA] * xshared[p];
                }

                if(tx + i + j < n)
                    x[(tx + i + j) * incx] -= val;
            }

            // store solved part back to global memory
            if(tx + i < n)
                x[(tx + i) * incx] = xshared[tx];

            __syncthreads();
        }
    }

    template <bool CONJ, rocblas_int BLK_SIZE, typename TConstPtr, typename TPtr>
    __global__ void rocblas_tpsv_kernel(rocblas_fill      uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal  diag,
                                        rocblas_int       n,
                                        TConstPtr         APa,
                                        ptrdiff_t         shift_A,
                                        rocblas_stride    stride_A,
                                        TPtr              xa,
                                        ptrdiff_t         shift_x,
                                        rocblas_int       incx,
                                        rocblas_stride    stride_x)
    {
        const auto* AP = load_ptr_batch(APa, hipBlockIdx_y, shift_A, stride_A);
        auto*       x  = load_ptr_batch(xa, hipBlockIdx_y, shift_x, stride_x);

        bool is_diag = diag == rocblas_diagonal_unit;

        if(transA == rocblas_operation_none)
        {
            if(uplo == rocblas_fill_upper)
                tpsvn_upper_kernel_calc<BLK_SIZE>(is_diag, n, AP, x, incx);
            else
                tpsvn_lower_kernel_calc<BLK_SIZE>(is_diag, n, AP, x, incx);
        }
        else if(uplo == rocblas_fill_upper)
            tpsvt_upper_kernel_calc<CONJ, BLK_SIZE>(is_diag, n, AP, x, incx);
        else
            tpsvt_lower_kernel_calc<CONJ, BLK_SIZE>(is_diag, n, AP, x, incx);
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

        rocblas_int blocks = 1;

        dim3 grid(blocks, batch_count);
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

} // namespace
