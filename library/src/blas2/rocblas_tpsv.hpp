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
    // template <typename T>
    // __device__ void cachet(bool     upper,
    //                        const T* A,
    //                        int      offset_A_row,
    //                        int      offset_A_col,
    //                        T*       cache,
    //                        int      nrows,
    //                        int      ncols,
    //                        int      na,
    //                        int      nc)
    // {
    //     if(nrows < 0 || ncols < 0)
    //         return;

    //     int index = threadIdx.x;
    //     while(index < nrows * ncols)
    //     {
    //         int row = index / ncols;
    //         int col = index % ncols;

    //         int rowA   = row + offset_A_row;
    //         int colA   = col + offset_A_col;
    //         int index1 = upper ? ((rowA * (rowA + 1)) / 2 + colA)
    //                            : ((rowA * (2 * na - rowA + 1)) / 2) + (colA - rowA);
    //         if(index1 >= (na * (na + 1)) / 2 || row < 0 || col < 0 || index1 < 0)
    //         {
    //             // break;
    //             index += blockDim.x;
    //             continue;
    //         }

    //         cache[col * nc + row] = A[index1];
    //         index += blockDim.x;
    //     }
    // }

    // template <typename T>
    // __device__ void cachen(bool     upper,
    //                        const T* A,
    //                        int      offset_A_row,
    //                        int      offset_A_col,
    //                        T*       cache,
    //                        int      nrows,
    //                        int      ncols,
    //                        int      na,
    //                        int      nc)
    // {
    //     if(nrows < 0 || ncols < 0)
    //         return;

    //     int index = threadIdx.x;
    //     while(index < nrows * ncols)
    //     {
    //         int row = index / ncols;
    //         int col = index % ncols;

    //         int rowA   = row + offset_A_row;
    //         int colA   = col + offset_A_col;
    //         int index1 = upper ? ((colA * (colA + 1)) / 2 + rowA)
    //                            : ((colA * (2 * na - colA + 1)) / 2) + (rowA - colA);
    //         if(index1 >= (na * (na + 1)) / 2 || row < 0 || col < 0)
    //             break;

    //         cache[col * nc + row] = A[index1];
    //         index += blockDim.x;
    //     }
    // }

    template <bool CONJ, rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvt_upper_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        // __shared__ T cache_even[BLK_SIZE * BLK_SIZE];
        int tx = threadIdx.x;

        // main loop
        for(int i = 0; i < n; i += BLK_SIZE)
        {
            // cache A and x into shared memory
            if(tx + i < n)
                xshared[tx] = x[(tx + i) * incx];
            // cachet<T>(true, A, i, i, cache_even, BLK_SIZE, BLK_SIZE, n, BLK_SIZE);
            __syncthreads();

            // solve diagonal block
            for(int j = 0; j < BLK_SIZE; j++)
            {
                if(tx == j && !diag)
                {
                    int colA   = j + i;
                    int rowA   = j + i;
                    int indexA = (rowA * (rowA + 1) / 2) + colA;
                    if(colA >= 0 && rowA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                        xshared[tx] = xshared[tx]
                                      / (CONJ ? conj(A[indexA])
                                              : A[indexA]); //cache_even[j * BLK_SIZE + j];
                }

                __syncthreads();

                if(tx >= j + 1)
                {
                    // for rest of block, subtract previous solved part
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = (rowA * (rowA + 1) / 2) + colA;
                    if(colA >= 0 && rowA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                        xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA])
                                       * xshared[j]; //cache_even[j * BLK_SIZE + tx] * xshared[j];
                }
            }

            __syncthreads();

            // apply diagonal block
            for(int j = BLK_SIZE; j < n - i; j += BLK_SIZE)
            {
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA   = i + p;
                    int rowA   = i + tx + j;
                    int indexA = (rowA * (rowA + 1) / 2) + colA;
                    if(colA >= 0 && rowA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
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

            // store back to global memory
            if(tx + i < n)
                x[(tx + i) * incx] = xshared[tx];
        }
    }

    template <bool CONJ, rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvt_lower_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        // __shared__ T cache_even[BLK_SIZE * BLK_SIZE];
        int tx = threadIdx.x;

        // main loop
        for(int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
        {
            // cache A and x into shared memory
            if(tx + i < n && tx >= 0)
                xshared[tx] = x[(tx + i) * incx];
            // cachet<T>(false, A, i, i, cache_even, BLK_SIZE, BLK_SIZE, n, BLK_SIZE);
            __syncthreads();

            // solve diagonal block
            for(int j = BLK_SIZE - 1; j >= 0; j--)
            {
                if(tx == j && !diag)
                {
                    int colA   = j + i;
                    int rowA   = j + i;
                    int indexA = ((rowA * (2 * n - rowA + 1)) / 2) + (colA - rowA);
                    if(rowA >= 0 && colA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                        xshared[tx] = xshared[tx]
                                      / (CONJ ? conj(A[indexA])
                                              : A[indexA]); //cache_even[j * BLK_SIZE + j];
                }

                __syncthreads();

                if(tx < j)
                {
                    // for rest of block, subtract previous solved part
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = ((rowA * (2 * n - rowA + 1)) / 2) + (colA - rowA);
                    if(rowA >= 0 && colA >= 0 && indexA >= 0 && indexA < (n * (n + 1) / 2))
                        xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA])
                                       * xshared[j]; //cache_even[j * BLK_SIZE + tx] * xshared[j];
                }
            }

            __syncthreads();

            // apply diagonal block
            for(int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
            {
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA   = i + p;
                    int rowA   = tx + j;
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

            // store back to global memory
            if(tx + i < n && tx + i >= 0)
                x[(tx + i) * incx] = xshared[tx];
        }
    }

    template <rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvn_upper_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        // __shared__ T cache_even[BLK_SIZE * BLK_SIZE];
        int tx = threadIdx.x;

        // main loop
        for(int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
        {
            // cache A and x into shared memory
            if(tx + i < n && tx >= 0)
                xshared[tx] = x[(tx + i) * incx];
            // cachen<T>(true, A, i, i, cache_even, BLK_SIZE, BLK_SIZE, n, BLK_SIZE);
            __syncthreads();

            // solve diagonal block
            for(int j = BLK_SIZE - 1; j >= 0; j--)
            {
                if(tx == j && !diag)
                {
                    int colA    = j + i;
                    int rowA    = j + i;
                    int indexA  = ((colA * (colA + 1) / 2) + rowA);
                    xshared[tx] = xshared[tx] / A[indexA]; //cache_even[j * BLK_SIZE + j];
                }

                __syncthreads();

                if(tx < j)
                {
                    // for rest of block, subtract previous solved part
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = ((colA * (colA + 1) / 2) + rowA);
                    xshared[tx]
                        -= A[indexA] * xshared[j]; //cache_even[j * BLK_SIZE + tx] * xshared[j];
                }
            }

            __syncthreads();

            // apply diagonal block
            for(int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
            {
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA   = i + p;
                    int rowA   = tx + j;
                    int indexA = ((colA * (colA + 1) / 2) + rowA);
                    if(rowA >= 0 && colA >= 0)
                    {
                        if(diag && colA == rowA)
                            val += xshared[p];
                        else
                            val += A[indexA] * xshared[p];
                    }
                }

                if(tx + j < n && tx + j >= 0)
                    x[(tx + j) * incx] -= val;
            }

            // store back to global memory
            if(tx + i < n && tx + i >= 0)
                x[(tx + i) * incx] = xshared[tx];
        }
    }

    // TODO: Change this to use shared memory.
    template <rocblas_int BLK_SIZE, typename T>
    __device__ void tpsvn_lower_kernel_calc(bool diag, int n, const T* A, T* x, rocblas_int incx)
    {
        // constexpr int blk_arr_size = BLK_SIZE * (BLK_SIZE + 1) / 2;
        __shared__ T xshared[BLK_SIZE];
        // __shared__ T cache_even[blk_arr_size];
        int tx = threadIdx.x;

        // main loop
        for(int i = 0; i < n; i += BLK_SIZE)
        {
            // cache A and x into shared memory
            if(tx + i < n)
                xshared[tx] = x[(tx + i) * incx];
            // TODO: custom kernel to copy part of the packed matrix?
            // copy_kernel<false>((blk_arr_size), A, 0, 1, 0, cache_even, 0, 1, 0);
            // cachen<T>(false, A, i, i, cache_even, BLK_SIZE, BLK_SIZE, n, BLK_SIZE);
            __syncthreads();

            // solve diagonal block
            for(int j = 0; j < BLK_SIZE; j++)
            {
                if(tx == j && !diag)
                {
                    int colA   = j + i;
                    int rowA   = j + i;
                    int indexA = ((colA * (2 * n - colA + 1)) / 2) + (rowA - colA);
                    xshared[tx]
                        = xshared[tx]
                          / A[indexA]; //cache_even[indexA];//A[indexA]; //cache_even[j * BLK_SIZE + j];
                }

                __syncthreads();

                if(tx >= j + 1)
                {
                    // for rest of block, subtract previous solved part
                    int colA   = j + i;
                    int rowA   = tx + i;
                    int indexA = ((colA * (2 * n - colA + 1)) / 2) + (rowA - colA);
                    xshared[tx]
                        -= A[indexA]
                           * xshared
                               [j]; //A[indexA] * xshared[j]; //cache_even[j * BLK_SIZE + tx] * xshared[j];
                }
            }

            __syncthreads();

            // apply diagonal block
            for(int j = BLK_SIZE; j < n - i; j += BLK_SIZE)
            {
                T val = 0;
                for(int p = 0; p < BLK_SIZE; p++)
                {
                    int colA   = i + p;
                    int rowA   = i + tx + j;
                    int indexA = ((colA * (2 * n - colA + 1)) / 2) + (rowA - colA);
                    if(diag && colA == rowA)
                        val += xshared[p];
                    else
                        val += A[indexA] * xshared[p];
                }

                if(tx + i + j < n)
                    x[(tx + i + j) * incx] -= val;
            }

            // store back to global memory
            if(tx + i < n)
                x[(tx + i) * incx] = xshared[tx];
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

        static constexpr int DIM_X  = 1; //128;
        static constexpr int DIM_Y  = 1;
        rocblas_int          blocks = 1; //(n - 1) / DIM_X + 1;

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
