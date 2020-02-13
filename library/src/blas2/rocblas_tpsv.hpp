/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
// #include "../blas3/trtri_trsm.hpp"
#include "../blas1/rocblas_copy.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
// #include "rocblas_tpmv.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

namespace
{
    using std::max;
    using std::min;

    constexpr rocblas_int NB_X = 1024;

    template <typename T>
    constexpr T negative_one = -1;
    template <typename T>
    constexpr T zero = 0;
    template <typename T>
    constexpr T one = 1;

    template <rocblas_int BLOCK, bool BATCHED, typename T>
    rocblas_status rocblas_tpsv_template_mem(rocblas_handle handle,
                                             rocblas_int    n,
                                             rocblas_int    batch_count,
                                             void**         mem_x_temp,
                                             void**         mem_x_temp_arr,
                                             void**         mem_invA,
                                             void**         mem_invA_arr)
    {
        return rocblas_status_success;
    }

    //     template </*rocblas_int DIM_X, rocblas_int DIM_Y, */typename T>
    //     __device__ void tpsv_block_multiply(bool upper, rocblas_int actual_n, rocblas_int m, rocblas_int n, const T* AP, rocblas_int offset_A_col, rocblas_int offset_A_row, T* x, rocblas_int incx, T* x_copy)
    //     {
    //         // do a simple gemv multiplication modified to use packed indexing where alpha and beta are both == 1
    //         if(!actual_n || !n || !m)
    //             return;
    //         // ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    //         // rocblas_int tx  = hipThreadIdx_x;
    //         // if(tid < m)
    //         // {
    //         //     int row_offset = tid + offset_A_row;
    //         //     // A += tid;
    //         //     T res = x[tid * incx];
    //         //     // if(diag == rocblas_diagonal_non_unit)
    //         //     {
    //         //         int index = ((row_offset * (2 * actual_n - row_offset + 1)) / 2);
    //         //         res *= AP[index];
    //         //     }
    //         //     if(upper)
    //         //     {
    //         //         for(rocblas_int col = tid + 1; col < m; ++col)
    //         //         {
    //         //             int col_offset = col + offset_A_col;
    //         //             int index = ((col_offset * (2 * n - col_offset + 1)) / 2) + (row_offset - col_offset);
    //         //             res += AP[index] * x[col * incx];
    //         //         }
    //         //     }
    //         //     else
    //         //     {
    //         //         for(rocblas_int col = 0; col < n; ++col)
    //         //         {
    //         //             int col_offset = col + offset_A_col;
    //         //             int index = ((col_offset * (2 * actual_n - col_offset + 1)) / 2) + (row_offset - col_offset);
    //         //             res += AP[index] * x[col * incx];
    //         //         }
    //         //     }
    //         // }
    //         x[0] = 5;

    //     }

    //     template </*rocblas_int DIM_X, rocblas_int DIM_Y,*/ typename T>
    //     __device__ void tpsv_propogation(rocblas_fill uplo, rocblas_int actual_n, rocblas_int n, const T* AP, rocblas_int offset_A_col, rocblas_int offset_A_row, T* x, rocblas_int incx, T* x_copy)
    //     {
    //         if(!n || !actual_n)
    //             return;
    //         x[0] = 4;//1*incx] = 4;
    //         // if(uplo == rocblas_fill_upper)
    //         // {
    //         //     for(int row = n - 1; row >= 0; row--)
    //         //     {
    //         //         rocblas_int offset_row = row + offset_A_row;
    //         //         T total = 0.0;
    //         //         for(int col = row + 1; col < n; col++)
    //         //         {
    //         //             rocblas_int offset_col = col + offset_A_col;
    //         //             int index = ((offset_col * (offset_col + 1)) / 2) + offset_row;
    //         //             total += x[col * incx] * AP[index];
    //         //         }
    //         //         int index = ((offset_row * (offset_row + 1)) / 2) + offset_row;
    //         //         // x[row * incx] = (x_copy[row] - total) / AP[index];
    //         //         // x[0] = 2;
    //         //     }
    //         // }
    //         // else
    //         // {
    //         //     for(int row = 0; row < n; row++)
    //         //     {
    //         //         rocblas_int offset_row = row + offset_A_row;
    //         //         T total = 0.0;
    //         //         for(int col = 0; col < row; col++)
    //         //         {
    //         //             rocblas_int offset_col = col + offset_A_col;
    //         //             int index = ((offset_col * (2 * n - offset_col + 1)) / 2) + (offset_row - offset_col);
    //         //             total += x[col * incx] * AP[index];
    //         //         }
    //         //         int index = offset_row * (2 * n - offset_row + 1) / 2;
    //         //         // x[row * incx] = (x_copy[row] - total) / AP[index];
    //         //         x[0] = 1;
    //         //     }
    //         // }
    //     }

    //     template </*rocblas_int DIM_X, rocblas_int DIM_Y, */typename T>
    //     __global__ /*__device__*/ void tpsv_kernel_calc(rocblas_fill uplo, rocblas_int n, const T* AP, T* x, rocblas_int incx, T* x_copy)
    //     {
    //         // if(uplo == rocblas_fill_upper)
    //         // {
    //         //     x[0] = 2.5;
    //         // }
    //         if(n == 8008)//uplo == rocblas_fill_full)///*rocblas_fill_lower*/ && n == 1000398)//false)//n==6)
    //         {
    //             x[0] = 1.5;
    //             n = 4;
    //             for(int i = 0; i < n; i+= NB_X)
    //             {
    //                 // 1. do a modified gemv to multiply A (at column i) with previously solved block of x, and store in
    //                 //    next block of x (in shared memory).
    //                 rocblas_int col          = i;
    //                 rocblas_int jb           = std::min(NB_X, n - i);
    //                 rocblas_int offset_A_row = 0;//col;
    //                 rocblas_int offset_A_col = 0;
    //                 tpsv_block_multiply</*DIM_X, DIM_Y*/>(uplo, n, jb, i, AP, offset_A_col, offset_A_row, x, incx, x_copy + col);

    //                 // 2. use forward/backwards propogation to solve next block of x (in shared memory)
    //                 offset_A_col = 0;//col;
    //                 // tpsv_propogation</*DIM_X, DIM_Y*/>(uplo, n, jb, AP, offset_A_col, offset_A_row, x /*+ col*/, incx, x_copy);// + col); // todo: indexes

    //                 // 3. store solved portion of x from shared memory into x.
    //             }
    //         }
    //         else
    //         {
    //             x[1] = 0.5;
    //         }

    //     }

    template <typename T>
    __device__ void cache(bool     upper,
                          const T* A,
                          int      offset_A_row,
                          int      offset_A_col,
                          T*       cache,
                          int      nrows,
                          int      ncols,
                          int      na,
                          int      nc)
    {
        if(nrows < 0 || ncols < 0)
            return;

        int index = threadIdx.x;
        while(index < nrows * ncols)
        {
            int row = index / ncols;
            int col = index % ncols;

            int rowA   = row + offset_A_row;
            int colA   = col + offset_A_col;
            int index1 = upper ? ((colA * (colA + 1)) / 2 + rowA)
                               : ((colA * (2 * na - colA + 1)) / 2) + (rowA - colA);
            if(index1 >= (na * (na + 1)) / 2 || row < 0 || col < 0)
                break;

            cache[col * nc + row] = A[index1]; //col * na + row];
            index += blockDim.x;
        }
    }

    template <rocblas_int BLK_SIZE, typename T>
    __device__ void tpsv_upper_kernel_calc(int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        __shared__ T cache_even[BLK_SIZE * BLK_SIZE];
        int          tx = threadIdx.x;

        // main loop
        for(int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
        {
            // cache A and x into shared memory
            if(tx + i < n && tx >= 0)
                xshared[tx] = x[(tx + i) * incx];
            cache<T>(true, A, i, i, cache_even, BLK_SIZE, BLK_SIZE, n, BLK_SIZE);
            __syncthreads();

            // solve diagonal block
            for(int j = BLK_SIZE - 1; j >= 0; j--)
            {
                if(tx == j)
                {
                    xshared[tx] = xshared[tx] / cache_even[j * BLK_SIZE + j];
                }

                __syncthreads();

                if(tx < j)
                {
                    // for rest of block, subtract previous solved part
                    xshared[tx] -= cache_even[j * BLK_SIZE + tx] * xshared[j];
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
                        val += A[indexA] * xshared[p];
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
    __device__ void tpsv_lower_kernel_calc(int n, const T* A, T* x, rocblas_int incx)
    {
        __shared__ T xshared[BLK_SIZE];
        __shared__ T cache_even[BLK_SIZE * BLK_SIZE];
        int          tx = threadIdx.x;

        // main loop
        for(int i = 0; i < n; i += BLK_SIZE)
        {
            // cache A and x into shared memory
            if(tx + i < n)
                xshared[tx] = x[(tx + i) * incx];
            cache<T>(false, A, i, i, cache_even, BLK_SIZE, BLK_SIZE, n, BLK_SIZE);
            __syncthreads();

            // solve diagonal block
            for(int j = 0; j < BLK_SIZE; j++)
            {
                if(tx == j)
                {
                    xshared[tx] = xshared[tx] / cache_even[j * BLK_SIZE + j];
                }

                __syncthreads();

                if(tx >= j + 1)
                {
                    // for rest of block, subtract previous solved part
                    xshared[tx] -= cache_even[j * BLK_SIZE + tx] * xshared[j];
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

    template <rocblas_int BLK_SIZE, typename TConstPtr, typename TPtr>
    __global__ void rocblas_tpsv_kernel(rocblas_fill   uplo,
                                        rocblas_int    n,
                                        TConstPtr      APa,
                                        ptrdiff_t      shift_A,
                                        rocblas_stride stride_A,
                                        TPtr           xa,
                                        ptrdiff_t      shift_x,
                                        rocblas_int    incx,
                                        rocblas_stride stride_x)
    {
        const auto* AP = load_ptr_batch(APa, hipBlockIdx_y, shift_A, stride_A);
        auto*       x  = load_ptr_batch(xa, hipBlockIdx_y, shift_x, stride_x);

        if(uplo == rocblas_fill_upper)
            tpsv_upper_kernel_calc<BLK_SIZE>(n, AP, x, incx);
        else
            tpsv_lower_kernel_calc<BLK_SIZE>(n, AP, x, incx);
    }

    template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
    rocblas_status rocblas_tpsv_template(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         rocblas_int       n,
                                         U                 A,
                                         rocblas_int       offset_A,
                                         rocblas_stride    stride_A,
                                         V                 x,
                                         rocblas_int       offset_x,
                                         rocblas_int       incx,
                                         rocblas_stride    stride_x,
                                         rocblas_int       batch_count,
                                         V                 x_temp)
    {
        if(batch_count == 0)
            return rocblas_status_success;

        rocblas_status status = rocblas_status_success;

        // Temporarily switch to host pointer mode, restoring on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        if(transA == rocblas_operation_conjugate_transpose)
            transA = rocblas_operation_transpose;

        ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
        ptrdiff_t shift_A = offset_A;

        static constexpr int DIM_X  = 1; //128;
        static constexpr int DIM_Y  = 1;
        rocblas_int          blocks = 1; //(n - 1) / DIM_X + 1;

        dim3 grid(blocks, batch_count);
        dim3 threads(32);

        hipLaunchKernelGGL((rocblas_tpsv_kernel<32>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           n,
                           A,
                           shift_A,
                           stride_A,
                           x,
                           shift_x,
                           incx,
                           stride_x);

        // int index = upper ? ((ty * (ty + 1)) / 2) + tx : ((ty * (2 * n - ty + 1)) / 2) + (tx - ty);
        // if(uplo == rocblas_fill_lower)
        // {
        //     for(int row = 0; row < n; row++)
        //     {
        //         T total = 0.0;
        //         for(int col = 0; col < row; col++)
        //         {
        //             int index = ((col * (2 * n - col + 1)) / 2) + (row - col);
        //             total += x[col * incx] * A[index];
        //         }
        //         int index = row * (2 * n - row + 1) / 2;
        //         x[row * incx] = (x_temp[row] - total) / A[index];
        //     }
        // }

        // if(BATCHED)
        // {
        //     setup_batched_array<BLOCK>(
        //         handle->rocblas_stream, (T*)x_temp, x_temp_els, (T**)x_temparr, batch_count);
        // }

        // if(exact_blocks)
        // {
        //     status = special_trsv_template<BLOCK, T>(handle,
        //                                              uplo,
        //                                              transA,
        //                                              diag,
        //                                              n,
        //                                              A,
        //                                              offset_A,
        //                                              lda,
        //                                              stride_A,
        //                                              x,
        //                                              offset_x,
        //                                              abs_incx,
        //                                              stride_x,
        //                                              (U)(BATCHED ? invAarr : invA),
        //                                              offset_invA,
        //                                              stride_invA,
        //                                              (V)(BATCHED ? x_temparr : x_temp),
        //                                              x_temp_els,
        //                                              batch_count);
        //     if(status != rocblas_status_success)
        //         return status;

        //     // TODO: workaround to fix negative incx issue
        //     if(incx < 0)
        //         flip_vector<T>(handle, x, n, abs_incx, stride_x, batch_count, offset_x);
        // }
        // else
        // {
        //     status = rocblas_trsv_left<BLOCK, T>(handle,
        //                                          uplo,
        //                                          transA,
        //                                          n,
        //                                          A,
        //                                          offset_A,
        //                                          lda,
        //                                          stride_A,
        //                                          x,
        //                                          offset_x,
        //                                          abs_incx,
        //                                          stride_B,
        //                                          (U)(BATCHED ? invAarr : invA),
        //                                          offset_invA,
        //                                          stride_invA,
        //                                          (V)(BATCHED ? x_temparr : x_temp),
        //                                          x_temp_els,
        //                                          batch_count);
        //     if(status != rocblas_status_success)
        //         return status;

        //     // copy solution X into x
        //     // TODO: workaround to fix negative incx issue
        //     strided_vector_copy<T>(handle,
        //                            x,
        //                            abs_incx,
        //                            stride_x,
        //                            (V)(BATCHED ? x_temparr : x_temp),
        //                            incx < 0 ? -1 : 1,
        //                            x_temp_els,
        //                            n,
        //                            batch_count,
        //                            offset_x,
        //                            incx < 0 ? n - 1 : 0);
        // }

        // return status;
        return rocblas_status_success;
    }

} // namespace
