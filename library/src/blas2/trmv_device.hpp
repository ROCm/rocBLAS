/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#include "rocblas.h"
#include <cstddef>

template <rocblas_int NB, typename T>
__device__ void trmvn_kernel_calc(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  rocblas_int      m,
                                  const T*         A,
                                  rocblas_int      lda,
                                  T*               x,
                                  rocblas_int      incx,
                                  T*               w)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;
    if(tid < m)
    {
        A += tid;
        T res = x[tid * incx];
        if(diag == rocblas_diagonal_non_unit)
        {
            res *= A[tid * lda];
        }
        if(rocblas_fill_upper == uplo)
        {
            for(rocblas_int col = tid + 1; col < m; ++col)
            {
                res += A[col * lda] * x[col * incx];
            }
        }
        else
        {
            for(rocblas_int col = 0; col < tid; ++col)
            {
                res += A[col * lda] * x[col * incx];
            }
        }
        w[tid] = res;
    }
}

template <rocblas_int NB, typename T>
__device__ void trmvc_kernel_calc(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  rocblas_int      m,
                                  const T*         A,
                                  rocblas_int      lda,
                                  T*               x,
                                  rocblas_int      incx,
                                  T*               w)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;

    if(tid < m)
    {
        A += tid * lda;
        T res = x[tid * incx];
        if(diag == rocblas_diagonal_non_unit)
        {
            res *= conj(A[tid]);
        }
        if(rocblas_fill_upper == uplo)
        {
            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += conj(A[row]) * x[row * incx];
            }
        }
        else
        {
            for(rocblas_int row = tid + 1; row < m; ++row)
            {
                res += conj(A[row]) * x[row * incx];
            }
        }
        w[tid] = res;
    }
}

template <rocblas_int NB, typename T>
__device__ void trmvt_kernel_calc(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  rocblas_int      m,
                                  const T*         A,
                                  rocblas_int      lda,
                                  T*               x,
                                  rocblas_int      incx,
                                  T*               w)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;

    T           res;
    rocblas_int row;

    if(tid < m)
    {
        A += tid * lda;
        T res = x[tid * incx];
        if(diag == rocblas_diagonal_non_unit)
        {
            res *= A[tid];
        }

        if(rocblas_fill_upper == uplo)
        {
            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += A[row] * x[row * incx];
            }
        }
        else
        {
            for(rocblas_int row = tid + 1; row < m; ++row)
            {
                res += A[row] * x[row * incx];
            }
        }
        w[tid] = res;
    }
}

template <rocblas_int NB, typename A, typename X, typename W>
__global__ void trmvn_kernel(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      m,
                             A                a,
                             ptrdiff_t        shifta,
                             rocblas_int      lda,
                             rocblas_stride   stridea,
                             X                x,
                             ptrdiff_t        shiftx,
                             rocblas_int      incx,
                             rocblas_stride   stridex,
                             W                w,
                             rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    trmvn_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          lda,
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(w, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
__global__ void trmvt_kernel(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      m,
                             A                a,
                             ptrdiff_t        shifta,
                             rocblas_int      lda,
                             rocblas_stride   stridea,
                             X                x,
                             ptrdiff_t        shiftx,
                             rocblas_int      incx,
                             rocblas_stride   stridex,
                             W                w,
                             rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    trmvt_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          lda,
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(w, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
__global__ void trmvc_kernel(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      m,
                             A                a,
                             ptrdiff_t        shifta,
                             rocblas_int      lda,
                             rocblas_stride   stridea,
                             X                x,
                             ptrdiff_t        shiftx,
                             rocblas_int      incx,
                             rocblas_stride   stridex,
                             W                w,
                             rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    trmvc_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          lda,
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(w, hipBlockIdx_y, shiftw, stridew));
}
