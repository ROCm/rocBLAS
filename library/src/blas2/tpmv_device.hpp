/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "utility.h"

#define uat(_i, _j) (((_j) * ((_j) + 1)) / 2 + (_i))
#define lat(_i, _j) ((_j)*m + ((_i) - (_j)) - (((_j)-1) * (_j)) / 2)

template <rocblas_int NB, typename T>
__device__ void tpmvn_kernel_calc(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  rocblas_int      m,
                                  const T*         A,
                                  T*               x,
                                  rocblas_int      incx,
                                  T*               w)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;
    if(tid < m)
    {
        T res = x[tid * incx];
        if(rocblas_fill_upper == uplo)
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[uat(tid, tid)];
            }
            for(rocblas_int col = tid + 1; col < m; ++col)
            {
                res += A[uat(tid, col)] * x[col * incx];
            }
        }
        else
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[lat(tid, tid)];
            }
            for(rocblas_int col = 0; col < tid; ++col)
            {
                res += A[lat(tid, col)] * x[col * incx];
            }
        }

        w[tid] = res;
    }
}

template <rocblas_int NB, typename T>
__device__ void tpmvc_kernel_calc(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  rocblas_int      m,
                                  const T*         A,
                                  T*               x,
                                  rocblas_int      incx,
                                  T*               w)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;
    if(tid < m)
    {
        T res = x[tid * incx];
        if(rocblas_fill_upper == uplo)
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= conj(A[uat(tid, tid)]);
            }
            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += conj(A[uat(row, tid)]) * x[row * incx];
            }
        }
        else
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= conj(A[lat(tid, tid)]);
            }
            for(rocblas_int row = tid + 1; row < m; ++row)
            {
                res += conj(A[lat(row, tid)]) * x[row * incx];
            }
        }
        w[tid] = res;
    }
}

template <rocblas_int NB, typename T>
__device__ void tpmvt_kernel_calc(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  rocblas_int      m,
                                  const T*         A,
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

        T res = x[tid * incx];
        if(rocblas_fill_upper == uplo)
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[uat(tid, tid)];
            }

            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += A[uat(row, tid)] * x[row * incx];
            }
        }
        else
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[lat(tid, tid)];
            }

            for(rocblas_int row = tid + 1; row < m; ++row)
            {
                res += A[lat(row, tid)] * x[row * incx];
            }
        }
        w[tid] = res;
    }
}

template <rocblas_int NB, typename A, typename X, typename W>
__global__ void tpmvn_kernel(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      m,
                             A                a,
                             ptrdiff_t        shifta,
                             rocblas_stride   stridea,
                             X                x,
                             ptrdiff_t        shiftx,
                             rocblas_int      incx,
                             rocblas_stride   stridex,
                             W                w,
                             rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    tpmvn_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(w, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
__global__ void tpmvt_kernel(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      m,
                             A                a,
                             ptrdiff_t        shifta,
                             rocblas_stride   stridea,
                             X                x,
                             ptrdiff_t        shiftx,
                             rocblas_int      incx,
                             rocblas_stride   stridex,
                             W                w,
                             rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    tpmvt_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(w, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
__global__ void tpmvc_kernel(rocblas_fill     uplo,
                             rocblas_diagonal diag,
                             rocblas_int      m,
                             A                a,
                             ptrdiff_t        shifta,
                             rocblas_stride   stridea,
                             X                x,
                             ptrdiff_t        shiftx,
                             rocblas_int      incx,
                             rocblas_stride   stridex,
                             W                w,
                             rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    tpmvc_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(w, hipBlockIdx_y, shiftw, stridew));
}

#undef uat
#undef lat
