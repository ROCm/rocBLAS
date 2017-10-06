/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*
 * ===========================================================================
 *    This file provide common device function for ger routines
 * ===========================================================================
 */

/* ============================================================================================ */

#include "../blas1/device_template.h"

template<typename T>
static __device__ void
ger_device(
    rocblas_int m, rocblas_int n,
    T alpha,
    const T * __restrict__ x, rocblas_int incx,
    const T * __restrict__ y, rocblas_int incy,
          T *              A, rocblas_int lda)
{
    if (m <= 0 || n <= 0) return;

    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(incx >= 0 && incy >= 0)
    {
        if (tx < m && ty < n)
        {
            A[tx + lda*ty] += (alpha) * x[tx*incx] * y[ty*incy];
        }
    }
    else if(incx < 0 && incy < 0)
    {
        if (tx < m && ty < n)
        {
            A[tx + lda*ty] += (alpha) * x[(1 - m + tx)*incx] * y[(1 - n + ty)*incy];
        }
    }
    else if (incx >= 0)
    {
        if (tx < m && ty < n)
        {
            A[tx + lda*ty] += (alpha) * x[tx*incx] * y[(1 - n + ty)*incy];
        }
    }
    else
    {
        if (tx < m && ty < n)
        {
            A[tx + lda*ty] += (alpha) * x[(1 - m + tx)*incx] * y[ty*incy];
        }
    }
}
