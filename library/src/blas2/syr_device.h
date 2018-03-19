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

template <typename T>
static __device__ void syr_device(rocblas_fill uplo,
                                  rocblas_int n,
                                  T alpha,
                                  const T* __restrict__ x,
                                  rocblas_int incx,
                                  T* A,
                                  rocblas_int lda)
{
    if(n <= 0)
        return;

    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(uplo == rocblas_fill_lower)
    {
        if(tx < n && ty <= tx)
        {
            if(incx > 0)
            {
                A[tx + lda * ty] += alpha * x[tx * incx] * x[ty * incx];
            }
            else
            {

                A[tx + lda * ty] += alpha * x[(1 - n + tx) * incx] * x[(1 - n + ty) * incx];
            }
        }
    }
    else if(uplo == rocblas_fill_upper)
    {
        if(ty < n && tx <= ty)
        {
            if(incx > 0)
            {
                A[tx + lda * ty] += alpha * x[tx * incx] * x[ty * incx];
            }
            else
            {

                A[tx + lda * ty] += alpha * x[(1 - n + tx) * incx] * x[(1 - n + ty) * incx];
            }
        }
    }
}
