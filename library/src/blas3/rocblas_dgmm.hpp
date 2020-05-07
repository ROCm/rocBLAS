/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"

template <bool side_right, typename TConstPtr, typename TPtr>
__global__ void dgmm_device(rocblas_int    m,
                            rocblas_int    n,
                            TConstPtr      Aa,
                            rocblas_int    offset_a,
                            rocblas_int    lda,
                            rocblas_stride stride_a,
                            TConstPtr      Xa,
                            rocblas_int    shift_x,
                            rocblas_int    incx,
                            rocblas_stride stride_x,
                            TPtr           Ca,
                            rocblas_int    offset_c,
                            rocblas_int    ldc,
                            rocblas_stride stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);
        auto* X = load_ptr_batch(Xa, hipBlockIdx_z, shift_x, stride_x);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        if(side_right)
        {
            C[tx + ldc * ty] = A[tx + lda * ty] + X[ty * incx];
        }
        else
        {
            C[tx + ldc * ty] = A[tx + lda * ty] + X[tx * incx];
        }
    }
}

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call DGMM C interfaces (see rocblas_dgmm*.cpp in the same dir)
 * ===========================================================================
 */

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_template(rocblas_handle handle,
                                     rocblas_side   side,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     TConstPtr      A,
                                     rocblas_int    offset_a,
                                     rocblas_int    lda,
                                     rocblas_stride stride_a,
                                     TConstPtr      X,
                                     rocblas_int    offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     TPtr           C,
                                     rocblas_int    offset_c,
                                     rocblas_int    ldc,
                                     rocblas_stride stride_c,
                                     rocblas_int    batch_count)

{
    hipStream_t rocblas_stream = handle->rocblas_stream;

    auto pointer_mode = handle->pointer_mode;

    {
        // in case of negative incx shift pointer to end of data for negative indexing
        ptrdiff_t shift_x = offset_x - ((incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0);

        // general case, any transA, transB, lda, incx, ldc
        static constexpr int DGMM_DIM_X = 16;
        static constexpr int DGMM_DIM_Y = 16;

        rocblas_int blocksX = (m - 1) / DGMM_DIM_X + 1;
        rocblas_int blocksY = (n - 1) / DGMM_DIM_Y + 1;

        dim3 dgmm_grid(blocksX, blocksY, batch_count);
        dim3 dgmm_threads(DGMM_DIM_X, DGMM_DIM_Y);

        if(rocblas_side_left == side)
        {
            hipLaunchKernelGGL(dgmm_device<false>,
                               dgmm_grid,
                               dgmm_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               X,
                               shift_x,
                               incx,
                               stride_x,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
        else
        {
            hipLaunchKernelGGL(dgmm_device<true>,
                               dgmm_grid,
                               dgmm_threads,
                               0,
                               rocblas_stream,
                               m,
                               n,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               X,
                               shift_x,
                               incx,
                               stride_x,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
    }
    return rocblas_status_success;
}
