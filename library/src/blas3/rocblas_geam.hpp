/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"

template <typename TPtr>
__global__ void geam_zero_matrix_device(rocblas_int    m,
                                        rocblas_int    n,
                                        TPtr           Ca,
                                        rocblas_int    offset_c,
                                        rocblas_int    ldc,
                                        rocblas_stride stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto* C       = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);
        int   c_index = tx + ldc * ty;
        C[c_index]    = 0.0;
    }
}

// general case for any alpha, beta, lda, ldb, ldc
template <typename TScal, typename TConstPtr, typename TPtr>
__global__ void geam_device(rocblas_operation transA,
                            rocblas_operation transB,
                            rocblas_int       m,
                            rocblas_int       n,
                            TScal             alpha_device_host,
                            TConstPtr         Aa,
                            rocblas_int       offset_a,
                            rocblas_int       lda,
                            rocblas_stride    stride_a,
                            TScal             beta_device_host,
                            TConstPtr         Ba,
                            rocblas_int       offset_b,
                            rocblas_int       ldb,
                            rocblas_stride    stride_b,
                            TPtr              Ca,
                            rocblas_int       offset_c,
                            rocblas_int       ldc,
                            rocblas_stride    stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, 0);
        auto beta  = load_scalar(beta_device_host, hipBlockIdx_z, 0);

        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);
        auto* B = load_ptr_batch(Ba, hipBlockIdx_z, offset_b, stride_b);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        int a_index;
        int b_index;
        int c_index = tx + ldc * ty;

        if(transA == rocblas_operation_none)
        {
            a_index = tx + ty * lda;
        }
        else
        {
            a_index = tx * lda + ty;
        }

        if(transB == rocblas_operation_none)
        {
            b_index = tx + ty * ldb;
        }
        else
        {
            b_index = tx * ldb + ty;
        }

        auto a_val = A[a_index];
        auto b_val = B[b_index];
        if(transA == rocblas_operation_conjugate_transpose)
            a_val = conj(a_val);
        if(transB == rocblas_operation_conjugate_transpose)
            b_val = conj(b_val);

        C[c_index] = beta * b_val + alpha * a_val;
    }
}

//  special case:
//  only one matrix contributes because   0 == alpha || 0 == beta
template <typename TScal, typename TConstPtr, typename TPtr>
__global__ void geam_2matrix_device(rocblas_operation transA,
                                    rocblas_int       m,
                                    rocblas_int       n,
                                    TScal             alpha_device_host,
                                    TConstPtr         Aa,
                                    rocblas_int       offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    TPtr              Ca,
                                    rocblas_int       offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, 0);

        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        int c_index = tx + ldc * ty;
        if(alpha == 0)
        {
            C[c_index] = 0;
        }
        else
        {
            int a_index;

            if(transA == rocblas_operation_none)
            {
                a_index = tx + ty * lda;
            }
            else
            {
                a_index = tx * lda + ty;
            }

            auto a_val = A[a_index];
            if(transA == rocblas_operation_conjugate_transpose)
                a_val = conj(a_val);
            C[c_index] = alpha * a_val;
        }
    }
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors
template <typename TScal, typename TConstPtr, typename TPtr>
__global__ void geam_1D_device(rocblas_int    size,
                               TScal          alpha_device_host,
                               TConstPtr      Aa,
                               rocblas_int    offset_a,
                               rocblas_stride stride_a,
                               TScal          beta_device_host,
                               TConstPtr      Ba,
                               rocblas_int    offset_b,
                               rocblas_stride stride_b,
                               TPtr           Ca,
                               rocblas_int    offset_c,
                               rocblas_stride stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tx < size)
    {
        auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, 0);
        auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, 0);

        auto* A = load_ptr_batch(Aa, hipBlockIdx_y, offset_a, stride_a);
        auto* B = load_ptr_batch(Ba, hipBlockIdx_y, offset_b, stride_b);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_y, offset_c, stride_c);

        if(alpha == 0 && beta == 0)
        {
            C[tx] = 0;
        }
        else
        {
            C[tx] = beta * B[tx] + alpha * A[tx];
        }
    }
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors.
// Also, alpha == 0  ||  beta == 0  so only one matrix contributes
template <typename TScal, typename TConstPtr, typename TPtr>
__global__ void geam_1D_2matrix_device(rocblas_int    size,
                                       TScal          alpha_device_host,
                                       TConstPtr      Aa,
                                       rocblas_int    offset_a,
                                       rocblas_stride stride_a,
                                       TPtr           Ca,
                                       rocblas_int    offset_c,
                                       rocblas_stride stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tx < size)
    {
        auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, 0);

        auto* A = load_ptr_batch(Aa, hipBlockIdx_y, offset_a, stride_a);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_y, offset_c, stride_c);

        if(alpha == 0)
        {
            C[tx] = 0;
        }
        else
        {
            C[tx] = alpha * A[tx];
        }
    }
}
// special cases where: A == C && lda == ldc && transA == none
// this is in place case C <- alpha*C + beta*B
template <typename TScal, typename TConstPtr, typename TPtr>
__global__ void geam_inplace_device(rocblas_operation transB,
                                    rocblas_int       m,
                                    rocblas_int       n,
                                    TScal             alpha_device_host,
                                    TScal             beta_device_host,
                                    TConstPtr         Ba,
                                    rocblas_int       offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    TPtr              Ca,
                                    rocblas_int       offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto alpha = load_scalar(alpha_device_host, 0, 0);
        auto beta  = load_scalar(beta_device_host, 0, 0);

        auto* B = load_ptr_batch(Ba, hipBlockIdx_z, offset_b, stride_b);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        int b_index;
        int c_index = tx + ldc * ty;

        if(beta == 0)
        {
            C[c_index] = alpha * C[c_index];
        }
        else
        {
            if(transB == rocblas_operation_none)
            {
                b_index = tx + ty * ldb;
            }
            else
            {
                b_index = tx * ldb + ty;
            }

            auto b_val = B[b_index];
            if(transB == rocblas_operation_conjugate_transpose)
                b_val = conj(b_val);

            if(alpha == 0)
            {
                C[c_index] = beta * b_val;
            }
            else
            {
                C[c_index] = beta * b_val + alpha * C[c_index];
            }
        }
    }
}

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call GEAM C interfaces (see geam.cpp in the same dir)
 * ===========================================================================
 */

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_geam_template(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal             alpha,
                                     TConstPtr         A,
                                     rocblas_int       offset_a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TScal             beta,
                                     TConstPtr         B,
                                     rocblas_int       offset_b,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_b,
                                     TPtr              C,
                                     rocblas_int       offset_c,
                                     rocblas_int       ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count)

{
    hipStream_t rocblas_stream = handle->rocblas_stream;

    auto pointer_mode = handle->pointer_mode;
    if(pointer_mode == rocblas_pointer_mode_host && !*alpha && !*beta)
    {
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;

        rocblas_int blocksX = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int blocksY = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY, batch_count);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        hipLaunchKernelGGL(geam_zero_matrix_device,
                           geam_grid,
                           geam_threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           C,
                           offset_c,
                           ldc,
                           stride_c);
    }
    else if(C == A)
    {
        // C <- alpha * C + beta * B
        // transA == rocblas_operation_none
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;
        rocblas_int          blocksX    = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int          blocksY    = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY, batch_count);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            hipLaunchKernelGGL(geam_inplace_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               *alpha,
                               *beta,
                               B,
                               offset_b,
                               ldb,
                               stride_b,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
        else
        {
            hipLaunchKernelGGL(geam_inplace_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               alpha,
                               beta,
                               B,
                               offset_b,
                               ldb,
                               stride_b,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
    }
    else if(C == B)
    {
        // C <- alpha * A + beta * C
        // transB == rocblas_operation_none
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;
        rocblas_int          blocksX    = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int          blocksY    = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY, batch_count);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            hipLaunchKernelGGL(geam_inplace_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               *beta,
                               *alpha,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
        else
        {
            hipLaunchKernelGGL(geam_inplace_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               beta,
                               alpha,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
    }
    else if(pointer_mode == rocblas_pointer_mode_host && !*beta)
    {
        if(m == lda && transA == rocblas_operation_none && m == ldc)
        {
            // beta == 0
            // special case: A, C are processed as vectors because
            // A, C are contiguous, and A is normal (not transpose)
            static constexpr int GEAM_DIM = 256;
            int                  size     = m * n;
            int                  blocks   = (size - 1) / GEAM_DIM + 1;

            dim3 geam_grid(blocks, batch_count);
            dim3 geam_threads(GEAM_DIM);

            hipLaunchKernelGGL(geam_1D_2matrix_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               *alpha,
                               A,
                               offset_a,
                               stride_a,
                               C,
                               offset_c,
                               stride_c);
        }
        else
        {
            // beta == 0
            // general case for any transA, lda, ldc
            static constexpr int GEAM_DIM_X = 16;
            static constexpr int GEAM_DIM_Y = 16;
            rocblas_int          blocksX    = (m - 1) / GEAM_DIM_X + 1;
            rocblas_int          blocksY    = (n - 1) / GEAM_DIM_Y + 1;

            dim3 geam_grid(blocksX, blocksY, batch_count);
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

            hipLaunchKernelGGL(geam_2matrix_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               *alpha,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
    }
    else if(rocblas_pointer_mode_host == pointer_mode && !*alpha)
    {
        if(m == ldb && transB == rocblas_operation_none && m == ldc)
        {
            // alpha == 0
            // special case: B, C are processed as vectors because
            // B, C are contiguous, and B is normal (not transpose)
            static constexpr int GEAM_DIM = 256;
            int                  size     = m * n;
            int                  blocks   = (size - 1) / GEAM_DIM + 1;

            dim3 geam_grid(blocks, batch_count);
            dim3 geam_threads(GEAM_DIM);

            hipLaunchKernelGGL(geam_1D_2matrix_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               *beta,
                               B,
                               offset_b,
                               stride_b,
                               C,
                               offset_c,
                               stride_c);
        }
        else
        {
            // alpha == 0
            // general case for any transB, ldb, ldc
            static constexpr int GEAM_DIM_X = 16;
            static constexpr int GEAM_DIM_Y = 16;

            rocblas_int blocksX = (m - 1) / GEAM_DIM_X + 1;
            rocblas_int blocksY = (n - 1) / GEAM_DIM_Y + 1;

            dim3 geam_grid(blocksX, blocksY, batch_count);
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

            hipLaunchKernelGGL(geam_2matrix_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               *beta,
                               B,
                               offset_b,
                               ldb,
                               stride_b,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
    }
    else if(m == lda && transA == rocblas_operation_none && m == ldb
            && transB == rocblas_operation_none && m == ldc)
    {
        // special case: A, B, C are processed as vectors because
        // A, B, C are contiguous, and A and B are normal (not transpose)
        static constexpr int GEAM_DIM = 256;
        int                  size     = m * n;
        int                  blocks   = (size - 1) / GEAM_DIM + 1;

        dim3 geam_grid(blocks, batch_count);
        dim3 geam_threads(GEAM_DIM);

        if(rocblas_pointer_mode_host == pointer_mode)
        {
            hipLaunchKernelGGL(geam_1D_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               *alpha,
                               A,
                               offset_a,
                               stride_a,
                               *beta,
                               B,
                               offset_b,
                               stride_b,
                               C,
                               offset_c,
                               stride_c);
        }
        else
        {
            hipLaunchKernelGGL(geam_1D_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               alpha,
                               A,
                               offset_a,
                               stride_a,
                               beta,
                               B,
                               offset_b,
                               stride_b,
                               C,
                               offset_c,
                               stride_c);
        }
    }
    else
    {
        // general case, any transA, transB, lda, ldb, ldc
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;

        rocblas_int blocksX = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int blocksY = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY, batch_count);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            hipLaunchKernelGGL(geam_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               transB,
                               m,
                               n,
                               *alpha,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               *beta,
                               B,
                               offset_b,
                               ldb,
                               stride_b,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
        else
        {
            hipLaunchKernelGGL(geam_device,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               transB,
                               m,
                               n,
                               alpha,
                               A,
                               offset_a,
                               lda,
                               stride_a,
                               beta,
                               B,
                               offset_b,
                               ldb,
                               stride_b,
                               C,
                               offset_c,
                               ldc,
                               stride_c);
        }
    }

    return rocblas_status_success;
}
