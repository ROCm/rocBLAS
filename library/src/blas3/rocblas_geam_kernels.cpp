/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "handle.hpp"
#include "rocblas_geam.hpp"

template <int DIM_X, int DIM_Y, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
geam_zero_matrix_device(rocblas_int    m,
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
        auto*  C       = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);
        size_t c_index = tx + size_t(ldc) * ty;
        C[c_index]     = 0.0;
    }
}

// general case for any alpha, beta, lda, ldb, ldc
template <int DIM_X, int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
geam_device(rocblas_operation transA,
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

        auto* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_z, offset_a, stride_a);
        auto* B = cond_load_ptr_batch(beta, Ba, hipBlockIdx_z, offset_b, stride_b);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        size_t a_index;
        size_t b_index;
        size_t c_index = tx + size_t(ldc) * ty;

        if(transA == rocblas_operation_none)
        {
            a_index = tx + ty * size_t(lda);
        }
        else
        {
            a_index = tx * size_t(lda) + ty;
        }

        if(transB == rocblas_operation_none)
        {
            b_index = tx + ty * size_t(ldb);
        }
        else
        {
            b_index = tx * size_t(ldb) + ty;
        }

        auto a_val = alpha ? A[a_index] : 0;
        auto b_val = beta ? B[b_index] : 0;
        if(transA == rocblas_operation_conjugate_transpose)
            a_val = conj(a_val);
        if(transB == rocblas_operation_conjugate_transpose)
            b_val = conj(b_val);

        C[c_index] = beta * b_val + alpha * a_val;
    }
}

//  special case:
//  only one matrix contributes because   0 == alpha || 0 == beta
template <int DIM_X, int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
geam_2matrix_device(rocblas_operation transA,
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

        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        size_t c_index = tx + size_t(ldc) * ty;
        if(alpha == 0)
        {
            C[c_index] = 0;
        }
        else
        {
            auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);

            size_t a_index;

            if(transA == rocblas_operation_none)
            {
                a_index = tx + ty * size_t(lda);
            }
            else
            {
                a_index = tx * size_t(lda) + ty;
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
template <int DIM_X, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X)
geam_1D_device(size_t         size,
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
    size_t tx = size_t(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;

    if(tx < size)
    {
        auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, 0);
        auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, 0);

        auto* C = load_ptr_batch(Ca, hipBlockIdx_y, offset_c, stride_c);

        if(alpha == 0 && beta == 0)
        {
            C[tx] = 0;
        }
        else
        {
            auto* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, offset_a, stride_a);
            auto* B = cond_load_ptr_batch(beta, Ba, hipBlockIdx_y, offset_b, stride_b);

            C[tx] = (beta ? beta * B[tx] : 0) + (alpha ? alpha * A[tx] : 0);
        }
    }
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors.
// Also, alpha == 0  ||  beta == 0  so only one matrix contributes
template <int DIM_X, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X)
geam_1D_2matrix_device(size_t         size,
                       TScal          alpha_device_host,
                       TConstPtr      Aa,
                       rocblas_int    offset_a,
                       rocblas_stride stride_a,
                       TPtr           Ca,
                       rocblas_int    offset_c,
                       rocblas_stride stride_c)
{
    size_t tx = size_t(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;

    if(tx < size)
    {
        auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, 0);

        auto* C = load_ptr_batch(Ca, hipBlockIdx_y, offset_c, stride_c);

        if(alpha == 0)
        {
            C[tx] = 0;
        }
        else
        {
            auto* A = load_ptr_batch(Aa, hipBlockIdx_y, offset_a, stride_a);
            C[tx]   = alpha * A[tx];
        }
    }
}

// special cases where: A == C && lda == ldc && transA == none
// this is in place case C <- alpha*C + beta*B
template <int DIM_X, int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
geam_inplace_device(rocblas_operation transB,
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

        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        size_t b_index;
        size_t c_index = tx + size_t(ldc) * ty;

        if(beta == 0)
        {
            C[c_index] = alpha ? alpha * C[c_index] : 0;
        }
        else
        {
            auto* B = load_ptr_batch(Ba, hipBlockIdx_z, offset_b, stride_b);

            if(transB == rocblas_operation_none)
            {
                b_index = tx + ty * size_t(ldb);
            }
            else
            {
                b_index = tx * size_t(ldb) + ty;
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
    hipStream_t rocblas_stream = handle->get_stream();

    auto pointer_mode = handle->pointer_mode;
    if(pointer_mode == rocblas_pointer_mode_host && !*alpha && !*beta)
    {
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;

        rocblas_int blocksX = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int blocksY = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY, batch_count);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        hipLaunchKernelGGL((geam_zero_matrix_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
            hipLaunchKernelGGL((geam_inplace_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
            hipLaunchKernelGGL((geam_inplace_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
            hipLaunchKernelGGL((geam_inplace_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
            hipLaunchKernelGGL((geam_inplace_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
            size_t               size     = size_t(m) * n;
            int                  blocks   = (size - 1) / GEAM_DIM + 1;

            dim3 geam_grid(blocks, batch_count);
            dim3 geam_threads(GEAM_DIM);

            hipLaunchKernelGGL((geam_1D_2matrix_device<GEAM_DIM>),
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

            hipLaunchKernelGGL((geam_2matrix_device<GEAM_DIM_X, GEAM_DIM_Y>),
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

            hipLaunchKernelGGL((geam_1D_2matrix_device<GEAM_DIM>),
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

            hipLaunchKernelGGL((geam_2matrix_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
        size_t               size     = size_t(m) * n;
        int                  blocks   = (size - 1) / GEAM_DIM + 1;
        // GEAM_DIM needs to be large to prevent blocks overflowing int datatype.

        dim3 geam_grid(blocks, batch_count);
        dim3 geam_threads(GEAM_DIM);

        if(rocblas_pointer_mode_host == pointer_mode)
        {
            hipLaunchKernelGGL((geam_1D_device<GEAM_DIM>),
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
            hipLaunchKernelGGL((geam_1D_device<GEAM_DIM>),
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
            hipLaunchKernelGGL((geam_device<GEAM_DIM_X, GEAM_DIM_Y>),
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
            hipLaunchKernelGGL((geam_device<GEAM_DIM_X, GEAM_DIM_Y>),
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

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_geam_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           TConstPtr         A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           TConstPtr         B,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           TPtr              C,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count,
                                           const int         check_numerics,
                                           bool              is_input)
{

    rocblas_status check_numerics_status = rocblas_status_success;

    if(is_input)
    {
        check_numerics_status = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                                                   handle,
                                                                                   trans_a,
                                                                                   m,
                                                                                   n,
                                                                                   A,
                                                                                   0,
                                                                                   lda,
                                                                                   stride_a,
                                                                                   batch_count,
                                                                                   check_numerics,
                                                                                   is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                                                   handle,
                                                                                   trans_b,
                                                                                   m,
                                                                                   n,
                                                                                   B,
                                                                                   0,
                                                                                   ldb,
                                                                                   stride_b,
                                                                                   batch_count,
                                                                                   check_numerics,
                                                                                   is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }
    check_numerics_status
        = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                             handle,
                                                             rocblas_operation_none,
                                                             m,
                                                             n,
                                                             C,
                                                             0,
                                                             ldc,
                                                             stride_c,
                                                             batch_count,
                                                             check_numerics,
                                                             is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files geam*.cpp

// clang-format off
#ifdef INSTANTIATE_GEAM_TEMPLATE
#error INSTANTIATE_GEAM_TEMPLATE already defined
#endif

#define INSTANTIATE_GEAM_TEMPLATE(TScal_, TConstPtr_, TPtr_)              \
template rocblas_status rocblas_geam_template<TScal_, TConstPtr_, TPtr_>  \
                                    (rocblas_handle    handle,            \
                                     rocblas_operation transA,            \
                                     rocblas_operation transB,            \
                                     rocblas_int       m,                 \
                                     rocblas_int       n,                 \
                                     TScal_            alpha,             \
                                     TConstPtr_        A,                 \
                                     rocblas_int       offset_a,          \
                                     rocblas_int       lda,               \
                                     rocblas_stride    stride_a,          \
                                     TScal_            beta,              \
                                     TConstPtr_        B,                 \
                                     rocblas_int       offset_b,          \
                                     rocblas_int       ldb,               \
                                     rocblas_stride    stride_b,          \
                                     TPtr_             C,                 \
                                     rocblas_int       offset_c,          \
                                     rocblas_int       ldc,               \
                                     rocblas_stride    stride_c,          \
                                     rocblas_int       batch_count);

// instantiate for rocblas_Xgeam and rocblas_Xgeam_strided_batched
INSTANTIATE_GEAM_TEMPLATE( float const*,  float const*,  float*)
INSTANTIATE_GEAM_TEMPLATE(double const*, double const*, double*)
INSTANTIATE_GEAM_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_GEAM_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xgeam_batched
INSTANTIATE_GEAM_TEMPLATE( float const*, float const* const*, float* const*)
INSTANTIATE_GEAM_TEMPLATE(double const*, double const* const*, double* const*)
INSTANTIATE_GEAM_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_GEAM_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_GEAM_TEMPLATE

#ifdef INSTANTIATE_GEAM_NUMERICS
#error INSTANTIATE_GEAM_NUMERICS already defined
#endif

#define INSTANTIATE_GEAM_NUMERICS(TConstPtr_, TPtr_)                         \
template rocblas_status rocblas_geam_check_numerics<TConstPtr_, TPtr_>       \
                                          (const char*       function_name,  \
                                           rocblas_handle    handle,         \
                                           rocblas_operation trans_a,        \
                                           rocblas_operation trans_b,        \
                                           rocblas_int       m,              \
                                           rocblas_int       n,              \
                                           TConstPtr_        A,              \
                                           rocblas_int       lda,            \
                                           rocblas_stride    stride_a,       \
                                           TConstPtr_        B,              \
                                           rocblas_int       ldb,            \
                                           rocblas_stride    stride_b,       \
                                           TPtr_             C,              \
                                           rocblas_int       ldc,            \
                                           rocblas_stride    stride_c,       \
                                           rocblas_int       batch_count,    \
                                           const int         check_numerics, \
                                           bool              is_input);

// instantiate for rocblas_Xgeam and rocblas_Xgeam_strided_batched
INSTANTIATE_GEAM_NUMERICS(float const*,  float*)
INSTANTIATE_GEAM_NUMERICS(double const*, double*)
INSTANTIATE_GEAM_NUMERICS(rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_GEAM_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xgeam_batched
INSTANTIATE_GEAM_NUMERICS(float const* const*, float* const*)
INSTANTIATE_GEAM_NUMERICS(double const* const*, double* const*)
INSTANTIATE_GEAM_NUMERICS(rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_GEAM_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_GEAM_NUMERICS
// clang-format on
