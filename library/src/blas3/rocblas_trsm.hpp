/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_TRSM_HPP__
#define __ROCBLAS_TRSM_HPP__

#include "../blas_ex/rocblas_gemm_ex.hpp"
#include "trtri_trsm.hpp"

template <typename T>
static const T negative_one = T(-1);
template <typename T>
static const T zero = T(0);
template <typename T>
static const T one = T(1);

template <typename T, typename U, typename V>
__global__ void copy_matrix_trsm(rocblas_int    rows,
                                 rocblas_int    cols,
                                 rocblas_int    elem_size,
                                 U              a,
                                 rocblas_int    lda,
                                 rocblas_stride stride_a,
                                 V              b,
                                 rocblas_int    ldb,
                                 rocblas_stride stride_b,
                                 rocblas_int    offset_a,
                                 rocblas_int    offset_b)
{
    const T* xa = load_ptr_batch(a, hipBlockIdx_z, offset_a, stride_a);
    T*       xb = load_ptr_batch(b, hipBlockIdx_z, offset_b, stride_b);

    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < rows && ty < cols)
        xb[tx + ldb * ty] = xa[tx + lda * ty];
}

/* ===============copy helper============================================= */
template <typename T, typename U, typename V>
void copy_block_unit(rocblas_handle handle,
                     rocblas_int    m,
                     rocblas_int    n,
                     U              src,
                     rocblas_int    src_ld,
                     rocblas_stride src_stride,
                     V              dst,
                     rocblas_int    dst_ld,
                     rocblas_stride dst_stride,
                     rocblas_int    batch_count,
                     rocblas_int    offset_src = 0,
                     rocblas_int    offset_dst = 0)
{
    rocblas_int blocksX = (m - 1) / 128 + 1; // parameters for device kernel
    rocblas_int blocksY = (n - 1) / 8 + 1;
    dim3        grid(blocksX, blocksY, batch_count);
    dim3        threads(128, 8);

    hipLaunchKernelGGL(copy_matrix_trsm<T>,
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       m,
                       n,
                       sizeof(T),
                       src,
                       src_ld,
                       src_stride,
                       dst,
                       dst_ld,
                       dst_stride,
                       offset_src,
                       offset_dst);
}

template <typename T, typename U>
__global__ void set_matrix_trsm(rocblas_int    rows,
                                rocblas_int    cols,
                                rocblas_int    elem_size,
                                U              a,
                                rocblas_int    lda,
                                rocblas_stride stride_a,
                                T              val,
                                rocblas_int    offset_a)
{
    T* xa = load_ptr_batch(a, hipBlockIdx_z, offset_a, stride_a);

    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < rows && ty < cols)
        xa[tx + lda * ty] = T(0.0);
}

/* ===============set helper============================================= */
template <typename T, typename U>
void set_block_unit(rocblas_handle handle,
                    rocblas_int    m,
                    rocblas_int    n,
                    U              src,
                    rocblas_int    src_ld,
                    rocblas_stride src_stride,
                    rocblas_int    batch_count,
                    T              val        = 0.0,
                    rocblas_int    offset_src = 0)
{
    rocblas_int blocksX = (m - 1) / 128 + 1; // parameters for device kernel
    rocblas_int blocksY = (n - 1) / 8 + 1;
    dim3        grid(blocksX, blocksY, batch_count);
    dim3        threads(128, 8);

    hipLaunchKernelGGL(set_matrix_trsm<T>,
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       m,
                       n,
                       sizeof(T),
                       src,
                       src_ld,
                       src_stride,
                       val,
                       offset_src);
}

/* ===============left==================================================== */

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_trsm_left(rocblas_handle    handle,
                                 rocblas_fill      uplo,
                                 rocblas_operation transA,
                                 rocblas_int       m,
                                 rocblas_int       n,
                                 const T*          alpha,
                                 U                 A,
                                 rocblas_int       offset_Ain,
                                 rocblas_int       lda,
                                 rocblas_stride    stride_A,
                                 V                 B,
                                 rocblas_int       offset_Bin,
                                 rocblas_int       ldb,
                                 rocblas_stride    stride_B,
                                 rocblas_int       batch_count,
                                 U                 invA,
                                 rocblas_int       offset_invAin,
                                 rocblas_stride    stride_invA,
                                 V                 X,
                                 rocblas_stride    stride_X)
{
    rocblas_int i, jb;

    // transB is always non-transpose
    static constexpr rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // left, lower no-transpose
            jb = std::min(BLOCK, m);
            rocblas_gemm_template<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           &zero<T>,
                                           X,
                                           0,
                                           m,
                                           stride_X,
                                           batch_count);

            if(BLOCK < m)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transA,
                                               transB,
                                               m - BLOCK,
                                               n,
                                               BLOCK,
                                               &negative_one<T>,
                                               A,
                                               BLOCK + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               0,
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               BLOCK + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);
                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = std::min(m - i, BLOCK);

                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   jb,
                                                   n,
                                                   jb,
                                                   &one<T>,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &zero<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= m) // this condition is not necessary at all and can be changed
                        // as if (i+BLOCK<m)
                        break;

                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   m - i - BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   A,
                                                   i + BLOCK + i * lda + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &one<T>,
                                                   B,
                                                   i + BLOCK + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }

#if 0
            for( i=0; i < m; i += BLOCK ) {
                jb = std::min(m-i, BLOCK);
                T *tmp = (i == 0) ? alpha : one;
                rocblas_gemm_template<false>(handle, transA, transB, jb, n, jb, tmp, invA(i), BLOCK, stride_invA, B(i,0), ldb, stride_B, &zero<T>, X(i,0), ldb, stride_X, batch_count); // strides?
                if(i + BLOCK < m){
                    rocblas_gemm_template<false>(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negative_one<T>, A(i+BLOCK,i), lda, stride_A, X(i,0), ldb, stride_X, tmp, B(i+BLOCK,0), ldb, stride_B, batch_count); // strides?
                }
            }

#endif
        }
        else
        {
            // left, upper no-transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;

            // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
            rocblas_gemm_template<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           i * BLOCK + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           i + offset_Bin,
                                           ldb,
                                           stride_B,
                                           &zero<T>,
                                           X,
                                           i,
                                           m,
                                           stride_X,
                                           batch_count);

            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transA,
                                               transB,
                                               i,
                                               n,
                                               jb,
                                               &negative_one<T>,
                                               A,
                                               i * lda + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               i,
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    //{32, 35, 32, 32, 35, 35}
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &one<T>,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &zero<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   i,
                                                   n,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   A,
                                                   i * lda + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &one<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // left, lower transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;
            rocblas_gemm_template<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           i * BLOCK + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           i + offset_Bin,
                                           ldb,
                                           stride_B,
                                           &zero<T>,
                                           X,
                                           i,
                                           m,
                                           stride_X,
                                           batch_count);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transA,
                                               transB,
                                               i,
                                               n,
                                               jb,
                                               &negative_one<T>,
                                               A,
                                               i + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               i,
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &one<T>,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &zero<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   i,
                                                   n,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   A,
                                                   i + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &one<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
        else
        {
            // left, upper transpose
            jb = std::min(BLOCK, m);
            rocblas_gemm_template<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           &zero<T>,
                                           X,
                                           0,
                                           m,
                                           stride_X,
                                           batch_count);
            if(BLOCK < m)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transA,
                                               transB,
                                               m - BLOCK,
                                               n,
                                               BLOCK,
                                               &negative_one<T>,
                                               A,
                                               BLOCK * lda + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               0,
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               BLOCK + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = std::min(m - i, BLOCK);
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   jb,
                                                   n,
                                                   jb,
                                                   &one<T>,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &zero<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= m)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   m - i - BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   A,
                                                   i + (i + BLOCK) * lda + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &one<T>,
                                                   B,
                                                   i + BLOCK + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    } // transpose

    return rocblas_status_success;
}

/* ===============right==================================================== */

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_trsm_right(rocblas_handle    handle,
                                  rocblas_fill      uplo,
                                  rocblas_operation transA,
                                  rocblas_int       m,
                                  rocblas_int       n,
                                  const T*          alpha,
                                  U                 A,
                                  rocblas_int       offset_Ain,
                                  rocblas_int       lda,
                                  rocblas_stride    stride_A,
                                  V                 B,
                                  rocblas_int       offset_Bin,
                                  rocblas_int       ldb,
                                  rocblas_stride    stride_B,
                                  rocblas_int       batch_count,
                                  U                 invA,
                                  rocblas_int       offset_invAin,
                                  rocblas_stride    stride_invA,
                                  V                 X,
                                  rocblas_stride    stride_X)
{
    rocblas_int i, jb;

    // transB is always non-transpose
    static constexpr rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // right, lower no-transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i  = n - jb;
            rocblas_gemm_template<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           U(B),
                                           i * ldb + offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           i * BLOCK + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &zero<T>,
                                           X,
                                           i * m,
                                           m,
                                           stride_X,
                                           batch_count);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               i,
                                               jb,
                                               &negative_one<T>,
                                               (U)X,
                                               i * m,
                                               m,
                                               stride_X,
                                               A,
                                               i + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   BLOCK,
                                                   BLOCK,
                                                   &one<T>,
                                                   (U)B,
                                                   i * ldb + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &zero<T>,
                                                   X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   i,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   (U)X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   A,
                                                   i + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &one<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
        else
        {
            // right, upper no-transpose
            jb = std::min(BLOCK, n);
            rocblas_gemm_template<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           (U)B,
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &zero<T>,
                                           X,
                                           0,
                                           m,
                                           stride_X,
                                           batch_count);
            if(BLOCK < n)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               n - BLOCK,
                                               BLOCK,
                                               &negative_one<T>,
                                               (U)X,
                                               0,
                                               m,
                                               stride_X,
                                               A,
                                               BLOCK * lda + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               BLOCK * ldb + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < n; i += BLOCK)
                {
                    jb = std::min(BLOCK, n - i);
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   jb,
                                                   jb,
                                                   &one<T>,
                                                   (U)B,
                                                   i * ldb + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &zero<T>,
                                                   X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= n)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   n - i - BLOCK,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   (U)X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   A,
                                                   i + (i + BLOCK) * lda + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &one<T>,
                                                   B,
                                                   (i + BLOCK) * ldb + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // right, lower transpose
            jb = std::min(BLOCK, n);
            rocblas_gemm_template<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           U(B),
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &zero<T>,
                                           X,
                                           0,
                                           m,
                                           stride_X,
                                           batch_count);
            if(BLOCK < n)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               n - BLOCK,
                                               BLOCK,
                                               &negative_one<T>,
                                               U(X),
                                               0,
                                               m,
                                               stride_X,
                                               A,
                                               BLOCK + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               BLOCK * ldb + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < n; i += BLOCK)
                {
                    jb = std::min(BLOCK, n - i);
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   jb,
                                                   jb,
                                                   &one<T>,
                                                   (U)B,
                                                   i * ldb + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &zero<T>,
                                                   X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= n)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   n - i - BLOCK,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   (U)X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   A,
                                                   BLOCK + i + i * lda + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &one<T>,
                                                   B,
                                                   (i + BLOCK) * ldb + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
        else
        {
            // right, upper transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i  = n - jb;
            rocblas_gemm_template<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           (U)B,
                                           i * ldb + offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           i * BLOCK + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &zero<T>,
                                           X,
                                           i * m,
                                           m,
                                           stride_X,
                                           batch_count);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               i,
                                               jb,
                                               &negative_one<T>,
                                               (U)X,
                                               i * m,
                                               m,
                                               stride_X,
                                               A,
                                               i * lda + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   BLOCK,
                                                   BLOCK,
                                                   &one<T>,
                                                   (U)B,
                                                   i * ldb + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * BLOCK + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &zero<T>,
                                                   X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   i,
                                                   BLOCK,
                                                   &negative_one<T>,
                                                   (U)X,
                                                   i * m,
                                                   m,
                                                   stride_X,
                                                   A,
                                                   i * lda + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &one<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    } // tranpsose

    return rocblas_status_success;
}

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
rocblas_status special_trsm_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     U                 A,
                                     rocblas_int       offset_Ain,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_A,
                                     V                 B,
                                     rocblas_int       offset_Bin,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_B,
                                     rocblas_int       batch_count,
                                     U                 invA,
                                     rocblas_int       offset_invAin,
                                     rocblas_stride    stride_invA,
                                     size_t            B_chunk_size,
                                     V                 x_temp,
                                     rocblas_stride    stride_X)
{
    bool   parity = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_upper);
    size_t k      = side == rocblas_side_left ? m : n;
    size_t R      = k / BLOCK;
    size_t bsize  = side == rocblas_side_left ? n : m;
    size_t W      = 1 + (bsize - 1) / B_chunk_size;

    for(size_t w = 0; w < W; w++)
    {
        size_t width = std::min(bsize - w * B_chunk_size, B_chunk_size);

        if(side == rocblas_side_left)
        {
            for(size_t r = 0; r < R; r++)
            {
                size_t q = R - 1 - r;
                size_t j = parity ? r : q;

                // copy a BLOCK*n piece we are solving at a time
                if(!r || !tensile_supports_ldc_ne_ldd())
                    copy_block_unit<T>(handle,
                                       BLOCK,
                                       width,
                                       B,
                                       ldb,
                                       stride_B,
                                       x_temp,
                                       BLOCK,
                                       stride_X,
                                       batch_count,
                                       j * BLOCK + w * B_chunk_size * ldb + offset_Bin,
                                       0);

                if(r)
                {
                    rocblas_int offsetA = 0;
                    rocblas_int offsetB = parity ? w * B_chunk_size * ldb
                                                 : w * B_chunk_size * ldb + (q + 1) * BLOCK;

                    if(transA == rocblas_operation_none)
                        offsetA = parity ? r * BLOCK : BLOCK * (q * lda + q + lda);
                    else
                        offsetA = parity ? r * BLOCK * lda : BLOCK * (q * lda + q + 1);

                    if(!tensile_supports_ldc_ne_ldd())
                    {
                        rocblas_gemm_template<BATCHED>(handle,
                                                       transA,
                                                       rocblas_operation_none,
                                                       BLOCK,
                                                       width,
                                                       r * BLOCK,
                                                       &negative_one<T>,
                                                       A,
                                                       offsetA + offset_Ain,
                                                       lda,
                                                       stride_A,
                                                       (U)B,
                                                       offsetB + offset_Bin,
                                                       ldb,
                                                       stride_B,
                                                       alpha,
                                                       x_temp,
                                                       0,
                                                       BLOCK,
                                                       stride_X,
                                                       batch_count);
                    }
                    else
                    {
                        rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                        int32_t           solution_index = 0;
                        uint32_t          flags          = 0;

                        rocblas_gemm_ex_template<BATCHED>(handle,
                                                          transA,
                                                          rocblas_operation_none,
                                                          BLOCK,
                                                          width,
                                                          r * BLOCK,
                                                          &negative_one<T>,
                                                          A,
                                                          compute_type,
                                                          offsetA + offset_Ain,
                                                          lda,
                                                          stride_A,
                                                          B,
                                                          compute_type,
                                                          offsetB + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          alpha,
                                                          B,
                                                          compute_type,
                                                          j * BLOCK + w * B_chunk_size * ldb
                                                              + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          x_temp,
                                                          compute_type,
                                                          0,
                                                          BLOCK,
                                                          stride_X,
                                                          batch_count,
                                                          compute_type);
                    }
                }

                rocblas_gemm_template<BATCHED>(handle,
                                               transA,
                                               rocblas_operation_none,
                                               BLOCK,
                                               width,
                                               BLOCK,
                                               r ? &one<T> : alpha,
                                               invA,
                                               j * BLOCK * BLOCK + offset_invAin,
                                               BLOCK,
                                               stride_invA,
                                               (U)x_temp,
                                               0,
                                               BLOCK,
                                               stride_X,
                                               &zero<T>,
                                               B,
                                               w * B_chunk_size * ldb + j * BLOCK + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);
            }
        }
        else
        {
            for(size_t r = 0; r < R; r++)
            {
                size_t q = R - 1 - r;
                size_t j = parity ? q : r;

                // copy a m*BLOCK piece we are solving at a time
                if(!r || !tensile_supports_ldc_ne_ldd())
                    copy_block_unit<T>(handle,
                                       width,
                                       BLOCK,
                                       B,
                                       ldb,
                                       stride_B,
                                       x_temp,
                                       width,
                                       stride_X,
                                       batch_count,
                                       j * BLOCK * ldb + w * B_chunk_size + offset_Bin,
                                       0);

                if(r)
                {
                    rocblas_int offsetA = 0;
                    rocblas_int offsetB
                        = parity ? w * B_chunk_size + (q + 1) * BLOCK * ldb : w * B_chunk_size;
                    if(transA == rocblas_operation_none)
                        offsetA = parity ? BLOCK * (q * lda + q + 1) : r * BLOCK * lda;
                    else
                        offsetA = parity ? BLOCK * (q * lda + q + lda) : r * BLOCK;

                    if(!tensile_supports_ldc_ne_ldd())
                    {
                        rocblas_gemm_template<BATCHED>(handle,
                                                       rocblas_operation_none,
                                                       transA,
                                                       width,
                                                       BLOCK,
                                                       r * BLOCK,
                                                       &negative_one<T>,
                                                       (U)B,
                                                       offsetB + offset_Bin,
                                                       ldb,
                                                       stride_B,
                                                       A,
                                                       offsetA + offset_Ain,
                                                       lda,
                                                       stride_A,
                                                       alpha,
                                                       x_temp,
                                                       0,
                                                       width,
                                                       stride_X,
                                                       batch_count);
                    }
                    else
                    {
                        rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                        int32_t           solution_index = 0;
                        uint32_t          flags          = 0;

                        rocblas_gemm_ex_template<BATCHED>(handle,
                                                          rocblas_operation_none,
                                                          transA,
                                                          width,
                                                          BLOCK,
                                                          r * BLOCK,
                                                          &negative_one<T>,
                                                          B,
                                                          compute_type,
                                                          offsetB + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          A,
                                                          compute_type,
                                                          offsetA + offset_Ain,
                                                          lda,
                                                          stride_A,
                                                          alpha,
                                                          B,
                                                          compute_type,
                                                          j * BLOCK * ldb + w * B_chunk_size
                                                              + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          x_temp,
                                                          compute_type,
                                                          0,
                                                          width,
                                                          stride_X,
                                                          batch_count,
                                                          compute_type);
                    }
                }

                rocblas_gemm_template<BATCHED>(handle,
                                               rocblas_operation_none,
                                               transA,
                                               width,
                                               BLOCK,
                                               BLOCK,
                                               r ? &one<T> : alpha,
                                               U(x_temp),
                                               0,
                                               width,
                                               stride_X,
                                               invA,
                                               j * BLOCK * BLOCK + offset_invAin,
                                               BLOCK,
                                               stride_invA,
                                               &zero<T>,
                                               B,
                                               w * B_chunk_size * ldb + j * BLOCK * ldb
                                                   + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);
            }
        }
    }

    return rocblas_status_success;
}

/**
 *  The purpose of this function is to allocate memory for trsm. It is added to remove
 *  memory allocation from the rocblas_trsm_template function, but also allow code reuse
 *  from the _impl functions.
 *
 *  Note that for the batched version of trsm, we are also allocating memory to store the
 *  arrays of pointers for invA and x_temp (mem_x_temp_arr, mem_invA_arr).
 */
template <rocblas_int BLOCK, bool BATCHED, typename T, typename U>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_trsm_template_mem(rocblas_handle handle,
                                                                 rocblas_side   side,
                                                                 rocblas_int    m,
                                                                 rocblas_int    n,
                                                                 rocblas_int    batch_count,
                                                                 void*&         mem_x_temp,
                                                                 void*&         mem_x_temp_arr,
                                                                 void*&         mem_invA,
                                                                 void*&         mem_invA_arr,
                                                                 U supplied_invA = nullptr,
                                                                 rocblas_int supplied_invA_size = 0)
{
    bool SUBSTITUTION_ENABLED = true;

    rocblas_status perf_status = rocblas_status_success;
    rocblas_int    k           = side == rocblas_side_left ? m : n;

    // bool is_small = k <= 64;
    bool is_small = (m <= 64 && n <= 64);
    if(SUBSTITUTION_ENABLED && is_small)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        return rocblas_status_success;
    }

    // Whether size is an exact multiple of blocksize
    const bool exact_blocks = (k % BLOCK) == 0;

    size_t invA_bytes   = 0;
    size_t invA_els     = 0;
    size_t c_temp_bytes = 0;
    size_t c_temp_els   = 0;

    // For user-supplied invA, check to make sure size is large enough
    // If not large enough, indicate degraded performance and ignore supplied invA
    if(supplied_invA && supplied_invA_size / BLOCK < k)
    {
        // One-time warning message
        static int msg = (rocblas_cerr << "WARNING: TRSM invA_size argument is too small; invA "
                                          "argument is being ignored; TRSM performance is degraded"
                                       << std::endl,
                          0);
        perf_status    = rocblas_status_perf_degraded;
        supplied_invA  = nullptr;
    }

    if(!supplied_invA)
    {
        // Only allocate bytes for invA if supplied_invA == nullptr or supplied_invA_size is too small
        invA_els   = BLOCK * k;
        invA_bytes = invA_els * sizeof(T) * batch_count;

        // When k < BLOCK, C is unnecessary for trtri
        c_temp_els   = (k / BLOCK) * ((BLOCK / 2) * (BLOCK / 2));
        c_temp_bytes = c_temp_els * sizeof(T);

        // For the TRTRI last diagonal block we need remainder space if k % BLOCK != 0
        if(!exact_blocks)
        {
            // TODO: Make this more accurate -- right now it's much larger than necessary
            size_t remainder_els   = ROCBLAS_TRTRI_NB * BLOCK * 2;
            size_t remainder_bytes = sizeof(T) * remainder_els;

            // C is the maximum of the temporary space needed for TRTRI
            c_temp_els   = std::max(c_temp_els, remainder_els);
            c_temp_bytes = c_temp_els * sizeof(T);
        }
    }

    // Chunk size for special algorithm
    size_t B_chunk_size = 0;

    // Temporary solution matrix
    size_t x_temp_els;
    size_t x_temp_bytes;

    if(exact_blocks)
    {
        // Optimal B_chunk_size is the orthogonal dimension to k
        B_chunk_size = size_t(m) + size_t(n) - size_t(k);

        // When k % BLOCK == 0, we only need BLOCK * B_chunk_size space
        x_temp_els   = BLOCK * B_chunk_size;
        x_temp_bytes = x_temp_els * sizeof(T) * batch_count;
    }
    else
    {
        // When k % BLOCK != 0, we need m * n space
        x_temp_els   = size_t(m) * n;
        x_temp_bytes = x_temp_els * sizeof(T) * batch_count;
    }

    // X and C temporaries can share space, so the maximum size is allocated
    size_t x_c_temp_bytes = std::max(x_temp_bytes, c_temp_bytes);
    size_t arrBytes       = BATCHED ? sizeof(T*) * batch_count : 0;
    size_t xarrBytes      = BATCHED ? sizeof(T*) * batch_count : 0;

    // If this is a device memory size query, set optimal size and return changed status
    if(handle->is_device_memory_size_query())
        return handle->set_optimal_device_memory_size(
            x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);

    // Attempt to allocate optimal memory size
    auto mem = handle->device_malloc(x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);

    if(!mem)
    {
        if(exact_blocks)
        {
            B_chunk_size   = 1; // Fall back on chunk size of 1 (like TRSV)
            x_temp_els     = BLOCK;
            x_temp_bytes   = x_temp_els * sizeof(T) * batch_count;
            x_c_temp_bytes = std::max(x_temp_bytes, c_temp_bytes);

            mem = handle->device_malloc(x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);
        }
        if(!mem)
            return rocblas_status_memory_error;

        // Mark performance as degraded
        perf_status = rocblas_status_perf_degraded;

        // One-time warning about degraded performance
        static int msg = (rocblas_cerr << "WARNING: Device memory allocation size is too small for "
                                          "TRSM; TRSM performance is degraded"
                                       << std::endl,
                          0);
    }

    std::tie(mem_x_temp, mem_x_temp_arr, mem_invA, mem_invA_arr) = mem;
    return perf_status;
}

/* T = float, double, etc.
 * SCAL = T* or T
 * ATYPE = const T* or const T* const *
 * BTYPE = T* or T* const *
 *
 *  Uses the substitution method to solve a small problem AX = B.
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
static __global__ void rocblas_trsm_small_right_device(rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_diagonal  diag,
                                                       int               m,
                                                       int               n,
                                                       SCAL              alpha_dev_host,
                                                       ATYPE             Aa,
                                                       ptrdiff_t         offset_A,
                                                       int               lda,
                                                       rocblas_stride    stride_A,
                                                       BTYPE             Ba,
                                                       ptrdiff_t         offset_B,
                                                       int               ldb,
                                                       rocblas_stride    stride_B)
{
    const int batchid = blockIdx.y;
    auto      A       = load_ptr_batch(Aa, batchid, offset_A, stride_A);
    auto      B       = load_ptr_batch(Ba, batchid, offset_B, stride_B);
    auto      alpha   = load_scalar(alpha_dev_host);

    bool      LOWER = uplo == rocblas_fill_lower;
    bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
    const int tx    = threadIdx.x;
    const int bx    = blockIdx.x;

    // max A column to read from
    int maxColA = NB - 1 > n - 1 ? n - 1 : NB - 1;
    // NB columns, unless last block, then do leftover
    const int maxColB = (bx < gridDim.x - 1) ? NB : m - bx * NB;

    // offset B into correct block row
    B += bx * NB;

    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    if(tx <= maxColA)
    {
        // Load A into sA, handle conjugation if necessary
        for(int i = 0; i <= maxColA; i++)
            sA[i * NB + tx] = (CONJ) ? conj(A[i * lda + tx]) : A[i * lda + tx];

        // set unit diagonal if needed
        if(diag == rocblas_diagonal_unit)
            sA[tx * NB + tx] = T(1.0);
    }

    if(tx < maxColB)
    {
        // Load B into sB and multiply by alpha
        for(int i = 0; i < n; i++)
            sB[i * NB + tx] = alpha * B[i * ldb + tx];
    }
    __syncthreads();

    // Solve for B in shared memory
    if(transA == rocblas_operation_none && uplo == rocblas_fill_upper)
    {
        for(int i = 0; i < NB; i++)
        {
            // Subtract previously solved parts
            for(int j = 0; j < i; j++)
                sB[i * NB + tx] -= sB[j * NB + tx] * sA[i * NB + j];
            // Solve
            sB[i * NB + tx] /= sA[i * NB + i];
        }
    }
    else if(transA == rocblas_operation_none && uplo == rocblas_fill_lower)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
                sB[i * NB + tx] -= sB[j * NB + tx] * sA[i * NB + j];
            sB[i * NB + tx] /= sA[i * NB + i];
        }
    }
    else if(uplo == rocblas_fill_upper)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
                sB[i * NB + tx] -= sB[j * NB + tx] * sA[j * NB + i];
            sB[i * NB + tx] /= sA[i * NB + i];
        }
    }
    else // lower (conjugate-)transpose
    {
        for(int i = 0; i < NB; i++)
        {
            for(int j = 0; j < i; j++)
                sB[i * NB + tx] -= sB[j * NB + tx] * sA[j * NB + i];
            sB[i * NB + tx] /= sA[i * NB + i];
        }
    }

    // Save shared memory back into B
    if(tx < maxColB)
    {
        for(int i = 0; i < n; i++)
            B[i * ldb + tx] = sB[i * NB + tx];
    }
}

/*
 *  Uses a substitution method to solve a small problem AX = B. This version uses
 *  less shared memory for double complex types (currently this means not using
 *  shared memory for A).
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
static __global__ void rocblas_trsm_small_64_right_device(rocblas_fill      uplo,
                                                          rocblas_operation transA,
                                                          rocblas_diagonal  diag,
                                                          int               m,
                                                          int               n,
                                                          SCAL              alpha_dev_host,
                                                          ATYPE             Aa,
                                                          ptrdiff_t         offset_A,
                                                          int               lda,
                                                          rocblas_stride    stride_A,
                                                          BTYPE             Ba,
                                                          ptrdiff_t         offset_B,
                                                          int               ldb,
                                                          rocblas_stride    stride_B)
{
    const int batchid = blockIdx.y;
    auto      A       = load_ptr_batch(Aa, batchid, offset_A, stride_A);
    auto      B       = load_ptr_batch(Ba, batchid, offset_B, stride_B);
    auto      alpha   = load_scalar(alpha_dev_host);

    bool      LOWER = uplo == rocblas_fill_lower;
    bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
    const int tx    = threadIdx.x;
    const int bx    = blockIdx.x;

    // max A column to read from
    int maxColA = NB - 1 > n - 1 ? n - 1 : NB - 1;
    // NB columns, unless last block, then do leftover
    const int maxColB = (bx < gridDim.x - 1) ? NB : m - bx * NB;

    // offset B into correct block row
    B += bx * NB;

    __shared__ T sB[NB * NB];

    if(tx < maxColB)
    {
        // Load B into sB and multiply by alpha
        for(int i = 0; i < n; i++)
            sB[i * NB + tx] = alpha * B[i * ldb + tx];
    }
    __syncthreads();

    // Solve for B in shared memory
    if(transA == rocblas_operation_none && uplo == rocblas_fill_upper)
    {
        for(int i = 0; i < NB; i++)
        {
            // Subtract previously solved parts
            for(int j = 0; j < i; j++)
            {
                T valA = A[i * lda + j];
                sB[i * NB + tx] -= sB[j * NB + tx] * valA;
            }
            // Solve
            if(diag != rocblas_diagonal_unit)
                sB[i * NB + tx] /= A[i * lda + i];
        }
    }
    else if(transA == rocblas_operation_none && uplo == rocblas_fill_lower)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
            {
                T valA = A[i * lda + j];
                sB[i * NB + tx] -= sB[j * NB + tx] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[i * NB + tx] /= A[i * lda + i];
        }
    }
    else if(uplo == rocblas_fill_upper)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
            {
                T valA = CONJ ? conj(A[j * lda + i]) : A[j * lda + i];
                sB[i * NB + tx] -= sB[j * NB + tx] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[i * NB + tx] /= CONJ ? conj(A[i * lda + i]) : A[i * lda + i];
        }
    }
    else // lower (conjugate-)transpose
    {
        for(int i = 0; i < NB; i++)
        {
            for(int j = 0; j < i; j++)
            {
                T valA = CONJ ? conj(A[j * lda + i]) : A[j * lda + i];
                sB[i * NB + tx] -= sB[j * NB + tx] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[i * NB + tx] /= CONJ ? conj(A[i * lda + i]) : A[i * lda + i];
        }
    }

    // Save shared memory back into B
    if(tx < maxColB)
    {
        for(int i = 0; i < n; i++)
            B[i * ldb + tx] = sB[i * NB + tx];
    }
}

/* T = float, double, etc.
 * SCAL = T* or T
 * ATYPE = const T* or const T* const *
 * BTYPE = T* or T* const *
 *
 * Uses the substitution method to solve a small problem XA = B.
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
static __global__ void rocblas_trsm_small_left_device(rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      int               m,
                                                      int               n,
                                                      SCAL              alpha_dev_host,
                                                      ATYPE             Aa,
                                                      ptrdiff_t         offset_A,
                                                      int               lda,
                                                      rocblas_stride    stride_A,
                                                      BTYPE             Ba,
                                                      ptrdiff_t         offset_B,
                                                      int               ldb,
                                                      rocblas_stride    stride_B)
{
    const int batchid = blockIdx.y;
    auto      A       = load_ptr_batch(Aa, batchid, offset_A, stride_A);
    auto      B       = load_ptr_batch(Ba, batchid, offset_B, stride_B);
    auto      alpha   = load_scalar(alpha_dev_host);

    bool      LOWER = uplo == rocblas_fill_lower;
    bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
    const int tx    = threadIdx.x;
    const int bx    = blockIdx.x;

    // max A column to read from
    int maxColA = NB - 1 > m - 1 ? m - 1 : NB - 1;
    // NB columns, unless last block, then do leftover
    const int maxColB = (bx < gridDim.x - 1) ? NB : n - bx * NB;

    // offset B into correct block column
    B += bx * NB * ldb;

    // shared A and shared B
    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];

    if(tx <= maxColA)
    {
        // Load A into sA, handle conjugation if necessary
        for(int i = 0; i <= maxColA; i++)
            sA[i * NB + tx] = (CONJ) ? conj(A[i * lda + tx]) : A[i * lda + tx];

        // set unit diagonal if needed
        if(diag == rocblas_diagonal_unit)
            sA[tx * NB + tx] = T(1.0);

        // Load B into sB and multiply by alpha
        for(int i = 0; i < maxColB; i++)
            sB[i * NB + tx] = alpha * B[i * ldb + tx];
    }
    __syncthreads();

    // Solve for B in shared memory
    if(LOWER && transA == rocblas_operation_none)
    {
        for(int i = 0; i < NB; i++)
        {
            // Subtract previously solved parts
            for(int j = 0; j < i; j++)
                sB[tx * NB + i] -= sB[tx * NB + j] * sA[j * NB + i];
            sB[tx * NB + i] /= sA[i * NB + i];
        }
    }
    else if(!LOWER && transA == rocblas_operation_none)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
                sB[tx * NB + i] -= sB[tx * NB + j] * sA[j * NB + i];
            sB[tx * NB + i] /= sA[i * NB + i];
        }
    }
    else if(LOWER)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
                sB[tx * NB + i] -= sB[tx * NB + j] * sA[i * NB + j];
            sB[tx * NB + i] /= sA[i * NB + i];
        }
    }
    else if(!LOWER)
    {
        for(int i = 0; i < NB; i++)
        {
            for(int j = 0; j < i; j++)
                sB[tx * NB + i] -= sB[tx * NB + j] * sA[i * NB + j];
            sB[tx * NB + i] /= sA[i * NB + i];
        }
    }

    __syncthreads();

    // Save shared memory back into B
    if(tx < m)
    {
        for(int i = 0; i < maxColB; i++)
            B[i * ldb + tx] = sB[i * NB + tx];
    }
}

/*
 *  Uses a substitution method to solve a small problem XA = B. This version uses
 *  less shared memory for double complex types (currently this means not using
 *  shared memory for A).
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
static __global__ void rocblas_trsm_small_64_left_device(rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int               m,
                                                         int               n,
                                                         SCAL              alpha_dev_host,
                                                         ATYPE             Aa,
                                                         ptrdiff_t         offset_A,
                                                         int               lda,
                                                         rocblas_stride    stride_A,
                                                         BTYPE             Ba,
                                                         ptrdiff_t         offset_B,
                                                         int               ldb,
                                                         rocblas_stride    stride_B)
{
    const int batchid = blockIdx.y;
    auto      A       = load_ptr_batch(Aa, batchid, offset_A, stride_A);
    auto      B       = load_ptr_batch(Ba, batchid, offset_B, stride_B);
    auto      alpha   = load_scalar(alpha_dev_host);

    bool      LOWER = uplo == rocblas_fill_lower;
    bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
    const int tx    = threadIdx.x;
    const int bx    = blockIdx.x;

    // max A column to read from
    int maxColA = NB - 1 > m - 1 ? m - 1 : NB - 1;
    // NB columns, unless last block, then do leftover
    const int maxColB = (bx < gridDim.x - 1) ? NB : n - bx * NB;

    // offset B into correct block column
    B += bx * NB * ldb;

    // shared B
    __shared__ T sB[NB * NB];

    if(tx <= maxColA)
    {
        // Load B into sB and multiply by alpha
        for(int i = 0; i < maxColB; i++)
            sB[i * NB + tx] = alpha * B[i * ldb + tx];
    }
    __syncthreads();

    // Solve for B in shared memory
    if(LOWER && transA == rocblas_operation_none)
    {
        for(int i = 0; i < NB; i++)
        {
            // Subtract previously solved parts
            for(int j = 0; j < i; j++)
            {
                T valA = A[j * lda + i];
                sB[tx * NB + i] -= sB[tx * NB + j] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[tx * NB + i] /= A[i * lda + i];
        }
    }
    else if(!LOWER && transA == rocblas_operation_none)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
            {
                T valA = A[j * lda + i];
                sB[tx * NB + i] -= sB[tx * NB + j] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[tx * NB + i] /= A[i * lda + i];
        }
    }
    else if(LOWER)
    {
        for(int i = maxColA; i >= 0; i--)
        {
            for(int j = maxColA; j > i; j--)
            {
                T valA = (CONJ) ? conj(A[i * lda + j]) : A[i * lda + j];
                sB[tx * NB + i] -= sB[tx * NB + j] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[tx * NB + i] /= (CONJ) ? conj(A[i * lda + i]) : A[i * lda + i];
        }
    }
    else if(!LOWER)
    {
        for(int i = 0; i < NB; i++)
        {
            for(int j = 0; j < i; j++)
            {
                T valA = (CONJ) ? conj(A[i * lda + j]) : A[i * lda + j];
                sB[tx * NB + i] -= sB[tx * NB + j] * valA;
            }
            if(diag != rocblas_diagonal_unit)
                sB[tx * NB + i] /= (CONJ) ? conj(A[i * lda + i]) : A[i * lda + i];
        }
    }

    __syncthreads();

    // Save shared memory back into B
    if(tx < m)
    {
        for(int i = 0; i < maxColB; i++)
            B[i * ldb + tx] = sB[i * NB + tx];
    }
}

/* T = float, double, etc.
 * SCAL = T* or T
 * ATYPE = const T* or const T* const *
 * BTYPE = T* or T* const *
 *
 * Sets kernel parameters and launches the appropriate substitution kernel to solve
 * a small trsm problem.
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
void rocblas_trsm_small(rocblas_handle    handle,
                        rocblas_side      side,
                        rocblas_fill      uplo,
                        rocblas_operation transA,
                        rocblas_diagonal  diag,
                        rocblas_int       m,
                        rocblas_int       n,
                        SCAL              alpha,
                        ATYPE             dA,
                        ptrdiff_t         offset_A,
                        rocblas_int       lda,
                        rocblas_stride    stride_A,
                        BTYPE             dB,
                        ptrdiff_t         offset_B,
                        rocblas_int       ldb,
                        rocblas_stride    stride_B,
                        rocblas_int       batch_count)
{
    // threadIdx.x = NB >= m
    dim3 threads(NB, 1, 1);

    // blockIdx.x = divide B's columns into NB sized blocks
    // blockIdx.y = batch_count
    if(side == rocblas_side_left)
    {
        dim3 grid((n + NB - 1) / NB, batch_count);
        hipLaunchKernelGGL((rocblas_trsm_small_left_device<T, SCAL, ATYPE, BTYPE, NB>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           alpha,
                           dA,
                           offset_A,
                           lda,
                           stride_A,
                           dB,
                           offset_B,
                           ldb,
                           stride_B);
    }
    else
    {
        dim3 grid((m + NB - 1) / NB, batch_count);
        hipLaunchKernelGGL((rocblas_trsm_small_right_device<T, SCAL, ATYPE, BTYPE, NB>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           alpha,
                           dA,
                           offset_A,
                           lda,
                           stride_A,
                           dB,
                           offset_B,
                           ldb,
                           stride_B);
    }
}

/*
 * For sizes in (32, 64] we need a specialization for double complex types. This function is for all other cases where we can use the
 * regular substitution kernels.
 */
template <typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE,
          const int NB,
          std::enable_if_t<!std::is_same<T, rocblas_double_complex>{}, int> = 0>
void rocblas_trsm_small_64(rocblas_handle    handle,
                           rocblas_side      side,
                           rocblas_fill      uplo,
                           rocblas_operation transA,
                           rocblas_diagonal  diag,
                           rocblas_int       m,
                           rocblas_int       n,
                           SCAL              alpha,
                           ATYPE             dA,
                           ptrdiff_t         offset_A,
                           rocblas_int       lda,
                           rocblas_stride    stride_A,
                           BTYPE             dB,
                           ptrdiff_t         offset_B,
                           rocblas_int       ldb,
                           rocblas_stride    stride_B,
                           rocblas_int       batch_count)
{
    rocblas_trsm_small<T, SCAL, ATYPE, BTYPE, NB>(handle,
                                                  side,
                                                  uplo,
                                                  transA,
                                                  diag,
                                                  m,
                                                  n,
                                                  alpha,
                                                  dA,
                                                  offset_A,
                                                  lda,
                                                  stride_A,
                                                  dB,
                                                  offset_B,
                                                  ldb,
                                                  stride_B,
                                                  batch_count);
}

/*
 * For sizes in (32, 64] we need a specialization for double complex types. This function sets
 * kernel parameters and launches the appropriate kernel (with less shared memory)
 * for a double complex trsm problem.
 */
template <typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE,
          const int NB,
          std::enable_if_t<std::is_same<T, rocblas_double_complex>{}, int> = 0>
void rocblas_trsm_small_64(rocblas_handle    handle,
                           rocblas_side      side,
                           rocblas_fill      uplo,
                           rocblas_operation transA,
                           rocblas_diagonal  diag,
                           rocblas_int       m,
                           rocblas_int       n,
                           SCAL              alpha,
                           ATYPE             dA,
                           ptrdiff_t         offset_A,
                           rocblas_int       lda,
                           rocblas_stride    stride_A,
                           BTYPE             dB,
                           ptrdiff_t         offset_B,
                           rocblas_int       ldb,
                           rocblas_stride    stride_B,
                           rocblas_int       batch_count)
{
    // threadIdx.x = NB >= m
    dim3 threads(NB, 1, 1);

    // blockIdx.x = divide B's columns into NB sized blocks
    // blockIdx.y = batch_count
    if(side == rocblas_side_left)
    {
        dim3 grid((n + NB - 1) / NB, batch_count);
        hipLaunchKernelGGL((rocblas_trsm_small_64_left_device<T, SCAL, ATYPE, BTYPE, NB>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           alpha,
                           dA,
                           offset_A,
                           lda,
                           stride_A,
                           dB,
                           offset_B,
                           ldb,
                           stride_B);
    }
    else
    {
        dim3 grid((m + NB - 1) / NB, batch_count);
        hipLaunchKernelGGL((rocblas_trsm_small_64_right_device<T, SCAL, ATYPE, BTYPE, NB>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           alpha,
                           dA,
                           offset_A,
                           lda,
                           stride_A,
                           dB,
                           offset_B,
                           ldb,
                           stride_B);
    }
}

//////////////////////////////
//////////////////////////////
//////////////////////////////
template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_trsm_template(rocblas_handle    handle,
                                                             rocblas_side      side,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation transA,
                                                             rocblas_diagonal  diag,
                                                             rocblas_int       m,
                                                             rocblas_int       n,
                                                             const T*          alpha,
                                                             U                 A,
                                                             rocblas_int       offset_A,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stride_A,
                                                             V                 B,
                                                             rocblas_int       offset_B,
                                                             rocblas_int       ldb,
                                                             rocblas_stride    stride_B,
                                                             rocblas_int       batch_count,
                                                             bool              optimal_mem,
                                                             void*             x_temp,
                                                             void*             x_temparr,
                                                             void*             invA       = nullptr,
                                                             void*             invAarr    = nullptr,
                                                             U              supplied_invA = nullptr,
                                                             rocblas_int    supplied_invA_size = 0,
                                                             rocblas_int    offset_invA        = 0,
                                                             rocblas_stride stride_invA        = 0)
{
    bool SUBSTITUTION_ENABLED = true;

    if(batch_count == 0)
        return rocblas_status_success;

    // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    // Get alpha - Check if zero for quick return
    T alpha_h;
    if(saved_pointer_mode == rocblas_pointer_mode_host)
        alpha_h = *alpha;
    else
        RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));

    if(alpha_h == T(0.0))
    {
        set_block_unit<T>(handle, m, n, B, ldb, stride_B, batch_count, offset_B);
        return rocblas_status_success;
    }

    if(!is_complex<T> && transA == rocblas_operation_conjugate_transpose)
        transA = rocblas_operation_transpose;

    rocblas_int k = side == rocblas_side_left ? m : n;

    // bool is_small = k <= 64;
    bool is_small = (m <= 64 && n <= 64);
    if(SUBSTITUTION_ENABLED && is_small)
    {
        if(k <= 2)
            rocblas_trsm_small<T, T, U, V, 2>(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              m,
                                              n,
                                              alpha_h,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              B,
                                              offset_B,
                                              ldb,
                                              stride_B,
                                              batch_count);
        else if(k <= 4)
            rocblas_trsm_small<T, T, U, V, 4>(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              m,
                                              n,
                                              alpha_h,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              B,
                                              offset_B,
                                              ldb,
                                              stride_B,
                                              batch_count);
        else if(k <= 8)
            rocblas_trsm_small<T, T, U, V, 8>(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              m,
                                              n,
                                              alpha_h,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              B,
                                              offset_B,
                                              ldb,
                                              stride_B,
                                              batch_count);
        else if(k <= 16)
            rocblas_trsm_small<T, T, U, V, 16>(handle,
                                               side,
                                               uplo,
                                               transA,
                                               diag,
                                               m,
                                               n,
                                               alpha_h,
                                               A,
                                               offset_A,
                                               lda,
                                               stride_A,
                                               B,
                                               offset_B,
                                               ldb,
                                               stride_B,
                                               batch_count);
        else if(k <= 32)
            rocblas_trsm_small<T, T, U, V, 32>(handle,
                                               side,
                                               uplo,
                                               transA,
                                               diag,
                                               m,
                                               n,
                                               alpha_h,
                                               A,
                                               offset_A,
                                               lda,
                                               stride_A,
                                               B,
                                               offset_B,
                                               ldb,
                                               stride_B,
                                               batch_count);
        else if(k <= 64)
            rocblas_trsm_small_64<T, T, U, V, 64>(handle,
                                                  side,
                                                  uplo,
                                                  transA,
                                                  diag,
                                                  m,
                                                  n,
                                                  alpha_h,
                                                  A,
                                                  offset_A,
                                                  lda,
                                                  stride_A,
                                                  B,
                                                  offset_B,
                                                  ldb,
                                                  stride_B,
                                                  batch_count);
    }
    else
    {
        // Whether size is an exact multiple of blocksize
        const bool exact_blocks = (k % BLOCK) == 0;

        // perf_status indicates whether optimal performance is obtainable with available memory
        rocblas_status perf_status = rocblas_status_success;

        if(supplied_invA && supplied_invA_size / BLOCK < k)
        {
            perf_status   = rocblas_status_perf_degraded;
            supplied_invA = nullptr;
        }

        rocblas_status status = rocblas_status_success;

        if(supplied_invA)
        {
            invAarr = V(supplied_invA);
            invA    = V(supplied_invA);
        }
        else
        {
            // c_temp and x_temp can reuse the same device memory
            T* c_temp   = (T*)x_temp;
            stride_invA = BLOCK * k;
            if(BATCHED)
            {
                // for c_temp, we currently can use the same memory from each batch since
                // trtri_batched is naive (since gemm_batched is naive)
                setup_batched_array<BLOCK>(
                    handle->rocblas_stream, (T*)c_temp, 0, (T**)x_temparr, batch_count);
                setup_batched_array<BLOCK>(
                    handle->rocblas_stream, (T*)invA, stride_invA, (T**)invAarr, batch_count);
            }

            status = rocblas_trtri_trsm_template<BLOCK, BATCHED, T>(handle,
                                                                    V(BATCHED ? x_temparr : c_temp),
                                                                    uplo,
                                                                    diag,
                                                                    k,
                                                                    A,
                                                                    offset_A,
                                                                    lda,
                                                                    stride_A,
                                                                    V(BATCHED ? invAarr : invA),
                                                                    offset_invA,
                                                                    stride_invA,
                                                                    batch_count);

            if(status != rocblas_status_success)
                return status;
        }

        size_t B_chunk_size = optimal_mem ? size_t(m) + size_t(n) - size_t(k) : 1;
        size_t x_temp_els   = exact_blocks ? BLOCK * B_chunk_size : size_t(m) * n;
        if(BATCHED)
        {
            setup_batched_array<BLOCK>(
                handle->rocblas_stream, (T*)x_temp, x_temp_els, (T**)x_temparr, batch_count);
        }

        if(exact_blocks)
        {
            status = special_trsm_template<BLOCK, BATCHED>(handle,
                                                           side,
                                                           uplo,
                                                           transA,
                                                           diag,
                                                           m,
                                                           n,
                                                           &alpha_h,
                                                           U(A),
                                                           offset_A,
                                                           lda,
                                                           stride_A,
                                                           V(B),
                                                           offset_B,
                                                           ldb,
                                                           stride_B,
                                                           batch_count,
                                                           U(BATCHED ? invAarr : invA),
                                                           offset_invA,
                                                           stride_invA,
                                                           B_chunk_size,
                                                           V(BATCHED ? x_temparr : x_temp),
                                                           x_temp_els);
        }
        else
        {
            if(side == rocblas_side_left)
                status = rocblas_trsm_left<BLOCK, BATCHED, T>(handle,
                                                              uplo,
                                                              transA,
                                                              m,
                                                              n,
                                                              &alpha_h,
                                                              U(A),
                                                              offset_A,
                                                              lda,
                                                              stride_A,
                                                              V(B),
                                                              offset_B,
                                                              ldb,
                                                              stride_B,
                                                              batch_count,
                                                              U(BATCHED ? invAarr : invA),
                                                              offset_invA,
                                                              stride_invA,
                                                              V(BATCHED ? x_temparr : x_temp),
                                                              x_temp_els);
            else
                status = rocblas_trsm_right<BLOCK, BATCHED, T>(handle,
                                                               uplo,
                                                               transA,
                                                               m,
                                                               n,
                                                               &alpha_h,
                                                               U(A),
                                                               offset_A,
                                                               lda,
                                                               stride_A,
                                                               V(B),
                                                               offset_B,
                                                               ldb,
                                                               stride_B,
                                                               batch_count,
                                                               U(BATCHED ? invAarr : invA),
                                                               offset_invA,
                                                               stride_invA,
                                                               V(BATCHED ? x_temparr : x_temp),
                                                               x_temp_els);

            if(status != rocblas_status_success)
                return status;

            copy_block_unit<T>(handle,
                               m,
                               n,
                               U(BATCHED ? x_temparr : x_temp),
                               m,
                               x_temp_els,
                               V(B),
                               ldb,
                               stride_B,
                               batch_count,
                               0,
                               offset_B);
        }

        // If status is successful, return perf_status; else return status
        return status == rocblas_status_success ? perf_status : status;
    }

    return rocblas_status_success;
}

#endif
