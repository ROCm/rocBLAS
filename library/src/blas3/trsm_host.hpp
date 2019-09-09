/* ************************************************************************
* Copyright 2016-2019 Advanced Micro Devices, Inc.
* ************************************************************************ */
#include "handle.h"
#include "rocblas.h"
#include "trsm_device.hpp"
#include "trtri_trsm.hpp"
#include "utility.h"

#define A(ii, jj) (A + (ii) + (jj)*lda)
#define B(ii, jj) (B + (ii) + (jj)*ldb)
#define X(ii, jj) (X + (ii) + (jj)*m)
#define invA(ii) (invA + (ii)*BLOCK)

///////////////////// //////
///// helper templates /////
////////////////////////////
namespace
{
    using std::max;
    using std::min;

    template <typename T>
    constexpr T negative_one = -1;
    template <typename T>
    constexpr T zero = 0;
    template <typename T>
    constexpr T one = 1;

    /* ===============copy helper============================================= */
    template <typename T>
    void copy_block_unit(rocblas_handle handle,
                         rocblas_int    m,
                         rocblas_int    n,
                         const T*       src,
                         rocblas_int    src_ld,
                         rocblas_int    src_stride,
                         T*             dst,
                         rocblas_int    dst_ld,
                         rocblas_int    dst_stride,
                         rocblas_int    batch_count)
    {
        rocblas_int blocksX = (m - 1) / 128 + 1; // parameters for device kernel
        rocblas_int blocksY = (n - 1) / 8 + 1;
        dim3        grid(blocksX, blocksY, batch_count);
        dim3        threads(128, 8);

        hipLaunchKernelGGL(copy_matrix_strided_batched_trsm,
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
                           dst_stride);
    }

    template <typename T>
    void copy_block_unit(rocblas_handle handle,
                         rocblas_int    m,
                         rocblas_int    n,
                         const T*       src[],
                         rocblas_int    src_ld,
                         T*             dst[],
                         rocblas_int    dst_ld,
                         rocblas_int    batch_count,
                         rocblas_int    offset_src = 0,
                         rocblas_int    offset_dst = 0)
    {
        rocblas_int blocksX = (m - 1) / 128 + 1; // paramteres for device kernel
        rocblas_int blocksY = (n - 1) / 8 + 1;
        dim3        grid(blocksX, blocksY, batch_count);
        dim3        threads(128, 8);

        hipLaunchKernelGGL(copy_matrix_batched_trsm,
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           m,
                           n,
                           sizeof(T),
                           src,
                           src_ld,
                           dst,
                           dst_ld,
                           offset_src,
                           offset_dst);
    }

    /* ===============left==================================================== */

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_strided_batched_left(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transA,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     const T*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_int       stride_A,
                                                     T*                B,
                                                     rocblas_int       ldb,
                                                     rocblas_int       stride_B,
                                                     rocblas_int       batch_count,
                                                     const T*          invA,
                                                     rocblas_int       stride_invA,
                                                     T*                X)
    {
        rocblas_int i, jb;
        rocblas_int stride_X = m * n;
        // transB is always non-transpose
        static constexpr rocblas_operation transB = rocblas_operation_none;

        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // left, lower no-transpose
                jb = min(BLOCK, m);
                rocblas_gemm_strided_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      jb,
                                                      n,
                                                      jb,
                                                      alpha,
                                                      invA,
                                                      BLOCK,
                                                      stride_invA,
                                                      B,
                                                      ldb,
                                                      stride_B,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      stride_X,
                                                      batch_count);

                if(BLOCK < m)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transA,
                                                          transB,
                                                          m - BLOCK,
                                                          n,
                                                          BLOCK,
                                                          &negative_one<T>,
                                                          A(BLOCK, 0),
                                                          lda,
                                                          stride_A,
                                                          X,
                                                          m,
                                                          stride_X,
                                                          alpha,
                                                          B(BLOCK, 0),
                                                          ldb,
                                                          stride_B,
                                                          batch_count);
                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);

                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              jb,
                                                              n,
                                                              jb,
                                                              &one<T>,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              B(i, 0),
                                                              ldb,
                                                              stride_B,
                                                              &zero<T>,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i + BLOCK
                           >= m) // this condition is not necessary at all and can be changed
                            // as if (i+BLOCK<m)
                            break;

                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              m - i - BLOCK,
                                                              n,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              A(i + BLOCK, i),
                                                              lda,
                                                              stride_A,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              &one<T>,
                                                              B(i + BLOCK, 0),
                                                              ldb,
                                                              stride_B,
                                                              batch_count);
                    }
                }

#if 0
            for( i=0; i < m; i += BLOCK ) {
                jb = min(m-i, BLOCK);
                T *tmp = (i == 0) ? alpha : one;
                rocblas_gemm_strided_batched_template(handle, transA, transB, jb, n, jb, tmp, invA(i), BLOCK, stride_invA, B(i,0), ldb, stride_B, &zero<T>, X(i,0), ldb, stride_X, batch_count); // strides?
                if(i + BLOCK < m){
                    rocblas_gemm_strided_batched_template(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negative_one<T>, A(i+BLOCK,i), lda, stride_A, X(i,0), ldb, stride_X, tmp, B(i+BLOCK,0), ldb, stride_B, batch_count); // strides?
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
                rocblas_gemm_strided_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      jb,
                                                      n,
                                                      jb,
                                                      alpha,
                                                      invA(i),
                                                      BLOCK,
                                                      stride_invA,
                                                      B(i, 0),
                                                      ldb,
                                                      stride_B,
                                                      &zero<T>,
                                                      X(i, 0),
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(i - BLOCK >= 0)
                {

                    rocblas_gemm_strided_batched_template(handle,
                                                          transA,
                                                          transB,
                                                          i,
                                                          n,
                                                          jb,
                                                          &negative_one<T>,
                                                          A(0, i),
                                                          lda,
                                                          stride_A,
                                                          X(i, 0),
                                                          m,
                                                          stride_X,
                                                          alpha,
                                                          B,
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        //{32, 35, 32, 32, 35, 35}
                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              BLOCK,
                                                              n,
                                                              BLOCK,
                                                              &one<T>,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              B(i, 0),
                                                              ldb,
                                                              stride_B,
                                                              &zero<T>,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              i,
                                                              n,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              A(0, i),
                                                              lda,
                                                              stride_A,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              &one<T>,
                                                              B,
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
                rocblas_gemm_strided_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      jb,
                                                      n,
                                                      jb,
                                                      alpha,
                                                      invA(i),
                                                      BLOCK,
                                                      stride_invA,
                                                      B(i, 0),
                                                      ldb,
                                                      stride_B,
                                                      &zero<T>,
                                                      X(i, 0),
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transA,
                                                          transB,
                                                          i,
                                                          n,
                                                          jb,
                                                          &negative_one<T>,
                                                          A(i, 0),
                                                          lda,
                                                          stride_A,
                                                          X(i, 0),
                                                          m,
                                                          stride_X,
                                                          alpha,
                                                          B,
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              BLOCK,
                                                              n,
                                                              BLOCK,
                                                              &one<T>,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              B(i, 0),
                                                              ldb,
                                                              stride_B,
                                                              &zero<T>,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              i,
                                                              n,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              A(i, 0),
                                                              lda,
                                                              stride_A,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              &one<T>,
                                                              B,
                                                              ldb,
                                                              stride_B,
                                                              batch_count);
                    }
                }
            }
            else
            {
                // left, upper transpose
                jb = min(BLOCK, m);
                rocblas_gemm_strided_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      jb,
                                                      n,
                                                      jb,
                                                      alpha,
                                                      invA,
                                                      BLOCK,
                                                      stride_invA,
                                                      B,
                                                      ldb,
                                                      stride_B,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(BLOCK < m)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transA,
                                                          transB,
                                                          m - BLOCK,
                                                          n,
                                                          BLOCK,
                                                          &negative_one<T>,
                                                          A(0, BLOCK),
                                                          lda,
                                                          stride_A,
                                                          X,
                                                          m,
                                                          stride_X,
                                                          alpha,
                                                          B(BLOCK, 0),
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              jb,
                                                              n,
                                                              jb,
                                                              &one<T>,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              B(i, 0),
                                                              ldb,
                                                              stride_B,
                                                              &zero<T>,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i + BLOCK >= m)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transA,
                                                              transB,
                                                              m - i - BLOCK,
                                                              n,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              A(i, i + BLOCK),
                                                              lda,
                                                              stride_A,
                                                              X(i, 0),
                                                              m,
                                                              stride_X,
                                                              &one<T>,
                                                              B(i + BLOCK, 0),
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

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_strided_batched_right(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_int       stride_A,
                                                      T*                B,
                                                      rocblas_int       ldb,
                                                      rocblas_int       stride_B,
                                                      rocblas_int       batch_count,
                                                      const T*          invA,
                                                      rocblas_int       stride_invA,
                                                      T*                X)
    {
        rocblas_int i, jb;
        rocblas_int stride_X = m * n;

        // transB is always non-transpose
        static constexpr rocblas_operation transB = rocblas_operation_none;

        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // right, lower no-transpose
                jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
                i  = n - jb;
                rocblas_gemm_strided_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      jb,
                                                      jb,
                                                      alpha,
                                                      B(0, i),
                                                      ldb,
                                                      stride_B,
                                                      invA(i),
                                                      BLOCK,
                                                      stride_invA,
                                                      &zero<T>,
                                                      X(0, i),
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transB,
                                                          transA,
                                                          m,
                                                          i,
                                                          jb,
                                                          &negative_one<T>,
                                                          X(0, i),
                                                          m,
                                                          stride_X,
                                                          A(i, 0),
                                                          lda,
                                                          stride_A,
                                                          alpha,
                                                          B,
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              BLOCK,
                                                              BLOCK,
                                                              &one<T>,
                                                              B(0, i),
                                                              ldb,
                                                              stride_B,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              &zero<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              i,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              A(i, 0),
                                                              lda,
                                                              stride_A,
                                                              &one<T>,
                                                              B,
                                                              ldb,
                                                              stride_B,
                                                              batch_count);
                    }
                }
            }
            else
            {
                // right, upper no-transpose
                jb = min(BLOCK, n);
                rocblas_gemm_strided_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      jb,
                                                      jb,
                                                      alpha,
                                                      B,
                                                      ldb,
                                                      stride_B,
                                                      invA,
                                                      BLOCK,
                                                      stride_invA,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(BLOCK < n)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transB,
                                                          transA,
                                                          m,
                                                          n - BLOCK,
                                                          BLOCK,
                                                          &negative_one<T>,
                                                          X,
                                                          m,
                                                          stride_X,
                                                          A(0, BLOCK),
                                                          lda,
                                                          stride_A,
                                                          alpha,
                                                          B(0, BLOCK),
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              jb,
                                                              jb,
                                                              &one<T>,
                                                              B(0, i),
                                                              ldb,
                                                              stride_B,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              &zero<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i + BLOCK >= n)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              n - i - BLOCK,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              A(i, i + BLOCK),
                                                              lda,
                                                              stride_A,
                                                              &one<T>,
                                                              B(0, i + BLOCK),
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
                jb = min(BLOCK, n);
                rocblas_gemm_strided_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      jb,
                                                      jb,
                                                      alpha,
                                                      B,
                                                      ldb,
                                                      stride_B,
                                                      invA,
                                                      BLOCK,
                                                      stride_invA,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(BLOCK < n)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transB,
                                                          transA,
                                                          m,
                                                          n - BLOCK,
                                                          BLOCK,
                                                          &negative_one<T>,
                                                          X,
                                                          m,
                                                          stride_X,
                                                          A(BLOCK, 0),
                                                          lda,
                                                          stride_A,
                                                          alpha,
                                                          B(0, BLOCK),
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              jb,
                                                              jb,
                                                              &one<T>,
                                                              B(0, i),
                                                              ldb,
                                                              stride_B,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              &zero<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i + BLOCK >= n)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              n - i - BLOCK,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              A(BLOCK + i, i),
                                                              lda,
                                                              stride_A,
                                                              &one<T>,
                                                              B(0, i + BLOCK),
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
                rocblas_gemm_strided_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      jb,
                                                      jb,
                                                      alpha,
                                                      B(0, i),
                                                      ldb,
                                                      stride_B,
                                                      invA(i),
                                                      BLOCK,
                                                      stride_invA,
                                                      &zero<T>,
                                                      X(0, i),
                                                      m,
                                                      stride_X,
                                                      batch_count);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_strided_batched_template(handle,
                                                          transB,
                                                          transA,
                                                          m,
                                                          i,
                                                          jb,
                                                          &negative_one<T>,
                                                          X(0, i),
                                                          m,
                                                          stride_X,
                                                          A(0, i),
                                                          lda,
                                                          stride_A,
                                                          alpha,
                                                          B,
                                                          ldb,
                                                          stride_B,
                                                          batch_count);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              BLOCK,
                                                              BLOCK,
                                                              &one<T>,
                                                              B(0, i),
                                                              ldb,
                                                              stride_B,
                                                              invA(i),
                                                              BLOCK,
                                                              stride_invA,
                                                              &zero<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              batch_count);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_strided_batched_template(handle,
                                                              transB,
                                                              transA,
                                                              m,
                                                              i,
                                                              BLOCK,
                                                              &negative_one<T>,
                                                              X(0, i),
                                                              m,
                                                              stride_X,
                                                              A(0, i),
                                                              lda,
                                                              stride_A,
                                                              &one<T>,
                                                              B,
                                                              ldb,
                                                              stride_B,
                                                              batch_count);
                    }
                }
            }
        } // tranpsose

        return rocblas_status_success;
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status special_trsm_strided_batched_template(rocblas_handle    handle,
                                                         rocblas_side      side,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         rocblas_int       m,
                                                         rocblas_int       n,
                                                         const T*          alpha,
                                                         const T*          A,
                                                         rocblas_int       lda,
                                                         rocblas_int       stride_A,
                                                         T*                B,
                                                         rocblas_int       ldb,
                                                         rocblas_int       stride_B,
                                                         rocblas_int       batch_count,
                                                         const T*          invA,
                                                         rocblas_int       stride_invA,
                                                         size_t            B_chunk_size,
                                                         T*                x_temp)
    {
        bool        parity     = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_upper);
        size_t      k          = side == rocblas_side_left ? m : n;
        size_t      R          = k / BLOCK;
        size_t      bsize      = side == rocblas_side_left ? n : m;
        size_t      W          = 1 + (bsize - 1) / B_chunk_size;
        bool        arch_lt906 = handle->device_arch_id() < 906;
        rocblas_int stride_X   = m * n;

        for(size_t w = 0; w < W; w++)
        {
            size_t width = min(bsize - w * B_chunk_size, B_chunk_size);

            if(side == rocblas_side_left)
            {
                T* Bw = B + w * B_chunk_size * ldb;

                for(size_t r = 0; r < R; r++)
                {
                    size_t q = R - 1 - r;
                    size_t j = parity ? r : q;

                    // copy a BLOCK*n piece we are solving at a time
                    if(!r || arch_lt906)
                        copy_block_unit(handle,
                                        BLOCK,
                                        width,
                                        Bw + j * BLOCK,
                                        ldb,
                                        stride_B,
                                        x_temp,
                                        BLOCK,
                                        stride_X,
                                        batch_count);

                    if(r)
                    {
                        const T* A_current;
                        T*       B_current = parity ? Bw : Bw + (q + 1) * BLOCK;

                        if(transA == rocblas_operation_none)
                            A_current = parity ? A + r * BLOCK : A + BLOCK * (q * lda + q + lda);
                        else
                            A_current
                                = parity ? A + r * BLOCK * lda : A + BLOCK * (q * lda + q + 1);

                        if(arch_lt906)
                        {
                            rocblas_gemm_strided_batched_template(handle,
                                                                  transA,
                                                                  rocblas_operation_none,
                                                                  BLOCK,
                                                                  width,
                                                                  r * BLOCK,
                                                                  &negative_one<T>,
                                                                  A_current,
                                                                  lda,
                                                                  stride_A,
                                                                  B_current,
                                                                  ldb,
                                                                  stride_B,
                                                                  alpha,
                                                                  x_temp,
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

                            rocblas_gemm_strided_batched_ex(handle,
                                                            transA,
                                                            rocblas_operation_none,
                                                            BLOCK,
                                                            width,
                                                            r * BLOCK,
                                                            &negative_one<T>,
                                                            A_current,
                                                            compute_type,
                                                            lda,
                                                            stride_A,
                                                            B_current,
                                                            compute_type,
                                                            ldb,
                                                            stride_B,
                                                            alpha,
                                                            Bw + j * BLOCK,
                                                            compute_type,
                                                            ldb,
                                                            stride_B,
                                                            x_temp,
                                                            compute_type,
                                                            BLOCK,
                                                            stride_X,
                                                            batch_count,
                                                            compute_type,
                                                            algo,
                                                            solution_index,
                                                            flags);
                        }
                    }

                    rocblas_gemm_strided_batched_template(handle,
                                                          transA,
                                                          rocblas_operation_none,
                                                          BLOCK,
                                                          width,
                                                          BLOCK,
                                                          r ? &one<T> : alpha,
                                                          invA + j * BLOCK * BLOCK,
                                                          BLOCK,
                                                          stride_invA, // ??
                                                          x_temp,
                                                          BLOCK,
                                                          stride_X, // ??
                                                          &zero<T>,
                                                          Bw + j * BLOCK,
                                                          ldb,
                                                          stride_B,
                                                          batch_count);
                }
            }
            else
            {
                T* Bw = B + w * B_chunk_size;
                for(size_t r = 0; r < R; r++)
                {
                    size_t q = R - 1 - r;
                    size_t j = parity ? q : r;

                    // copy a m*BLOCK piece we are solving at a time
                    if(!r || arch_lt906)
                        copy_block_unit(handle,
                                        width,
                                        BLOCK,
                                        Bw + j * BLOCK * ldb,
                                        ldb,
                                        stride_B,
                                        x_temp,
                                        width,
                                        stride_X,
                                        batch_count);

                    if(r)
                    {
                        const T* A_current;
                        T*       B_current = parity ? Bw + (q + 1) * BLOCK * ldb : Bw;
                        if(transA == rocblas_operation_none)
                            A_current
                                = parity ? A + BLOCK * (q * lda + q + 1) : A + r * BLOCK * lda;
                        else
                            A_current = parity ? A + BLOCK * (q * lda + q + lda) : A + r * BLOCK;

                        if(arch_lt906)
                        {
                            rocblas_gemm_strided_batched_template(handle,
                                                                  rocblas_operation_none,
                                                                  transA,
                                                                  width,
                                                                  BLOCK,
                                                                  r * BLOCK,
                                                                  &negative_one<T>,
                                                                  B_current,
                                                                  ldb,
                                                                  stride_B,
                                                                  A_current,
                                                                  lda,
                                                                  stride_A,
                                                                  alpha,
                                                                  x_temp,
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

                            rocblas_gemm_strided_batched_ex(handle,
                                                            rocblas_operation_none,
                                                            transA,
                                                            width,
                                                            BLOCK,
                                                            r * BLOCK,
                                                            &negative_one<T>,
                                                            B_current,
                                                            compute_type,
                                                            ldb,
                                                            stride_B,
                                                            A_current,
                                                            compute_type,
                                                            lda,
                                                            stride_A,
                                                            alpha,
                                                            Bw + j * BLOCK * ldb,
                                                            compute_type,
                                                            ldb,
                                                            stride_B,
                                                            x_temp,
                                                            compute_type,
                                                            width,
                                                            stride_X,
                                                            batch_count,
                                                            compute_type,
                                                            algo,
                                                            solution_index,
                                                            flags);
                        }
                    }

                    rocblas_gemm_strided_batched_template(handle,
                                                          rocblas_operation_none,
                                                          transA,
                                                          width,
                                                          BLOCK,
                                                          BLOCK,
                                                          r ? &one<T> : alpha,
                                                          x_temp,
                                                          width,
                                                          stride_B,
                                                          invA + j * BLOCK * BLOCK,
                                                          BLOCK,
                                                          stride_invA,
                                                          &zero<T>,
                                                          Bw + j * BLOCK * ldb,
                                                          ldb,
                                                          stride_B,
                                                          batch_count);
                }
            }
        }

        return rocblas_status_success;
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_batched_left(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             T*                B[],
                                             rocblas_int       ldb,
                                             rocblas_int       batch_count,
                                             const T*          invA[],
                                             T*                X[])
    {
        rocblas_int i, jb;
        rocblas_int stride_X = m * n;
        // transB is always non-transpose
        static constexpr rocblas_operation transB = rocblas_operation_none;

        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // left, lower no-transpose
                jb = min(BLOCK, m);
                rocblas_gemm_batched_template(handle,
                                              transA,
                                              transB,
                                              jb,
                                              n,
                                              jb,
                                              alpha,
                                              invA,
                                              BLOCK,
                                              B,
                                              ldb,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count);

                if(BLOCK < m)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transA,
                                                  transB,
                                                  m - BLOCK,
                                                  n,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A,
                                                  lda,
                                                  X,
                                                  m,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  BLOCK,
                                                  0,
                                                  BLOCK);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      jb,
                                                      n,
                                                      jb,
                                                      &one<T>,
                                                      invA,
                                                      BLOCK,
                                                      B,
                                                      ldb,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * BLOCK,
                                                      i,
                                                      i);

                        if(i + BLOCK
                           >= m) // this condition is not necessary at all and can be changed
                            // as if (i+BLOCK<m)
                            break;

                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      m - i - BLOCK,
                                                      n,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      A,
                                                      lda,
                                                      X,
                                                      m,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i + BLOCK + i * lda,
                                                      i,
                                                      i + BLOCK);
                    }
                }
            }
            else
            {
                // left, upper no-transpose
                jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
                i  = m - jb;

                // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
                rocblas_gemm_batched_template(handle,
                                              transA,
                                              transB,
                                              jb,
                                              n,
                                              jb,
                                              alpha,
                                              invA,
                                              BLOCK,
                                              B,
                                              ldb,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count,
                                              i * BLOCK,
                                              i,
                                              i);
                if(i - BLOCK >= 0)
                {

                    rocblas_gemm_batched_template(handle,
                                                  transA,
                                                  transB,
                                                  i,
                                                  n,
                                                  jb,
                                                  &negative_one<T>,
                                                  A,
                                                  lda,
                                                  X,
                                                  m,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  i * lda,
                                                  i,
                                                  0);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        //{32, 35, 32, 32, 35, 35}
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      BLOCK,
                                                      n,
                                                      BLOCK,
                                                      &one<T>,
                                                      invA,
                                                      BLOCK,
                                                      B,
                                                      ldb,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * BLOCK,
                                                      i,
                                                      i);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      i,
                                                      n,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      A,
                                                      lda,
                                                      X,
                                                      m,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i * lda,
                                                      i,
                                                      0);
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
                rocblas_gemm_batched_template(handle,
                                              transA,
                                              transB,
                                              jb,
                                              n,
                                              jb,
                                              alpha,
                                              invA,
                                              BLOCK,
                                              B,
                                              ldb,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count,
                                              i * BLOCK,
                                              i,
                                              i);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transA,
                                                  transB,
                                                  i,
                                                  n,
                                                  jb,
                                                  &negative_one<T>,
                                                  A,
                                                  lda,
                                                  X,
                                                  m,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  i,
                                                  i,
                                                  0);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      BLOCK,
                                                      n,
                                                      BLOCK,
                                                      &one<T>,
                                                      invA,
                                                      BLOCK,
                                                      B,
                                                      ldb,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * BLOCK,
                                                      i,
                                                      i);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      i,
                                                      n,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      A,
                                                      lda,
                                                      X,
                                                      m,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i,
                                                      i,
                                                      0);
                    }
                }
            }
            else
            {
                // left, upper transpose
                jb = min(BLOCK, m);
                rocblas_gemm_batched_template(handle,
                                              transA,
                                              transB,
                                              jb,
                                              n,
                                              jb,
                                              alpha,
                                              invA,
                                              BLOCK,
                                              B,
                                              ldb,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count);
                if(BLOCK < m)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transA,
                                                  transB,
                                                  m - BLOCK,
                                                  n,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A,
                                                  lda,
                                                  X,
                                                  m,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  BLOCK * lda,
                                                  0,
                                                  BLOCK);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      jb,
                                                      n,
                                                      jb,
                                                      &one<T>,
                                                      invA,
                                                      BLOCK,
                                                      B,
                                                      ldb,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * BLOCK,
                                                      i,
                                                      i);
                        if(i + BLOCK >= m)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transA,
                                                      transB,
                                                      m - i - BLOCK,
                                                      n,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      A,
                                                      lda,
                                                      X,
                                                      m,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i + (i + BLOCK) * lda,
                                                      i,
                                                      i + BLOCK);
                    }
                }
            }
        } // transpose

        return rocblas_status_success;
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_batched_right(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T* const    A[],
                                                      rocblas_int       lda,
                                                      T*                B[],
                                                      rocblas_int       ldb,
                                                      rocblas_int       batch_count,
                                                      const T*          invA[],
                                                      T*                X[])
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
                rocblas_gemm_batched_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              jb,
                                              jb,
                                              alpha,
                                              B,
                                              ldb,
                                              invA,
                                              BLOCK,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count,
                                              i * ldb,
                                              i * BLOCK,
                                              i * m);

                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  i,
                                                  jb,
                                                  &negative_one<T>,
                                                  X,
                                                  m,
                                                  A,
                                                  lda,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  i * m,
                                                  i,
                                                  0);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      BLOCK,
                                                      BLOCK,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      invA,
                                                      BLOCK,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * ldb,
                                                      i * BLOCK,
                                                      i * m);

                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      i,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      X,
                                                      m,
                                                      A,
                                                      lda,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i * m,
                                                      i,
                                                      0);
                    }
                }
            }
            else
            {
                // right, upper no-transpose
                jb = min(BLOCK, n);
                rocblas_gemm_batched_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              jb,
                                              jb,
                                              alpha,
                                              B,
                                              ldb,
                                              invA,
                                              BLOCK,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count);
                if(BLOCK < n)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  n - BLOCK,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  X,
                                                  m,
                                                  A,
                                                  lda,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  0,
                                                  BLOCK * lda,
                                                  BLOCK * ldb);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      jb,
                                                      jb,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      invA,
                                                      BLOCK,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * ldb,
                                                      i * BLOCK,
                                                      i * m);
                        if(i + BLOCK >= n)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      n - i - BLOCK,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      X,
                                                      m,
                                                      A,
                                                      lda,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i * m,
                                                      i + (i + BLOCK) * lda,
                                                      (i + BLOCK) * ldb);
                    }
                }
            }
        }
        else
        { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
            if(uplo == rocblas_fill_lower)
            {
                // right, lower transpose
                jb = min(BLOCK, n);
                rocblas_gemm_batched_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              jb,
                                              jb,
                                              alpha,
                                              B,
                                              ldb,
                                              invA,
                                              BLOCK,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count);
                if(BLOCK < n)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  n - BLOCK,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  X,
                                                  m,
                                                  A,
                                                  lda,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  0,
                                                  BLOCK,
                                                  BLOCK * ldb);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      jb,
                                                      jb,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      invA,
                                                      BLOCK,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * ldb,
                                                      i * BLOCK,
                                                      i * m);
                        if(i + BLOCK >= n)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      n - i - BLOCK,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      X,
                                                      m,
                                                      A,
                                                      lda,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i * m,
                                                      BLOCK + i + i * lda,
                                                      (i + BLOCK) * ldb);
                    }
                }
            }
            else
            {
                // right, upper transpose
                jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
                i  = n - jb;
                rocblas_gemm_batched_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              jb,
                                              jb,
                                              alpha,
                                              B,
                                              ldb,
                                              invA,
                                              BLOCK,
                                              &zero<T>,
                                              X,
                                              m,
                                              batch_count,
                                              i * ldb,
                                              i * BLOCK,
                                              i * m);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_batched_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  i,
                                                  jb,
                                                  &negative_one<T>,
                                                  X,
                                                  m,
                                                  A,
                                                  lda,
                                                  alpha,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  i * m,
                                                  i * lda,
                                                  0);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      BLOCK,
                                                      BLOCK,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      invA,
                                                      BLOCK,
                                                      &zero<T>,
                                                      X,
                                                      m,
                                                      batch_count,
                                                      i * ldb,
                                                      i * BLOCK,
                                                      i * m);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_batched_template(handle,
                                                      transB,
                                                      transA,
                                                      m,
                                                      i,
                                                      BLOCK,
                                                      &negative_one<T>,
                                                      X,
                                                      m,
                                                      A,
                                                      lda,
                                                      &one<T>,
                                                      B,
                                                      ldb,
                                                      batch_count,
                                                      i * m,
                                                      i * lda,
                                                      0);
                    }
                }
            }
        } // tranpsose

        return rocblas_status_success;
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status special_trsm_batched_template(rocblas_handle    handle,
                                                 rocblas_side      side,   
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,   
                                                 rocblas_diagonal  diag,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 const T*          alpha,
                                                 const T* const    A[],
                                                 rocblas_int       lda,
                                                 T*                B[],
                                                 rocblas_int       ldb,
                                                 rocblas_int       batch_count,
                                                 const T*          invA[],
                                                 size_t            B_chunk_size,
                                                 T*                x_temp[])
    {
        bool        parity     = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_upper);
        size_t      k          = side == rocblas_side_left ? m : n;
        size_t      R          = k / BLOCK;
        size_t      bsize      = side == rocblas_side_left ? n : m;
        size_t      W          = 1 + (bsize - 1) / B_chunk_size;
        bool        arch_lt906 = handle->device_arch_id() < 906;

        for(size_t w = 0; w < W; w++)
        {
            size_t width = min(bsize - w * B_chunk_size, B_chunk_size);
            if(side == rocblas_side_left)
            {
                for(size_t r = 0; r < R; r++)
                {
                    size_t q = R - 1 - r;
                    size_t j = parity ? r : q;

                    // copy a BLOCK*n piece we are solving at a time
                    if(!r || arch_lt906)
                    {
                        copy_block_unit(handle,
                                        BLOCK,
                                        width,
                                        (const T**)B,
                                        ldb,
                                        x_temp,
                                        BLOCK,
                                        batch_count,
                                        j * BLOCK + w * B_chunk_size * ldb,
                                        0);
                    }

                    if(r)
                    {
                        rocblas_int offsetA = 0;
                        rocblas_int offsetB = parity ? w * B_chunk_size * ldb : w * B_chunk_size * ldb + (q + 1) * BLOCK;
                        if(transA == rocblas_operation_none)
                            offsetA = parity ? r * BLOCK : BLOCK * (q * lda + q + lda);
                        else
                            offsetA = parity ? r * BLOCK * lda : BLOCK * (q * lda + q + 1);

                        if(arch_lt906)
                        {
                            rocblas_gemm_batched_template(handle,
                                                          transA,
                                                          rocblas_operation_none,
                                                          BLOCK,
                                                          width,
                                                          r * BLOCK,
                                                          &negative_one<T>,
                                                          A,
                                                          lda,
                                                          B,
                                                          ldb,
                                                          alpha,
                                                          x_temp,
                                                          BLOCK,
                                                          batch_count,
                                                          offsetA,
                                                          offsetB,
                                                          0);
                        }
                        else
                        {
                            rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                            rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                            int32_t           solution_index = 0;
                            uint32_t          flags          = 0;

                            rocblas_gemm_batched_ex_offset(handle,
                                                    transA,
                                                    rocblas_operation_none,
                                                    BLOCK,
                                                    width,
                                                    r * BLOCK,
                                                    &negative_one<T>,
                                                    A,
                                                    compute_type,
                                                    lda,
                                                    B,
                                                    compute_type,
                                                    ldb,
                                                    alpha,
                                                    B,
                                                    compute_type,
                                                    ldb,
                                                    x_temp,
                                                    compute_type,
                                                    BLOCK,
                                                    batch_count,
                                                    compute_type,
                                                    algo,
                                                    offsetA,
                                                    offsetB,
                                                    j * BLOCK + w * B_chunk_size * ldb,
                                                    0,
                                                    solution_index,
                                                    flags);
                        }
                    }

                    rocblas_gemm_batched_template(handle,
                                                  transA,
                                                  rocblas_operation_none,
                                                  BLOCK,
                                                  width,
                                                  BLOCK,
                                                  r ? &one<T> : alpha,
                                                  (const T**)invA,
                                                  BLOCK,
                                                  x_temp,
                                                  BLOCK,
                                                  &zero<T>,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  j * BLOCK * BLOCK,
                                                  0,
                                                  w * B_chunk_size * ldb + j * BLOCK);
                }
            }
            else
            {
                for(size_t r = 0; r < R; r++)
                {
                    size_t q = R - 1 - r;
                    size_t j = parity ? q : r;

                    if(!r || arch_lt906)
                        copy_block_unit(handle,
                                        width,
                                        BLOCK,
                                        (const T**)B,
                                        ldb,
                                        x_temp,
                                        width,
                                        batch_count,
                                        j * BLOCK * ldb + w * B_chunk_size,
                                        0);

                    if(r)
                    {
                        rocblas_int offsetA = 0;
                        rocblas_int offsetB = parity ? w * B_chunk_size + (q + 1) * BLOCK * ldb : w * B_chunk_size;
                        if(transA == rocblas_operation_none)
                            offsetA = parity ? BLOCK * (q * lda + q + 1) : r * BLOCK * lda;
                        else
                            offsetA = parity ? BLOCK * (q * lda + q + lda) : r * BLOCK;

                        if(arch_lt906)
                        {
                            rocblas_gemm_batched_template(handle,
                                                          rocblas_operation_none,
                                                          transA,
                                                          width,
                                                          BLOCK,
                                                          r * BLOCK,
                                                          &negative_one<T>,
                                                          B,
                                                          ldb,
                                                          A,
                                                          lda,
                                                          alpha,
                                                          x_temp,
                                                          width,
                                                          batch_count,
                                                          offsetA,
                                                          offsetB,
                                                          0);
                        }
                        else
                        {
                            rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                            rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                            int32_t           solution_index = 0;
                            uint32_t          flags          = 0;

                            rocblas_gemm_batched_ex_offset(handle,
                                                    rocblas_operation_none,
                                                    transA,
                                                    width,
                                                    BLOCK,
                                                    r * BLOCK,
                                                    &negative_one<T>,
                                                    B,
                                                    compute_type,
                                                    ldb,
                                                    A,
                                                    compute_type,
                                                    lda,
                                                    alpha,
                                                    B,
                                                    compute_type,
                                                    ldb,
                                                    x_temp,
                                                    compute_type,
                                                    width,
                                                    batch_count,
                                                    compute_type,
                                                    algo,
                                                    offsetB,
                                                    offsetA,
                                                    j * BLOCK * ldb + w * B_chunk_size,
                                                    0,
                                                    solution_index,
                                                    flags);
                        }
                    }

                    rocblas_gemm_batched_template(handle,
                                                  rocblas_operation_none,
                                                  transA,
                                                  width,
                                                  BLOCK,
                                                  BLOCK,
                                                  r ? &one<T> : alpha,
                                                  x_temp,
                                                  width,
                                                  (const T**)invA,
                                                  BLOCK,
                                                  &zero<T>,
                                                  B,
                                                  ldb,
                                                  batch_count,
                                                  0,
                                                  j * BLOCK * BLOCK,
                                                  w * B_chunk_size * ldb + j * BLOCK * ldb);
                }
            }
        }

        return rocblas_status_success;
    }
} // \namespace

//////////////////////////////
//////////////////////////////
//////////////////////////////
template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsm_batched_template(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             T*                B[],
                                             rocblas_int       ldb,
                                             rocblas_int       batch_count,
                                             const T*          supplied_invA      = nullptr,
                                             rocblas_int       supplied_invA_size = 0)
{
    // return rocblas_status_not_implemented;
    if(batch_count == 0)
        return rocblas_status_success;

    if(transA == rocblas_operation_conjugate_transpose)
        transA = rocblas_operation_transpose;

    rocblas_int k = side == rocblas_side_left ? m : n;
    // Whether size is an exact multiple of blocksize
    const bool exact_blocks = (k % BLOCK) == 0;

    // perf_status indicates whether optimal performance is obtainable with available memory
    rocblas_status perf_status = rocblas_status_success;

    size_t invA_bytes   = 0;
    size_t invA_els     = 0;
    size_t c_temp_bytes = 0;
    size_t c_temp_els   = 0;

    // For user-supplied invA, check to make sure size is large enough
    // If not large enough, indicate degraded performance and ignore supplied invA
    if(supplied_invA && supplied_invA_size / BLOCK < k)
    {
        static int msg = fputs("WARNING: TRSM invA_size argument is too small; invA argument "
                               "is being ignored; TRSM performance is degraded\n",
                               stderr);
        perf_status    = rocblas_status_perf_degraded;
        supplied_invA  = nullptr;
    }
    rocblas_int stride_invA = k * BLOCK;
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
            c_temp_els = max(c_temp_els, remainder_els);
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
        x_temp_els   = m * n;
        x_temp_bytes = x_temp_els * sizeof(T) * batch_count;
    }

    // X and C temporaries can share space, so the maximum size is allocated
    size_t x_c_temp_bytes = max(x_temp_bytes, c_temp_bytes);
    size_t arrBytes = sizeof(T*) * batch_count;
    size_t xarrBytes = sizeof(T*) * batch_count;

    // If this is a device memory size query, set optimal size and return changed status
    if(handle->is_device_memory_size_query())
        return handle->set_optimal_device_memory_size(x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);

    // Attempt to allocate optimal memory size
    auto mem = handle->device_malloc(x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);
    if(!mem)
    {
        if(exact_blocks)
        {
            B_chunk_size   = 1; // Fall back on chunk size of 1 (like TRSV)
            x_temp_els     = BLOCK;
            x_temp_bytes   = x_temp_els * sizeof(T) * batch_count;
            x_c_temp_bytes = max(x_temp_bytes, c_temp_bytes);
            mem            = handle->device_malloc(x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);

        }
        if(!mem)
            return rocblas_status_memory_error;

        // Mark performance as degraded
        perf_status = rocblas_status_perf_degraded;

        // One-time warning about degraded performance
        static int msg = fputs("WARNING: Device memory allocation size is too small for TRSM; "
                               "TRSM performance is degraded\n",
                               stderr);
    }

    // Get pointers to allocated device memory
    // Note: Order of pointers in std::tie(...) must match order of sizes in handle->device_malloc(...)
    void* x_temp;
    void* x_temparr;
    void* invA;
    void* invAarr;
    std::tie(x_temp, x_temparr, invA, invAarr) = mem;

    // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    // Get alpha
    T alpha_h;
    if(saved_pointer_mode == rocblas_pointer_mode_host)
        alpha_h = *alpha;
    else
        RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));

    rocblas_status status = rocblas_status_success;

    T* invarrt[batch_count];
    for(int b = 0; b < batch_count; b++)
    {
       invarrt[b] = (T*)invA + b * invA_els;//(m*n);//BLOCK * k;
    }
    RETURN_IF_HIP_ERROR(hipMemcpy(invAarr, invarrt, batch_count * sizeof(T*), hipMemcpyHostToDevice));

    if(supplied_invA)
        invA = const_cast<T*>(supplied_invA);
    else
    {
        // c_temp and x_temp can reuse the same device memory
        T* c_temp = (T*)x_temp;
        T* ctemparrt[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            ctemparrt[b] = (T*)c_temp + b * c_temp_els;//b * m * n;

        }
        RETURN_IF_HIP_ERROR(hipMemcpy(x_temparr, ctemparrt, batch_count * sizeof(T*), hipMemcpyHostToDevice));

        status = rocblas_trtri_trsm_batched_template<BLOCK>(handle,
                                                            (T**)x_temparr,
                                                            uplo,
                                                            diag,
                                                            k,
                                                            A,
                                                            lda,
                                                            (T**)invAarr,
                                                            batch_count);

        if(status != rocblas_status_success)
            return status;
    }

    if(exact_blocks)
    {
        T* xtemparrt[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            xtemparrt[b] = (T*)x_temp + b * x_temp_els;//b * BLOCK * B_chunk_size;
        }
        hipMemcpy(x_temparr, xtemparrt, batch_count * sizeof(T*), hipMemcpyHostToDevice);

        status = special_trsm_batched_template<BLOCK>(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        m,
                                                        n,
                                                        &alpha_h,
                                                        A,
                                                        lda,
                                                        B,
                                                        ldb,
                                                        batch_count,
                                                        (const T**)invAarr,
                                                        B_chunk_size,
                                                        (T**)x_temparr);
    }
    else
    {
        T* xtemparrt[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            xtemparrt[b] = ((T*)x_temp) + b * x_temp_els;//b * m * n;
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(x_temparr, xtemparrt, batch_count * sizeof(T*), hipMemcpyHostToDevice));

        status
            = (side == rocblas_side_left
                    ? rocblas_trsm_batched_left<BLOCK, T>
                    : rocblas_trsm_batched_right<BLOCK, T>)(handle,
                                                            uplo,
                                                            transA,
                                                            m,
                                                            n,
                                                            &alpha_h,
                                                            A,
                                                            lda,
                                                            B,
                                                            ldb,
                                                            batch_count,
                                                            (const T**)invAarr,
                                                            (T**)x_temparr);

        if(status == rocblas_status_success)
            copy_block_unit(
                handle, m, n, (const T**)x_temparr, m, B, ldb, batch_count);
    }

    if(status != rocblas_status_success)
        return status;

    // If status is successful, return perf_status; else return error
    // return rocblas_status_success;
    return status == rocblas_status_success ? perf_status : status;
return rocblas_status_success;
}

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsm_strided_batched_template(rocblas_handle    handle,
                                                     rocblas_side      side,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transA,
                                                     rocblas_diagonal  diag,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     const T*          alpha,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_int       stride_A,
                                                     T*                B,
                                                     rocblas_int       ldb,
                                                     rocblas_int       stride_B,
                                                     rocblas_int       batch_count,
                                                     const T*          supplied_invA      = nullptr,
                                                     rocblas_int       supplied_invA_size = 0,
                                                     rocblas_int       stride_invA        = 0)
{
    if(batch_count == 0)
        return rocblas_status_success;

    if(transA == rocblas_operation_conjugate_transpose)
        transA = rocblas_operation_transpose;

    rocblas_int k = side == rocblas_side_left ? m : n;
    // Whether size is an exact multiple of blocksize
    const bool exact_blocks = (k % BLOCK) == 0;

    // perf_status indicates whether optimal performance is obtainable with available memory
    rocblas_status perf_status = rocblas_status_success;

    size_t invA_bytes   = 0;
    size_t c_temp_bytes = 0;

    // For user-supplied invA, check to make sure size is large enough
    // If not large enough, indicate degraded performance and ignore supplied invA
    if(supplied_invA && supplied_invA_size / BLOCK < k)
    {
        static int msg = fputs("WARNING: TRSM invA_size argument is too small; invA argument "
                               "is being ignored; TRSM performance is degraded\n",
                               stderr);
        perf_status    = rocblas_status_perf_degraded;
        supplied_invA  = nullptr;
    }

    if(!supplied_invA)
    {
        // Only allocate bytes for invA if supplied_invA == nullptr or supplied_invA_size is too small
        invA_bytes = sizeof(T) * BLOCK * k * batch_count + ((batch_count - 1) * stride_invA);

        // When k < BLOCK, C is unnecessary for trtri
        c_temp_bytes = (k / BLOCK) * (sizeof(T) * (BLOCK / 2) * (BLOCK / 2));

        // For the TRTRI last diagonal block we need remainder space if k % BLOCK != 0
        if(!exact_blocks)
        {
            // TODO: Make this more accurate -- right now it's much larger than necessary
            size_t remainder_bytes = sizeof(T) * ROCBLAS_TRTRI_NB * BLOCK * 2;

            // C is the maximum of the temporary space needed for TRTRI
            c_temp_bytes = max(c_temp_bytes, remainder_bytes);
        }
    }
    else
    {
        if(stride_invA < supplied_invA_size && batch_count > 1)
        {
            return rocblas_status_invalid_size;
        }
    }

    // Chunk size for special algorithm
    size_t B_chunk_size = 0;

    // Temporary solution matrix
    size_t x_temp_bytes;

    if(exact_blocks)
    {
        // Optimal B_chunk_size is the orthogonal dimension to k
        B_chunk_size = size_t(m) + size_t(n) - size_t(k);

        // When k % BLOCK == 0, we only need BLOCK * B_chunk_size space
        x_temp_bytes = sizeof(T) * BLOCK * B_chunk_size * batch_count;
    }
    else
    {
        // When k % BLOCK != 0, we need m * n space
        x_temp_bytes = sizeof(T) * m * n * batch_count;
    }

    // X and C temporaries can share space, so the maximum size is allocated
    size_t x_c_temp_bytes = max(x_temp_bytes, c_temp_bytes);

    // If this is a device memory size query, set optimal size and return changed status
    if(handle->is_device_memory_size_query())
        return handle->set_optimal_device_memory_size(x_c_temp_bytes, invA_bytes);

    // Attempt to allocate optimal memory size
    auto mem = handle->device_malloc(x_c_temp_bytes, invA_bytes);
    if(!mem)
    {
        if(exact_blocks)
        {
            B_chunk_size   = 1; // Fall back on chunk size of 1 (like TRSV)
            x_temp_bytes   = sizeof(T) * BLOCK * batch_count;
            x_c_temp_bytes = max(x_temp_bytes, c_temp_bytes);
            mem            = handle->device_malloc(x_c_temp_bytes, invA_bytes);
        }
        if(!mem)
            return rocblas_status_memory_error;

        // Mark performance as degraded
        perf_status = rocblas_status_perf_degraded;

        // One-time warning about degraded performance
        static int msg = fputs("WARNING: Device memory allocation size is too small for TRSM; "
                               "TRSM performance is degraded\n",
                               stderr);
    }

    // Get pointers to allocated device memory
    // Note: Order of pointers in std::tie(...) must match order of sizes in handle->device_malloc(...)
    void* x_temp;
    void* invA;
    std::tie(x_temp, invA) = mem;

    // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    // Get alpha
    T alpha_h;
    if(saved_pointer_mode == rocblas_pointer_mode_host)
        alpha_h = *alpha;
    else
        RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));

    rocblas_status status = rocblas_status_success;
    if(supplied_invA)
        invA = const_cast<T*>(supplied_invA);
    else
    {
        stride_invA = BLOCK * k;
        // c_temp and x_temp can reuse the same device memory
        auto c_temp = x_temp;

        // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
        int b  = 0;
        status = rocblas_trtri_trsm_strided_batched_template<BLOCK>(handle,
                                                                    (T*)c_temp,
                                                                    uplo,
                                                                    diag,
                                                                    k,
                                                                    A + b * stride_A,
                                                                    lda,
                                                                    stride_A,
                                                                    (T*)invA + b * stride_invA,
                                                                    stride_invA,
                                                                    batch_count);

        if(status != rocblas_status_success)
            return status;
    }

    if(exact_blocks)
    {
        status = special_trsm_strided_batched_template<BLOCK>(handle,
                                                              side,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              m,
                                                              n,
                                                              &alpha_h,
                                                              A,
                                                              lda,
                                                              stride_A,
                                                              B,
                                                              ldb,
                                                              stride_B,
                                                              batch_count,
                                                              (T*)invA,
                                                              stride_invA,
                                                              B_chunk_size,
                                                              (T*)x_temp);
    }
    else
    {
        status = (side == rocblas_side_left
                      ? rocblas_trsm_strided_batched_left<BLOCK, T>
                      : rocblas_trsm_strided_batched_right<BLOCK, T>)(handle,
                                                                      uplo,
                                                                      transA,
                                                                      m,
                                                                      n,
                                                                      &alpha_h,
                                                                      A,
                                                                      lda,
                                                                      stride_A,
                                                                      B,
                                                                      ldb,
                                                                      stride_B,
                                                                      batch_count,
                                                                      (T*)invA,
                                                                      stride_invA,
                                                                      (T*)x_temp);
        // Copy solution to B

        if(status == rocblas_status_success)
            copy_block_unit(
                handle, m, n, (const T*)x_temp, m, (m * n), (T*)B, ldb, stride_B, batch_count);
    }

    // If status is successful, return perf_status; else return error
    return status == rocblas_status_success ? perf_status : status;
}
