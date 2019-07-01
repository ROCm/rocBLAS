/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "definitions.h"
#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "trtri_trsm.hpp"
#include "utility.h"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <tuple>

#define A(ii, jj) (A + (ii) + (jj)*lda)
#define B(ii, jj) (B + (ii) + (jj)*ldb)
#define X(ii, jj) (X + (ii) + (jj)*m)
#define invA(ii) (invA + (ii)*BLOCK)

namespace
{
    // shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
    // you can use all 64K, but in practice no.
    constexpr rocblas_int STRSM_BLOCK = 128;
    constexpr rocblas_int DTRSM_BLOCK = 128;
    constexpr rocblas_int NB          = 16;

    template <typename T>
    constexpr T negative_one = -1;
    template <typename T>
    constexpr T zero = 0;
    template <typename T>
    constexpr T one = 1;

    /* ===============left==================================================== */
    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_left(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     const T*          A,
                                     rocblas_int       lda,
                                     T*                B,
                                     rocblas_int       ldb,
                                     const T*          invA,
                                     T*                X)
    {
        rocblas_int i, jb;

        // transB is always non-transpose
        rocblas_operation transB = rocblas_operation_none;

        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // left, lower no-transpose
                jb = min(BLOCK, m);
                rocblas_gemm_template(
                    handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero<T>, X, m);
                if(BLOCK < m)
                {
                    rocblas_gemm_template(handle,
                                          transA,
                                          transB,
                                          m - BLOCK,
                                          n,
                                          BLOCK,
                                          &negative_one<T>,
                                          A(BLOCK, 0),
                                          lda,
                                          X,
                                          m,
                                          alpha,
                                          B(BLOCK, 0),
                                          ldb);
                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);

                        rocblas_gemm_template(handle,
                                              transA,
                                              transB,
                                              jb,
                                              n,
                                              jb,
                                              &one<T>,
                                              invA(i),
                                              BLOCK,
                                              B(i, 0),
                                              ldb,
                                              &zero<T>,
                                              X(i, 0),
                                              m);
                        if(i + BLOCK < m)
                            rocblas_gemm_template(handle,
                                                  transA,
                                                  transB,
                                                  m - i - BLOCK,
                                                  n,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A(i + BLOCK, i),
                                                  lda,
                                                  X(i, 0),
                                                  m,
                                                  &one<T>,
                                                  B(i + BLOCK, 0),
                                                  ldb);
                    }
                }

#if 0
            for( i=0; i < m; i += BLOCK ) {
                jb = min(m-i, BLOCK);
                T *tmp = (i == 0) ? alpha : one;
                rocblas_gemm_template(handle, transA, transB, jb, n, jb, tmp, invA(i), BLOCK, B(i,0), ldb, &zero<T>, X(i,0), ldb);
                if(i + BLOCK < m){
                    rocblas_gemm_template(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negative_one<T>, A(i+BLOCK,i), lda, X(i,0), ldb, tmp, B(i+BLOCK,0), ldb);
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
                rocblas_gemm_template(handle,
                                      transA,
                                      transB,
                                      jb,
                                      n,
                                      jb,
                                      alpha,
                                      invA(i),
                                      BLOCK,
                                      B(i, 0),
                                      ldb,
                                      &zero<T>,
                                      X(i, 0),
                                      m);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template(handle,
                                          transA,
                                          transB,
                                          i,
                                          n,
                                          jb,
                                          &negative_one<T>,
                                          A(0, i),
                                          lda,
                                          X(i, 0),
                                          m,
                                          alpha,
                                          B,
                                          ldb);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        //{32, 35, 32, 32, 35, 35}
                        rocblas_gemm_template(handle,
                                              transA,
                                              transB,
                                              BLOCK,
                                              n,
                                              BLOCK,
                                              &one<T>,
                                              invA(i),
                                              BLOCK,
                                              B(i, 0),
                                              ldb,
                                              &zero<T>,
                                              X(i, 0),
                                              m);
                        if(i >= BLOCK)
                            rocblas_gemm_template(handle,
                                                  transA,
                                                  transB,
                                                  i,
                                                  n,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A(0, i),
                                                  lda,
                                                  X(i, 0),
                                                  m,
                                                  &one<T>,
                                                  B,
                                                  ldb);
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
                rocblas_gemm_template(handle,
                                      transA,
                                      transB,
                                      jb,
                                      n,
                                      jb,
                                      alpha,
                                      invA(i),
                                      BLOCK,
                                      B(i, 0),
                                      ldb,
                                      &zero<T>,
                                      X(i, 0),
                                      m);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template(handle,
                                          transA,
                                          transB,
                                          i,
                                          n,
                                          jb,
                                          &negative_one<T>,
                                          A(i, 0),
                                          lda,
                                          X(i, 0),
                                          m,
                                          alpha,
                                          B,
                                          ldb);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_template(handle,
                                              transA,
                                              transB,
                                              BLOCK,
                                              n,
                                              BLOCK,
                                              &one<T>,
                                              invA(i),
                                              BLOCK,
                                              B(i, 0),
                                              ldb,
                                              &zero<T>,
                                              X(i, 0),
                                              m);
                        if(i >= BLOCK)
                            rocblas_gemm_template(handle,
                                                  transA,
                                                  transB,
                                                  i,
                                                  n,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A(i, 0),
                                                  lda,
                                                  X(i, 0),
                                                  m,
                                                  &one<T>,
                                                  B,
                                                  ldb);
                    }
                }
            }
            else
            {
                // left, upper transpose
                jb = min(BLOCK, m);
                rocblas_gemm_template(
                    handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero<T>, X, m);
                if(BLOCK < m)
                {
                    rocblas_gemm_template(handle,
                                          transA,
                                          transB,
                                          m - BLOCK,
                                          n,
                                          BLOCK,
                                          &negative_one<T>,
                                          A(0, BLOCK),
                                          lda,
                                          X,
                                          m,
                                          alpha,
                                          B(BLOCK, 0),
                                          ldb);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemm_template(handle,
                                              transA,
                                              transB,
                                              jb,
                                              n,
                                              jb,
                                              &one<T>,
                                              invA(i),
                                              BLOCK,
                                              B(i, 0),
                                              ldb,
                                              &zero<T>,
                                              X(i, 0),
                                              m);
                        if(i + BLOCK < m)
                            rocblas_gemm_template(handle,
                                                  transA,
                                                  transB,
                                                  m - i - BLOCK,
                                                  n,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A(i, i + BLOCK),
                                                  lda,
                                                  X(i, 0),
                                                  m,
                                                  &one<T>,
                                                  B(i + BLOCK, 0),
                                                  ldb);
                    }
                }
            }
        } // transpose

        return rocblas_status_success;
    }

    /* ===============right==================================================== */
    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_right(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                B,
                                      rocblas_int       ldb,
                                      const T*          invA,
                                      T*                X)
    {
        rocblas_int i, jb;

        // transB is always non-transpose
        rocblas_operation transB = rocblas_operation_none;
        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // right, lower no-transpose
                jb = n % BLOCK == 0 ? BLOCK : n % BLOCK;
                i  = n - jb;
                rocblas_gemm_template(handle,
                                      transB,
                                      transA,
                                      m,
                                      jb,
                                      jb,
                                      alpha,
                                      B(0, i),
                                      ldb,
                                      invA(i),
                                      BLOCK,
                                      &zero<T>,
                                      X(0, i),
                                      m);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template(handle,
                                          transB,
                                          transA,
                                          m,
                                          i,
                                          jb,
                                          &negative_one<T>,
                                          X(0, i),
                                          m,
                                          A(i, 0),
                                          lda,
                                          alpha,
                                          B,
                                          ldb);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              BLOCK,
                                              BLOCK,
                                              &one<T>,
                                              B(0, i),
                                              ldb,
                                              invA(i),
                                              BLOCK,
                                              &zero<T>,
                                              X(0, i),
                                              m);
                        if(i >= BLOCK)
                            rocblas_gemm_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  i,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  X(0, i),
                                                  m,
                                                  A(i, 0),
                                                  lda,
                                                  &one<T>,
                                                  B,
                                                  ldb);
                    }
                }
            }
            else
            {
                // right, upper no-transpose
                jb = min(BLOCK, n);
                rocblas_gemm_template(
                    handle, transB, transA, m, jb, jb, alpha, B, ldb, invA, BLOCK, &zero<T>, X, m);
                if(BLOCK < n)
                {
                    rocblas_gemm_template(handle,
                                          transB,
                                          transA,
                                          m,
                                          n - BLOCK,
                                          BLOCK,
                                          &negative_one<T>,
                                          X,
                                          m,
                                          A(0, BLOCK),
                                          lda,
                                          alpha,
                                          B(0, BLOCK),
                                          ldb);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              jb,
                                              jb,
                                              &one<T>,
                                              B(0, i),
                                              ldb,
                                              invA(i),
                                              BLOCK,
                                              &zero<T>,
                                              X(0, i),
                                              m);
                        if(i + BLOCK < n)
                            rocblas_gemm_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  n - i - BLOCK,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  X(0, i),
                                                  m,
                                                  A(i, i + BLOCK),
                                                  lda,
                                                  &one<T>,
                                                  B(0, i + BLOCK),
                                                  ldb);
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
                rocblas_gemm_template(
                    handle, transB, transA, m, jb, jb, alpha, B, ldb, invA, BLOCK, &zero<T>, X, m);
                if(BLOCK < n)
                {
                    rocblas_gemm_template(handle,
                                          transB,
                                          transA,
                                          m,
                                          n - BLOCK,
                                          BLOCK,
                                          &negative_one<T>,
                                          X,
                                          m,
                                          A(BLOCK, 0),
                                          lda,
                                          alpha,
                                          B(0, BLOCK),
                                          ldb);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              jb,
                                              jb,
                                              &one<T>,
                                              B(0, i),
                                              ldb,
                                              invA(i),
                                              BLOCK,
                                              &zero<T>,
                                              X(0, i),
                                              m);
                        if(i + BLOCK < n)
                            rocblas_gemm_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  n - i - BLOCK,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  X(0, i),
                                                  m,
                                                  A(BLOCK + i, i),
                                                  lda,
                                                  &one<T>,
                                                  B(0, i + BLOCK),
                                                  ldb);
                    }
                }
            }
            else
            {
                // right, upper transpose
                jb = n % BLOCK == 0 ? BLOCK : n % BLOCK;
                i  = n - jb;
                rocblas_gemm_template(handle,
                                      transB,
                                      transA,
                                      m,
                                      jb,
                                      jb,
                                      alpha,
                                      B(0, i),
                                      ldb,
                                      invA(i),
                                      BLOCK,
                                      &zero<T>,
                                      X(0, i),
                                      m);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template(handle,
                                          transB,
                                          transA,
                                          m,
                                          i,
                                          jb,
                                          &negative_one<T>,
                                          X(0, i),
                                          m,
                                          A(0, i),
                                          lda,
                                          alpha,
                                          B,
                                          ldb);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_template(handle,
                                              transB,
                                              transA,
                                              m,
                                              BLOCK,
                                              BLOCK,
                                              &one<T>,
                                              B(0, i),
                                              ldb,
                                              invA(i),
                                              BLOCK,
                                              &zero<T>,
                                              X(0, i),
                                              m);
                        if(i >= BLOCK)
                            rocblas_gemm_template(handle,
                                                  transB,
                                                  transA,
                                                  m,
                                                  i,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  X(0, i),
                                                  m,
                                                  A(0, i),
                                                  lda,
                                                  &one<T>,
                                                  B,
                                                  ldb);
                    }
                }
            }
        } // tranpsose

        return rocblas_status_success;
    }

    template <typename T>
    __global__ void copy_matrix_trsm(rocblas_int rows,
                                     rocblas_int cols,
                                     rocblas_int elem_size,
                                     const T*    a,
                                     rocblas_int lda,
                                     T*          b,
                                     rocblas_int ldb)
    {
        size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

        if(tx < rows && ty < cols)
            b[tx + ldb * ty] = a[tx + lda * ty];
    }

    template <typename T>
    void copy_block_unit(rocblas_handle handle,
                         rocblas_int    m,
                         rocblas_int    n,
                         const T*       src,
                         rocblas_int    src_ld,
                         T*             dst,
                         rocblas_int    dst_ld)
    {
        rocblas_int blocksX = ((m - 1) / 128) + 1; // parameters for device kernel
        rocblas_int blocksY = ((n - 1) / 8) + 1;
        dim3        grid(blocksX, blocksY);
        dim3        threads(128, 8);

        hipLaunchKernelGGL(copy_matrix_trsm,
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
                           dst_ld);
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status special_trsm_template(rocblas_handle    handle,
                                         rocblas_side      side,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         const T*          alpha,
                                         const T*          A,
                                         rocblas_int       lda,
                                         T*                B,
                                         rocblas_int       ldb,
                                         const T*          invA,
                                         size_t            B_chunk_size,
                                         T*                x_temp)
    {
        size_t k          = side == rocblas_side_left ? m : n;
        size_t R          = k / BLOCK;
        size_t bsize      = side == rocblas_side_left ? n : m;
        size_t W          = 1 + (bsize - 1) / B_chunk_size;
        bool   arch_lt906 = handle->device_arch_id() < 906;

        for(int w = 0; w < W; w++)
        {
            if(side == rocblas_side_left)
            {
                T*  Bw = B + w * B_chunk_size * ldb;
                int width
                    = bsize > (w + 1) * B_chunk_size ? B_chunk_size : bsize - w * B_chunk_size;

                for(int r = 0; r < R; r++)
                {
                    int q = R - 1 - r;

                    int j = (((uplo == rocblas_fill_lower) && (transA == rocblas_operation_none))
                             || ((uplo == rocblas_fill_upper)
                                 && (transA == rocblas_operation_transpose)))
                                ? r
                                : q;

                    // copy a BLOCK*n piece we are solving at a time
                    if(r == 0 || arch_lt906)
                        copy_block_unit<T>(
                            handle, BLOCK, width, Bw + j * BLOCK, ldb, x_temp, BLOCK);

                    if(r > 0)
                    {
                        const T* A_current = nullptr;
                        T*       B_current = nullptr;

                        if((uplo == rocblas_fill_upper) && (transA == rocblas_operation_transpose))
                        {
                            A_current = A + r * BLOCK * lda;
                            B_current = Bw;
                        }
                        else if((uplo == rocblas_fill_lower) && (transA == rocblas_operation_none))
                        {
                            A_current = A + r * BLOCK;
                            B_current = Bw;
                        }
                        else if((uplo == rocblas_fill_lower)
                                && (transA == rocblas_operation_transpose))
                        {
                            A_current = A + q * BLOCK * lda + (q + 1) * BLOCK;
                            B_current = Bw + (q + 1) * BLOCK;
                        }
                        else // ((uplo == rocblas_fill_upper) && (transA == rocblas_operation_none))
                        {
                            A_current = A + (q + 1) * BLOCK * lda + q * BLOCK;
                            B_current = Bw + (q + 1) * BLOCK;
                        }

                        if(arch_lt906)
                        {
                            rocblas_gemm_template(handle,
                                                  transA,
                                                  rocblas_operation_none,
                                                  BLOCK,
                                                  width,
                                                  r * BLOCK,
                                                  &negative_one<T>,
                                                  A_current,
                                                  lda,
                                                  B_current,
                                                  ldb,
                                                  alpha,
                                                  (T*)x_temp,
                                                  BLOCK);
                        }
                        else
                        {
                            rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                            rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                            int32_t           solution_index = 0;
                            uint32_t          flags          = 0;

                            rocblas_gemm_ex(handle,
                                            transA,
                                            rocblas_operation_none,
                                            BLOCK,
                                            width,
                                            r * BLOCK,
                                            &negative_one<T>,
                                            A_current,
                                            compute_type,
                                            lda,
                                            B_current,
                                            compute_type,
                                            ldb,
                                            alpha,
                                            Bw + j * BLOCK,
                                            compute_type,
                                            ldb,
                                            (T*)x_temp,
                                            compute_type,
                                            BLOCK,
                                            compute_type,
                                            algo,
                                            solution_index,
                                            flags);
                        }
                    }

                    const T* theta = (r == 0 ? alpha : &one<T>);
                    rocblas_gemm_template(handle,
                                          transA,
                                          rocblas_operation_none,
                                          BLOCK,
                                          width,
                                          BLOCK,
                                          theta,
                                          ((T*)invA) + j * BLOCK * BLOCK,
                                          BLOCK,
                                          (T*)x_temp,
                                          BLOCK,
                                          &zero<T>,
                                          Bw + j * BLOCK,
                                          ldb);
                }
            }
            else
            {
                T*  Bw = B + size_t(w) * B_chunk_size;
                int width
                    = bsize > (w + 1) * B_chunk_size ? B_chunk_size : bsize - w * B_chunk_size;

                for(int r = 0; r < R; r++)
                {
                    int q = R - 1 - r;

                    int j = (uplo == rocblas_fill_lower && transA == rocblas_operation_transpose)
                                    || (uplo == rocblas_fill_upper
                                        && transA == rocblas_operation_none)
                                ? r
                                : q;

                    // copy a m*BLOCK piece we are solving at a time
                    if(r == 0 || arch_lt906)
                        copy_block_unit<T>(
                            handle, width, BLOCK, Bw + j * BLOCK * ldb, ldb, x_temp, width);

                    if(r > 0)
                    {
                        const T* A_current = nullptr;
                        T*       B_current = nullptr;

                        if((uplo == rocblas_fill_lower) && (transA == rocblas_operation_transpose))
                        {
                            A_current = A + r * BLOCK;
                            B_current = Bw;
                        }
                        else if((uplo == rocblas_fill_upper) && (transA == rocblas_operation_none))
                        {
                            A_current = A + r * BLOCK * lda;
                            B_current = Bw;
                        }
                        else if((uplo == rocblas_fill_upper)
                                && (transA == rocblas_operation_transpose))
                        {
                            A_current = A + size_t(q + 1) * BLOCK * lda + size_t(q) * BLOCK;
                            B_current = Bw + size_t(q + 1) * BLOCK * size_t(ldb);
                        }
                        else // ((uplo == rocblas_fill_lower) && (transA == rocblas_operation_none))
                        {
                            A_current = A + size_t(q) * BLOCK * lda + size_t(q + 1) * BLOCK;
                            B_current = Bw + size_t(q + 1) * BLOCK * size_t(ldb);
                        }

                        if(arch_lt906)
                        {
                            rocblas_gemm_template(handle,
                                                  rocblas_operation_none,
                                                  transA,
                                                  width,
                                                  BLOCK,
                                                  r * BLOCK,
                                                  &negative_one<T>,
                                                  B_current,
                                                  ldb,
                                                  A_current,
                                                  lda,
                                                  alpha,
                                                  (T*)x_temp,
                                                  width);
                        }
                        else
                        {
                            rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                            rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                            int32_t           solution_index = 0;
                            uint32_t          flags          = 0;

                            rocblas_gemm_ex(handle,
                                            rocblas_operation_none,
                                            transA,
                                            width,
                                            BLOCK,
                                            r * BLOCK,
                                            &negative_one<T>,
                                            B_current,
                                            compute_type,
                                            ldb,
                                            A_current,
                                            compute_type,
                                            lda,
                                            alpha,
                                            Bw + j * BLOCK * ldb,
                                            compute_type,
                                            ldb,
                                            (T*)x_temp,
                                            compute_type,
                                            width,
                                            compute_type,
                                            algo,
                                            solution_index,
                                            flags);
                        }
                    }

                    const T* theta = r == 0 ? alpha : &one<T>;

                    rocblas_gemm_template(handle,
                                          rocblas_operation_none,
                                          transA,
                                          width,
                                          BLOCK,
                                          BLOCK,
                                          theta,
                                          (T*)x_temp,
                                          width,
                                          ((T*)invA) + j * BLOCK * BLOCK,
                                          BLOCK,
                                          &zero<T>,
                                          Bw + j * BLOCK * ldb,
                                          ldb);
                }
            }
        }

        return rocblas_status_success;
    }

    template <typename>
    constexpr char rocblas_trsm_name[] = "unknown";
    template <>
    constexpr char rocblas_trsm_name<float>[] = "rocblas_strsm";
    template <>
    constexpr char rocblas_trsm_name<double>[] = "rocblas_dtrsm";

    /* ============================================================================================ */

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_ex_template(rocblas_handle    handle,
                                            rocblas_side      side,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const T*          alpha,
                                            const T*          A,
                                            rocblas_int       lda,
                                            T*                B,
                                            rocblas_int       ldb,
                                            const T*          supplied_invA      = nullptr,
                                            rocblas_int       supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto side_letter   = rocblas_side_letter(side);
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trsm_name<T>,
                              side,
                              uplo,
                              transA,
                              diag,
                              m,
                              n,
                              *alpha,
                              A,
                              lda,
                              B,
                              ldb);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f trsm -r",
                              rocblas_precision_string<T>,
                              "--side",
                              side_letter,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              "--alpha",
                              *alpha,
                              "--lda",
                              lda,
                              "--ldb",
                              ldb);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trsm_name<T>,
                              side,
                              uplo,
                              transA,
                              diag,
                              m,
                              n,
                              alpha,
                              A,
                              lda,
                              B,
                              ldb);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_trsm_name<T>,
                            "side",
                            side_letter,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "m",
                            m,
                            "n",
                            n,
                            "lda",
                            lda,
                            "ldb",
                            ldb);
            }
        }

        if(transA == rocblas_operation_conjugate_transpose)
            transA = rocblas_operation_transpose;

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(m < 0 || n < 0)
            return rocblas_status_invalid_size;
        if(!alpha || !A)
            return rocblas_status_invalid_pointer;

        // A is of size lda*k
        rocblas_int k = side == rocblas_side_left ? m : n;

        if(lda < k)
            return rocblas_status_invalid_size;
        if(!B)
            return rocblas_status_invalid_pointer;
        if(ldb < m)
            return rocblas_status_invalid_size;

        // quick return if possible.
        // return status_size_unchanged if device memory size query
        if(!m || !n)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        // perf_status indicates whether optimal performance is obtainable with available memory
        rocblas_status perf_status = rocblas_status_success;

        // For user-supplied invA, check to make sure size is large enough
        // If not large enough, indicate degraded performance and ignore supplied invA
        if(supplied_invA && supplied_invA_size / BLOCK < k)
        {
            supplied_invA = nullptr;
            perf_status   = rocblas_status_perf_degraded;
        }

        // Only allocate bytes for invA if supplied_invA == nullptr or supplied_invA_size is too small
        size_t invA_size     = supplied_invA ? 0 : size_t(k) * BLOCK;
        size_t c_temp_size   = size_t(k) * (BLOCK / 4);
        size_t x_temp_size   = size_t(n) * size_t(m);
        size_t max_temp_size = std::max(c_temp_size, x_temp_size);

        // If this is a device memory size query, set optimal size and return changed status
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(max_temp_size * sizeof(T),
                                                          invA_size * sizeof(T));
        // Attempt to allocate optimal memory size
        auto mem = handle->device_malloc(max_temp_size * sizeof(T), invA_size * sizeof(T));
        if(!mem)
        { // If allocation of optimal size failed, try smaller x_temp_size
            x_temp_size   = m;
            max_temp_size = std::max(c_temp_size, x_temp_size);

            // If allocation of smaller size fails, return rocblas_status_memory_error
            mem = handle->device_malloc(max_temp_size * sizeof(T), invA_size * sizeof(T));
            if(!mem)
                return rocblas_status_memory_error;

            // Indicate degraded performance
            perf_status = rocblas_status_perf_degraded;
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

        rocblas_status status;

        if(supplied_invA)
        {
            invA = const_cast<T*>(supplied_invA);
        }
        else
        {
            // c_temp and x_temp can reuse the same device memory
            auto c_temp = x_temp;

            // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
            status = rocblas_trtri_trsm_template<NB>(
                handle, (T*)c_temp, uplo, diag, k, A, lda, (T*)invA);
            if(status != rocblas_status_success)
                return status;
        }

        if(k % BLOCK == 0)
        {
            status = special_trsm_template<BLOCK>(handle,
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
                                                  (T*)invA,
                                                  x_temp_size,
                                                  (T*)x_temp);
        }
        else
        {
            status = (side == rocblas_side_left ? rocblas_trsm_left<BLOCK, T>
                                                : rocblas_trsm_right<BLOCK, T>)(handle,
                                                                                uplo,
                                                                                transA,
                                                                                m,
                                                                                n,
                                                                                alpha,
                                                                                A,
                                                                                lda,
                                                                                B,
                                                                                ldb,
                                                                                (T*)invA,
                                                                                (T*)x_temp);
            // Copy solution to B
            if(status == rocblas_status_success)
                copy_block_unit(handle, m, n, (const T*)x_temp, m, (T*)B, ldb);
        }

        // If status is successful, return perf_status; else return error
        return status == rocblas_status_success ? perf_status : status;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsm(rocblas_handle    handle,
                             rocblas_side      side,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             rocblas_int       n,
                             const float*      alpha,
                             const float*      A,
                             rocblas_int       lda,
                             float*            B,
                             rocblas_int       ldb)
{
    return rocblas_trsm_ex_template<STRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

rocblas_status rocblas_dtrsm(rocblas_handle    handle,
                             rocblas_side      side,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             rocblas_int       n,
                             const double*     alpha,
                             const double*     A,
                             rocblas_int       lda,
                             double*           B,
                             rocblas_int       ldb)
{
    return rocblas_trsm_ex_template<DTRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

rocblas_status rocblas_trsm_ex(rocblas_handle    handle,
                               rocblas_side      side,
                               rocblas_fill      uplo,
                               rocblas_operation transA,
                               rocblas_diagonal  diag,
                               rocblas_int       m,
                               rocblas_int       n,
                               const void*       alpha,
                               const void*       A,
                               rocblas_int       lda,
                               void*             B,
                               rocblas_int       ldb,
                               const void*       invA,
                               rocblas_int       invA_size,
                               rocblas_datatype  compute_type)
{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsm_ex_template<DTRSM_BLOCK>(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     m,
                                                     n,
                                                     static_cast<const double*>(alpha),
                                                     static_cast<const double*>(A),
                                                     lda,
                                                     static_cast<double*>(B),
                                                     ldb,
                                                     static_cast<const double*>(invA),
                                                     invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsm_ex_template<STRSM_BLOCK>(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     m,
                                                     n,
                                                     static_cast<const float*>(alpha),
                                                     static_cast<const float*>(A),
                                                     lda,
                                                     static_cast<float*>(B),
                                                     ldb,
                                                     static_cast<const float*>(invA),
                                                     invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}

} // extern "C"
