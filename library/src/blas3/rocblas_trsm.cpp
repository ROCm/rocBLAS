/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "trtri_trsm.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

// #define A(ii, jj) (A + (ii) + (jj)*lda)
// #define B(ii, jj) (B + (ii) + (jj)*ldb)
// #define X(ii, jj) (X + (ii) + (jj)*m)
// #define invA(ii) (invA + (ii)*BLOCK)

namespace
{
    using std::max;
    using std::min;

    // Shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
    // you can use all 64K, but in practice no.
    constexpr rocblas_int STRSM_BLOCK = 128;
    constexpr rocblas_int DTRSM_BLOCK = 128;

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
        static constexpr rocblas_operation transB = rocblas_operation_none;

        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // left, lower no-transpose
                jb = min(BLOCK, m);
                rocblas_gemm_template<false, false>(handle,
                                                    transA,
                                                    transB,
                                                    jb,
                                                    n,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    invA,
                                                    0,
                                                    BLOCK,
                                                    0,
                                                    (const T*)B,
                                                    0,
                                                    ldb,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    0,
                                                    m,
                                                    0,
                                                    1);
                if(BLOCK < m)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transA,
                                                        transB,
                                                        m - BLOCK,
                                                        n,
                                                        BLOCK,
                                                        &negative_one<T>,
                                                        0,
                                                        A,
                                                        BLOCK,
                                                        lda,
                                                        0,
                                                        (const T*)X,
                                                        0,
                                                        m,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        BLOCK,
                                                        ldb,
                                                        0,
                                                        1);
                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);

                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            jb,
                                                            n,
                                                            jb,
                                                            &one<T>,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            (const T*)B,
                                                            i,
                                                            ldb,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i,
                                                            m,
                                                            0,
                                                            1);
                        if(i + BLOCK
                           >= m) // this condition is not necessary at all and can be changed
                            // as if (i+BLOCK<m)
                            break;

                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            m - i - BLOCK,
                                                            n,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            A,
                                                            i + BLOCK + i * lda,
                                                            lda,
                                                            0,
                                                            (const T*)X,
                                                            i,
                                                            m,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            i + BLOCK,
                                                            ldb,
                                                            0,
                                                            1);
                    }
                }

#if 0
            for( i=0; i < m; i += BLOCK ) {
                jb = min(m-i, BLOCK);
                T *tmp = (i == 0) ? alpha : one;
                rocblas_gemm_template<false, false>(handle, transA, transB, jb, n, jb, tmp, 0, invA, i * BLOCK, BLOCK, 0, B, i, ldb, 0, &zero<T>, 0, X, i, ldb, 0, 1);
                if(i + BLOCK < m){
                    rocblas_gemm_template<false, false>(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negative_one<T>, 0, A,i + BLOCK + i * lda, lda, 0, X, i, ldb, 0, tmp, 0, B, i + BLOCK, ldb, 0, 1);
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
                rocblas_gemm_template<false, false>(handle,
                                                    transA,
                                                    transB,
                                                    jb,
                                                    n,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    invA,
                                                    i * BLOCK,
                                                    BLOCK,
                                                    0,
                                                    (const T*)B,
                                                    i,
                                                    ldb,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    i,
                                                    m,
                                                    0,
                                                    1);
                if(i - BLOCK >= 0)
                {

                    rocblas_gemm_template<false, false>(handle,
                                                        transA,
                                                        transB,
                                                        i,
                                                        n,
                                                        jb,
                                                        &negative_one<T>,
                                                        0,
                                                        A,
                                                        i * lda,
                                                        lda,
                                                        0,
                                                        (const T*)X,
                                                        i,
                                                        m,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        0,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        //{32, 35, 32, 32, 35, 35}
                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            BLOCK,
                                                            n,
                                                            BLOCK,
                                                            &one<T>,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            (const T*)B,
                                                            i,
                                                            ldb,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i,
                                                            m,
                                                            0,
                                                            1);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            i,
                                                            n,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            A,
                                                            i * lda,
                                                            lda,
                                                            0,
                                                            (const T*)X,
                                                            i,
                                                            m,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            0,
                                                            ldb,
                                                            0,
                                                            1);
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
                rocblas_gemm_template<false, false>(handle,
                                                    transA,
                                                    transB,
                                                    jb,
                                                    n,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    invA,
                                                    i * BLOCK,
                                                    BLOCK,
                                                    0,
                                                    (const T*)B,
                                                    i,
                                                    ldb,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    i,
                                                    m,
                                                    0,
                                                    1);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transA,
                                                        transB,
                                                        i,
                                                        n,
                                                        jb,
                                                        &negative_one<T>,
                                                        0,
                                                        A,
                                                        i,
                                                        lda,
                                                        0,
                                                        (const T*)X,
                                                        i,
                                                        m,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        0,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            BLOCK,
                                                            n,
                                                            BLOCK,
                                                            &one<T>,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            (const T*)B,
                                                            i,
                                                            ldb,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i,
                                                            m,
                                                            0,
                                                            1);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            i,
                                                            n,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            A,
                                                            i,
                                                            lda,
                                                            0,
                                                            (const T*)X,
                                                            i,
                                                            m,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            0,
                                                            ldb,
                                                            0,
                                                            1);
                    }
                }
            }
            else
            {
                // left, upper transpose
                jb = min(BLOCK, m);
                rocblas_gemm_template<false, false>(handle,
                                                    transA,
                                                    transB,
                                                    jb,
                                                    n,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    invA,
                                                    0,
                                                    BLOCK,
                                                    0,
                                                    (const T*)B,
                                                    0,
                                                    ldb,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    0,
                                                    m,
                                                    0,
                                                    1);
                if(BLOCK < m)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transA,
                                                        transB,
                                                        m - BLOCK,
                                                        n,
                                                        BLOCK,
                                                        &negative_one<T>,
                                                        0,
                                                        A,
                                                        BLOCK * lda,
                                                        lda,
                                                        0,
                                                        (const T*)X,
                                                        0,
                                                        m,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        BLOCK,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            jb,
                                                            n,
                                                            jb,
                                                            &one<T>,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            (const T*)B,
                                                            i,
                                                            ldb,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i,
                                                            m,
                                                            0,
                                                            1);
                        if(i + BLOCK >= m)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transA,
                                                            transB,
                                                            m - i - BLOCK,
                                                            n,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            A,
                                                            i + (i + BLOCK) * lda,
                                                            lda,
                                                            0,
                                                            (const T*)X,
                                                            i,
                                                            m,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            i + BLOCK,
                                                            ldb,
                                                            0,
                                                            1);
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
        static constexpr rocblas_operation transB = rocblas_operation_none;

        if(transA == transB)
        {
            if(uplo == rocblas_fill_lower)
            {
                // right, lower no-transpose
                jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
                i  = n - jb;
                rocblas_gemm_template<false, false>(handle,
                                                    transB,
                                                    transA,
                                                    m,
                                                    jb,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    (const T*)B,
                                                    i * ldb,
                                                    ldb,
                                                    0,
                                                    invA,
                                                    i * BLOCK,
                                                    BLOCK,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    i * m,
                                                    m,
                                                    0,
                                                    1);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transB,
                                                        transA,
                                                        m,
                                                        i,
                                                        jb,
                                                        &negative_one<T>,
                                                        0,
                                                        (const T*)X,
                                                        i * m,
                                                        m,
                                                        0,
                                                        A,
                                                        i,
                                                        lda,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        0,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            BLOCK,
                                                            BLOCK,
                                                            &one<T>,
                                                            0,
                                                            (const T*)B,
                                                            i * m,
                                                            ldb,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            1);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            i,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            (const T*)X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            A,
                                                            i,
                                                            lda,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            0,
                                                            ldb,
                                                            0,
                                                            1);
                    }
                }
            }
            else
            {
                // right, upper no-transpose
                jb = min(BLOCK, n);
                rocblas_gemm_template<false, false>(handle,
                                                    transB,
                                                    transA,
                                                    m,
                                                    jb,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    (const T*)B,
                                                    0,
                                                    ldb,
                                                    0,
                                                    invA,
                                                    0,
                                                    BLOCK,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    0,
                                                    m,
                                                    0,
                                                    1);
                if(BLOCK < n)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transB,
                                                        transA,
                                                        m,
                                                        n - BLOCK,
                                                        BLOCK,
                                                        &negative_one<T>,
                                                        0,
                                                        (const T*)X,
                                                        0,
                                                        m,
                                                        0,
                                                        A,
                                                        BLOCK * lda,
                                                        lda,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        BLOCK * ldb,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            jb,
                                                            jb,
                                                            &one<T>,
                                                            0,
                                                            (const T*)B,
                                                            i * ldb,
                                                            ldb,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            1);
                        if(i + BLOCK >= n)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            n - i - BLOCK,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            (const T*)X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            A,
                                                            i + (i + BLOCK) * lda,
                                                            lda,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            (i + BLOCK) * ldb,
                                                            ldb,
                                                            0,
                                                            1);
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
                rocblas_gemm_template<false, false>(handle,
                                                    transB,
                                                    transA,
                                                    m,
                                                    jb,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    (const T*)B,
                                                    0,
                                                    ldb,
                                                    0,
                                                    invA,
                                                    0,
                                                    BLOCK,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    0,
                                                    m,
                                                    0,
                                                    1);
                if(BLOCK < n)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transB,
                                                        transA,
                                                        m,
                                                        n - BLOCK,
                                                        BLOCK,
                                                        &negative_one<T>,
                                                        0,
                                                        (const T*)X,
                                                        0,
                                                        m,
                                                        0,
                                                        A,
                                                        BLOCK,
                                                        lda,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        BLOCK * ldb,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = BLOCK; i < n; i += BLOCK)
                    {
                        jb = min(BLOCK, n - i);
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            jb,
                                                            jb,
                                                            &one<T>,
                                                            0,
                                                            (const T*)B,
                                                            i * ldb,
                                                            ldb,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            1);
                        if(i + BLOCK >= n)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            n - i - BLOCK,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            (const T*)X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            A,
                                                            BLOCK + i + i * lda,
                                                            lda,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            (i + BLOCK) * ldb,
                                                            ldb,
                                                            0,
                                                            1);
                    }
                }
            }
            else
            {
                // right, upper transpose
                jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
                i  = n - jb;
                rocblas_gemm_template<false, false>(handle,
                                                    transB,
                                                    transA,
                                                    m,
                                                    jb,
                                                    jb,
                                                    alpha,
                                                    0,
                                                    (const T*)B,
                                                    i * ldb,
                                                    ldb,
                                                    0,
                                                    invA,
                                                    i * BLOCK,
                                                    BLOCK,
                                                    0,
                                                    &zero<T>,
                                                    0,
                                                    X,
                                                    i * m,
                                                    m,
                                                    0,
                                                    1);
                if(i - BLOCK >= 0)
                {
                    rocblas_gemm_template<false, false>(handle,
                                                        transB,
                                                        transA,
                                                        m,
                                                        i,
                                                        jb,
                                                        &negative_one<T>,
                                                        0,
                                                        (const T*)X,
                                                        i * m,
                                                        m,
                                                        0,
                                                        A,
                                                        i * lda,
                                                        lda,
                                                        0,
                                                        alpha,
                                                        0,
                                                        B,
                                                        0,
                                                        ldb,
                                                        0,
                                                        1);

                    // remaining blocks
                    for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            BLOCK,
                                                            BLOCK,
                                                            &one<T>,
                                                            0,
                                                            (const T*)B,
                                                            i * ldb,
                                                            ldb,
                                                            0,
                                                            invA,
                                                            i * BLOCK,
                                                            BLOCK,
                                                            0,
                                                            &zero<T>,
                                                            0,
                                                            X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            1);
                        if(i - BLOCK < 0)
                            break;
                        rocblas_gemm_template<false, false>(handle,
                                                            transB,
                                                            transA,
                                                            m,
                                                            i,
                                                            BLOCK,
                                                            &negative_one<T>,
                                                            0,
                                                            (const T*)X,
                                                            i * m,
                                                            m,
                                                            0,
                                                            A,
                                                            i * lda,
                                                            lda,
                                                            0,
                                                            &one<T>,
                                                            0,
                                                            B,
                                                            0,
                                                            ldb,
                                                            0,
                                                            1);
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
        rocblas_int blocksX = (m - 1) / 128 + 1; // parameters for device kernel
        rocblas_int blocksY = (n - 1) / 8 + 1;
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
        bool   parity     = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_upper);
        size_t k          = side == rocblas_side_left ? m : n;
        size_t R          = k / BLOCK;
        size_t bsize      = side == rocblas_side_left ? n : m;
        size_t W          = 1 + (bsize - 1) / B_chunk_size;
        bool   arch_lt906 = handle->device_arch_id() < 906;

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
                        copy_block_unit(handle, BLOCK, width, Bw + j * BLOCK, ldb, x_temp, BLOCK);

                    if(r)
                    {
                        rocblas_int offA;
                        rocblas_int offB = parity ? 0 : (q + 1) * BLOCK;

                        if(transA == rocblas_operation_none)
                            offA = parity ? r * BLOCK : BLOCK * (q * lda + q + lda);
                        else
                            offA = parity ? r * BLOCK * lda : BLOCK * (q * lda + q + 1);
                        const T* A_current = A + offA;
                        T*       B_current = Bw + offB;

                        if(arch_lt906)
                        {
                            rocblas_gemm_template<false, false>(handle,
                                                                transA,
                                                                rocblas_operation_none,
                                                                BLOCK,
                                                                width,
                                                                r * BLOCK,
                                                                &negative_one<T>,
                                                                0,
                                                                (const T*)A,
                                                                offA,
                                                                lda,
                                                                0,
                                                                (const T*)Bw,
                                                                offB,
                                                                ldb,
                                                                0,
                                                                alpha,
                                                                0,
                                                                x_temp,
                                                                0,
                                                                BLOCK,
                                                                0,
                                                                1);
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
                                            x_temp,
                                            compute_type,
                                            BLOCK,
                                            compute_type,
                                            algo,
                                            solution_index,
                                            flags);
                        }
                    }

                    rocblas_gemm_template<false, false>(handle,
                                                        transA,
                                                        rocblas_operation_none,
                                                        BLOCK,
                                                        width,
                                                        BLOCK,
                                                        r ? &one<T> : alpha,
                                                        0,
                                                        invA,
                                                        j * BLOCK * BLOCK,
                                                        BLOCK,
                                                        0,
                                                        (const T*)x_temp,
                                                        0,
                                                        BLOCK,
                                                        0,
                                                        &zero<T>,
                                                        0,
                                                        Bw,
                                                        j * BLOCK,
                                                        ldb,
                                                        0,
                                                        1);
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
                        copy_block_unit(
                            handle, width, BLOCK, Bw + j * BLOCK * ldb, ldb, x_temp, width);

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
                            rocblas_gemm_template<false, false>(handle,
                                                                rocblas_operation_none,
                                                                transA,
                                                                width,
                                                                BLOCK,
                                                                r * BLOCK,
                                                                &negative_one<T>,
                                                                0,
                                                                (const T*)B_current,
                                                                0,
                                                                ldb,
                                                                0,
                                                                A_current,
                                                                0,
                                                                lda,
                                                                0,
                                                                alpha,
                                                                0,
                                                                x_temp,
                                                                0,
                                                                width,
                                                                0,
                                                                1);
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
                                            x_temp,
                                            compute_type,
                                            width,
                                            compute_type,
                                            algo,
                                            solution_index,
                                            flags);
                        }
                    }

                    rocblas_gemm_template<false, false>(handle,
                                                        rocblas_operation_none,
                                                        transA,
                                                        width,
                                                        BLOCK,
                                                        BLOCK,
                                                        r ? &one<T> : alpha,
                                                        0,
                                                        (const T*)x_temp,
                                                        0,
                                                        width,
                                                        0,
                                                        invA,
                                                        j * BLOCK * BLOCK,
                                                        BLOCK,
                                                        0,
                                                        &zero<T>,
                                                        0,
                                                        Bw,
                                                        j * BLOCK * ldb,
                                                        ldb,
                                                        0,
                                                        1);
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
    rocblas_status rocblas_trsm_ex_impl(rocblas_handle    handle,
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
            invA_bytes = sizeof(T) * BLOCK * k;

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

        // Chunk size for special algorithm
        size_t B_chunk_size = 0;

        // Temporary solution matrix
        size_t x_temp_bytes;

        if(exact_blocks)
        {
            // Optimal B_chunk_size is the orthogonal dimension to k
            B_chunk_size = size_t(m) + size_t(n) - size_t(k);

            // When k % BLOCK == 0, we only need BLOCK * B_chunk_size space
            x_temp_bytes = sizeof(T) * BLOCK * B_chunk_size;
        }
        else
        {
            // When k % BLOCK != 0, we need m * n space
            x_temp_bytes = sizeof(T) * m * n;
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
                x_temp_bytes   = sizeof(T) * BLOCK;
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
            // c_temp and x_temp can reuse the same device memory
            auto c_temp = x_temp;

            // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
            status = rocblas_trtri_trsm_template<BLOCK>(
                handle, (T*)c_temp, uplo, diag, k, A, lda, (T*)invA);
            if(status != rocblas_status_success)
                return status;
        }

        if(exact_blocks)
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
                                                  B_chunk_size,
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
                                                                                &alpha_h,
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
    return rocblas_trsm_ex_impl<STRSM_BLOCK>(
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
    return rocblas_trsm_ex_impl<DTRSM_BLOCK>(
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
        return rocblas_trsm_ex_impl<DTRSM_BLOCK>(handle,
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
        return rocblas_trsm_ex_impl<STRSM_BLOCK>(handle,
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
