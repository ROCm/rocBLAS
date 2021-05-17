/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

// using templated paramter variation of syr2k/her2k for now
#include "../blas3/Tensile/gemm.hpp"
#include "definitions.hpp"
#include "rocblas_syr2k.hpp"

#define OFFSET_A(i1) offset_a + i1* a_s1
#define OFFSET_B(i1) offset_b + i1* b_s1
#define OFFSET_C(i1, i2) offset_c + i1* c_s1 + i2* c_s2

template <int MIN_NB, bool BATCHED, typename T, typename TScal, typename TPtr, typename TConstPtr>
rocblas_status rocblas_syrkx_template(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      TScal*            alpha,
                                      TConstPtr*        da,
                                      rocblas_int       lda,
                                      TConstPtr*        db,
                                      rocblas_int       ldb,
                                      TScal*            beta,
                                      TPtr*             dc,
                                      rocblas_int       ldc)
{
    static constexpr bool           TWOK     = false;
    static constexpr rocblas_int    offset_c = 0, offset_a = 0, offset_b = 0, batch_count = 1;
    static constexpr rocblas_stride stride_c = 0, stride_a = 0, stride_b = 0;

    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_int c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    // call syr2k strided_batched for n_nb diagonal blocks
    // clang-format off
    rocblas_internal_syr2k_template<TWOK>(
        handle, uplo, trans, nb, k, alpha,
        da, offset_a, lda, nb * a_s1,
        db, offset_b, ldb, nb * b_s1, beta,
        dc, offset_c, ldc, nb * (c_s1 + c_s2), n_nb);
    // clang-format on

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;
        // clang-format off
        rocblas_internal_syr2k_template<TWOK>(
              handle, uplo, trans, n_diag, k, alpha,
              da, OFFSET_A(i_diag), lda, stride_a,
              db, OFFSET_B(i_diag), ldb, stride_b, beta,
              dc, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    rocblas_operation trans_a
        = rocblas_operation_none == trans ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation trans_b
        = rocblas_operation_none == trans ? rocblas_operation_transpose : rocblas_operation_none;

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, and each iteration of the outer loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }

        // call gemm strided_batched for n_nb square blocks of size nb x nb
        if(rocblas_fill_lower == uplo)
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                 handle, trans_a, trans_b, nb, nb, k, alpha,
                 da, OFFSET_A(i_start),   lda, stride * a_s1,
                 db, OFFSET_B(0),         ldb, stride * b_s1,          beta,
                 dc, OFFSET_C(i_start, 0), ldc, stride * (c_s1 + c_s2), n_nb   )));
            // clang-format on
        }
        else
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                 handle, trans_a, trans_b, nb, nb, k, alpha,
                 da, OFFSET_A(0),          lda, stride * a_s1,
                 db, OFFSET_B(i_start),    ldb, stride * b_s1,          beta,
                 dc, OFFSET_C(0, i_start), ldc, stride * (c_s1 + c_s2), n_nb)));
            // clang-format on
        }

        // call gemm for remainder block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, n1, nb, k, alpha,
                     da, OFFSET_A(i1), lda, stride_a,
                     db, OFFSET_B(i2), ldb, stride_b, beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, n1, k, alpha,
                     da, OFFSET_A(i2), lda, stride_a,
                     db, OFFSET_B(i1), ldb, stride_b, beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}

template <int MIN_NB, bool BATCHED, typename T, typename TScal, typename TPtr, typename TConstPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal*            alpha,
                                    TConstPtr*        da,
                                    rocblas_int       offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    TConstPtr*        db,
                                    rocblas_int       offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    TScal*            beta,
                                    TPtr*             dc,
                                    rocblas_int       offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count)
{
    static constexpr bool TWOK = false;

    if(batch_count == 1)
    {
        return rocblas_syrkx_template<MIN_NB, BATCHED, T>(
            handle, uplo, trans, n, k, alpha, da, lda, db, ldb, beta, dc, ldc);
    }

    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_int c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    // diagonal blocks of size nb
    for(int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb; // diag block at c[i_diag, i_diag], size is nb
        // clang-format off
        rocblas_internal_syr2k_template<TWOK>(
              handle, uplo, trans, nb, k, alpha,
              da, OFFSET_A(i_diag), lda, stride_a,
              db, OFFSET_B(i_diag), ldb, stride_b, beta,
              dc, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;
        // clang-format off
        rocblas_internal_syr2k_template<TWOK>(
              handle, uplo, trans, n_diag, k, alpha,
              da, OFFSET_A(i_diag), lda, stride_a,
              db, OFFSET_B(i_diag), ldb, stride_b, beta,
              dc, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    rocblas_operation trans_a
        = rocblas_operation_none == trans ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation trans_b
        = rocblas_operation_none == trans ? rocblas_operation_transpose : rocblas_operation_none;

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, and each iteration of the outer loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }
        // gemm blocks of size nb x nb
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, nb, k, alpha,
                     da, OFFSET_A(i1), lda, stride_a,
                     db, OFFSET_B(i2), ldb, stride_b, beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, nb, k, alpha,
                     da, OFFSET_A(i2), lda, stride_a,
                     db, OFFSET_B(i1), ldb, stride_b, beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }

        // remainder gemm block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, n1, nb, k, alpha,
                     da, OFFSET_A(i1), lda, stride_a,
                     db, OFFSET_B(i2), ldb, stride_b, beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_a, trans_b, nb, n1, k, alpha,
                     da, OFFSET_A(i2), lda, stride_a,
                     db, OFFSET_B(i1), ldb, stride_b, beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}
#undef OFFSET_A
#undef OFFSET_B
#undef OFFSET_C
