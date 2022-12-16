/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "Tensile/gemm.hpp"
#include "definitions.hpp"
#include "handle.hpp"
#include "herk_syrk_device.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_syr2k_her2k.hpp"
#include "utility.hpp"

template <typename T>
static const T beta_1 = T(1);

template <bool TWOK, bool HERK, typename T, typename TConstPtr, typename TPtr>
void syrkx_syr2k_dispatch(rocblas_fill      uplo,
                          rocblas_operation trans,
                          rocblas_int       n,
                          rocblas_int       k,
                          const T           alpha,
                          TConstPtr*        dA,
                          rocblas_int       lda,
                          rocblas_stride    stride_a,
                          TConstPtr*        dB,
                          rocblas_int       ldb,
                          rocblas_stride    stride_b,
                          const T           beta,
                          TPtr*             dC,
                          rocblas_int       ldc,
                          rocblas_stride    stride_c,
                          rocblas_int       batch_count,
                          hipStream_t       stream)
{
    if(TWOK)
    {
        syr2k_her2k_dispatch<TWOK, HERK, 32>(uplo,
                                             trans,
                                             n,
                                             k,
                                             alpha,
                                             dA,
                                             lda,
                                             stride_a,
                                             dB,
                                             ldb,
                                             stride_b,
                                             dC,
                                             ldc,
                                             stride_c,
                                             batch_count,
                                             stream);
    }
    else
    {
        syrkx_herkx_dispatch<HERK, T>(uplo,
                                      trans,
                                      n,
                                      k,
                                      alpha,
                                      dA,
                                      lda,
                                      stride_a,
                                      dB,
                                      ldb,
                                      stride_b,
                                      beta,
                                      dC,
                                      ldc,
                                      stride_c,
                                      batch_count,
                                      stream);
    }
}

#define OFFSET_A(i1) offset_a + i1* rocblas_stride(a_s1)
#define OFFSET_B(i1) offset_b + i1* rocblas_stride(b_s1)
#define OFFSET_C(i1, i2) offset_c + i1* rocblas_stride(c_s1) + i2* rocblas_stride(c_s2)

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <rocblas_int MIN_NB,
          bool        TWOK,
          bool        HERK,
          typename T,
          typename U,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syr2k_syrkx_block_recursive_template(rocblas_handle    handle,
                                                                     rocblas_fill      uplo,
                                                                     rocblas_operation trans,
                                                                     rocblas_int       n,
                                                                     rocblas_int       k,
                                                                     const T*          alpha,
                                                                     TConstPtr         da_in,
                                                                     rocblas_stride    offsetA,
                                                                     rocblas_int       lda,
                                                                     TConstPtr         db_in,
                                                                     rocblas_stride    offsetB,
                                                                     rocblas_int       ldb,
                                                                     const U*          beta,
                                                                     TPtr              dc_in,
                                                                     rocblas_stride    offsetC,
                                                                     rocblas_int       ldc)
{
    // quick return
    if(!n)
        return rocblas_status_success;

    constexpr bool BATCHED = false;

    // Can't be batched, so can just add offset at the beginning
    TConstPtr da = da_in + offsetA;
    TConstPtr db = db_in + offsetB;
    TPtr      dc = dc_in + offsetC;

    static constexpr rocblas_stride offset_c = 0, offset_a = 0, offset_b = 0;
    static constexpr rocblas_int    batch_count = 1;
    static constexpr rocblas_stride stride_c = 0, stride_a = 0, stride_b = 0;

    rocblas_stride a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_stride b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_stride c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    hipStream_t stream = handle->get_stream();

    if(TWOK)
    {
        // for syr2k/her2k we first scale C so we c an use directly for output without work buffer
        static constexpr int syr2k_SCALE_DIM_X = 128;
        static constexpr int syr2k_SCALE_DIM_Y = 8;
        rocblas_int          gx                = (n - 1) / (syr2k_SCALE_DIM_X) + 1;
        rocblas_int          gy                = (n - 1) / (syr2k_SCALE_DIM_Y) + 1;
        dim3                 syr2k_scale_grid(gx, gy, batch_count);
        dim3                 syr2k_scale_threads(syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y);

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel<syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y, HERK>),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           *alpha,
                           *beta,
                           dc,
                           ldc,
                           0);
    }

    // call syrkx_syr2k_dispatch with batch_count = n_nb for n_nb diagonal blocks
    // clang-format off
    syrkx_syr2k_dispatch<TWOK, HERK, T>(uplo, trans, nb, k, *alpha,
                         da, lda, nb * a_s1,
                         db, ldb, nb * b_s1, *beta,
                         dc, ldc, nb * (c_s1 + c_s2), n_nb, stream);
    // clang-format on

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;
        // call syrkx_syr2k_dispatch for one remainder diagonal block of size n_diag
        // clang-format off
        syrkx_syr2k_dispatch<TWOK, HERK, T>(uplo, trans, n_diag, k, *alpha,
                          da + i_diag * a_s1, lda, stride_a,
                          db + i_diag * b_s1, ldb, stride_b, *beta,
                          dc + i_diag * (c_s1 + c_s2), ldc, stride_c, batch_count, stream);
        // clang-format on
    }

    rocblas_operation trans_orig
        = rocblas_operation_none == trans
              ? rocblas_operation_none
              : (HERK ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose);
    rocblas_operation trans_opp
        = rocblas_operation_none == trans
              ? (HERK ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose)
              : rocblas_operation_none;
    const T alpha_conj = conj(*alpha);
    T*      alpha_conj_h;

    if(handle->is_stream_in_capture_mode())
    {
        alpha_conj_h = (T*)handle->host_malloc(sizeof(T));
        std::memcpy(alpha_conj_h, &alpha_conj, sizeof(T));
    }

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, then for each iteration of nb,i_start loop:
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

        // call gemm with batch_count = n_nb for n_nb square blocks of size nb x nb
        if(rocblas_fill_lower == uplo)
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                 handle, trans_orig, trans_opp, nb, nb, k, alpha,
                 da, OFFSET_A(i_start),    lda, stride * a_s1,
                 db, OFFSET_B(0),          ldb, stride * b_s1, TWOK ? &beta_1<T> : (T*) beta,
                 dc, OFFSET_C(i_start, 0), ldc, stride * (c_s1 + c_s2), n_nb   )));

            // a second call to gemm in the TWOK case
            if(TWOK)
            {
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                    handle, trans_orig, trans_opp, nb, nb, k, (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                    db, OFFSET_B(i_start),    ldb, stride * b_s1,
                    da, OFFSET_A(0),          lda, stride * a_s1, &beta_1<T>,
                    dc, OFFSET_C(i_start, 0), ldc, stride * (c_s1 + c_s2), n_nb   )));
            }
            // clang-format on
        }
        else
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                 handle, trans_orig, trans_opp, nb, nb, k, alpha,
                 da, OFFSET_A(0),          lda, stride * a_s1,
                 db, OFFSET_B(i_start),    ldb, stride * b_s1,   TWOK ? &beta_1<T> : (T*) beta,
                 dc, OFFSET_C(0, i_start), ldc, stride * (c_s1 + c_s2), n_nb)));

            if(TWOK)
            {
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                    handle, trans_orig, trans_opp, nb, nb, k, (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                    db, OFFSET_B(0),          ldb, stride * b_s1,
                    da, OFFSET_A(i_start),    lda, stride * a_s1, &beta_1<T>,
                    dc, OFFSET_C(0, i_start), ldc, stride * (c_s1 + c_s2), n_nb)));
            }
            // clang-format on
        }

        // call gemm for remainder block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_stride i1 = i_start + n_nb * stride;
            rocblas_stride i2 = i1 - nb;
            rocblas_stride n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                     handle, trans_orig, trans_opp, n1, nb, k, alpha,
                     da, OFFSET_A(i1),     lda, stride_a,
                     db, OFFSET_B(i2),     ldb, stride_b, TWOK ? &beta_1<T> : (T*) beta,
                     dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));

                if(TWOK)
                {
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                        handle, trans_orig, trans_opp, n1, nb, k, (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                        db, OFFSET_B(i1),     ldb, stride_b,
                        da, OFFSET_A(i2),     lda, stride_a, &beta_1<T>,
                        dc, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                }
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                     handle, trans_orig, trans_opp, nb, n1, k, alpha,
                     da, OFFSET_A(i2),     lda, stride_a,
                     db, OFFSET_B(i1),     ldb, stride_b, TWOK ? &beta_1<T> : (T*) beta,
                     dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));

                if(TWOK)
                {
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED>(
                        handle, trans_orig, trans_opp, nb, n1, k, (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                        db, OFFSET_B(i2),     ldb, stride_b,
                        da, OFFSET_A(i1),     lda, stride_a, &beta_1<T>,
                        dc, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                }
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}

template <rocblas_int MIN_NB,
          bool        BATCHED,
          bool        TWOK,
          bool        HERK,
          typename T,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syr2k_her2k_non_recursive_template(rocblas_handle    handle,
                                                                   rocblas_fill      uplo,
                                                                   rocblas_operation trans,
                                                                   rocblas_int       n,
                                                                   rocblas_int       k,
                                                                   const T*          alpha,
                                                                   TConstPtr         AP,
                                                                   rocblas_stride    offsetA,
                                                                   rocblas_int       lda,
                                                                   rocblas_stride    strideA,
                                                                   TConstPtr         BP,
                                                                   rocblas_stride    offsetB,
                                                                   rocblas_int       ldb,
                                                                   rocblas_stride    strideB,
                                                                   TPtr              CP,
                                                                   rocblas_stride    offsetC,
                                                                   rocblas_int       ldc,
                                                                   rocblas_stride    strideC,
                                                                   rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    static constexpr int syr2k_DIM_XY = 32;
    rocblas_int          bx           = (n - 1) / (syr2k_DIM_XY) + 1;
    rocblas_int          by           = (n - 1) / (syr2k_DIM_XY) + 1;
    dim3                 syr2k_grid(bx, by, batch_count);
    dim3                 syr2k_threads(syr2k_DIM_XY, syr2k_DIM_XY);

    TPtr           CP_krn;
    TConstPtr      BP_krn;
    TConstPtr      AP_krn;
    rocblas_stride a_st_or_of;
    rocblas_stride b_st_or_of;
    rocblas_stride c_st_or_of;

    if(BATCHED)
    {
        CP_krn     = CP;
        BP_krn     = BP;
        AP_krn     = AP;
        a_st_or_of = offsetA;
        b_st_or_of = offsetB;
        c_st_or_of = offsetC;
    }
    else
    {
        CP_krn     = CP + offsetC;
        BP_krn     = BP + offsetB;
        AP_krn     = AP + offsetA;
        a_st_or_of = strideA;
        b_st_or_of = strideB;
        c_st_or_of = strideC;
    }

    // Launch a herk kernel for syr2k.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, HERK, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               n,
                               k,
                               alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, HERK, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               n,
                               k,
                               alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
    }
    else
    {
        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, HERK, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               n,
                               k,
                               *alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, HERK, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               n,
                               k,
                               *alpha,
                               AP_krn,
                               lda,
                               a_st_or_of,
                               BP_krn,
                               ldb,
                               b_st_or_of,
                               CP_krn,
                               ldc,
                               c_st_or_of);
        }
    }

    return rocblas_status_success;
}

template <rocblas_int MIN_NB,
          bool        BATCHED,
          bool        TWOK,
          bool        HERK,
          typename T,
          typename U,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_her2k_template(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation trans,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const T*          alpha,
                                          TConstPtr         dA_in,
                                          rocblas_stride    offset_a,
                                          rocblas_int       lda,
                                          rocblas_stride    stride_a,
                                          TConstPtr         dB_in,
                                          rocblas_stride    offset_b,
                                          rocblas_int       ldb,
                                          rocblas_stride    stride_b,
                                          const U*          beta,
                                          TPtr              dC_in,
                                          rocblas_stride    offset_c,
                                          rocblas_int       ldc,
                                          rocblas_stride    stride_c,
                                          rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    // Note: alpha and beta always copied over to host by now
    if(*beta == 1 && (k == 0 || *alpha == 0))
        return rocblas_status_success;

    // Can't use block-recursive algorithm with batched version
    // Can use block-recursive algorithm with strided_batched when batch_count == 1
    if(!BATCHED && batch_count == 1)
    {
        return rocblas_internal_syr2k_syrkx_block_recursive_template<MIN_NB, TWOK, HERK, T>(
            handle,
            uplo,
            trans,
            n,
            k,
            alpha,
            dA_in,
            offset_a,
            lda,
            dB_in,
            offset_b,
            ldb,
            beta,
            dC_in,
            offset_c,
            ldc);
    }

    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_int c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    const T alpha_conj = conj(*alpha);

    T* alpha_conj_h;

    if(handle->is_stream_in_capture_mode())
    {
        alpha_conj_h = (T*)handle->host_malloc(sizeof(T));
        std::memcpy(alpha_conj_h, &alpha_conj, sizeof(T));
    }

    TPtr      dC = dC_in;
    TConstPtr dB = dB_in;
    TConstPtr dA = dA_in;

    static constexpr int syr2k_SCALE_DIM_X = 128;
    static constexpr int syr2k_SCALE_DIM_Y = 8;
    rocblas_int          gx                = (n - 1) / (syr2k_SCALE_DIM_X) + 1;
    rocblas_int          gy                = (n - 1) / (syr2k_SCALE_DIM_Y) + 1;
    dim3                 syr2k_scale_grid(gx, gy, batch_count);
    dim3                 syr2k_scale_threads(syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y);

    // first scale C so we can use directly for output without work buffer
    hipLaunchKernelGGL((syr2k_scale_kernel<syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y, HERK>),
                       syr2k_scale_grid,
                       syr2k_scale_threads,
                       0,
                       handle->get_stream(),
                       uplo == rocblas_fill_upper,
                       n,
                       k,
                       *alpha,
                       *beta,
                       dC,
                       ldc,
                       BATCHED ? offset_c : stride_c);

    if(k == 0)
        return rocblas_status_success;

    // n_nb diagonal blocks of size nb
    for(int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb; // diag block at c[i_diag, i_diag], size is nb

        // clang-format off
        rocblas_internal_syr2k_her2k_non_recursive_template<MIN_NB, BATCHED, TWOK, HERK>(
                handle, uplo, trans, nb, k, alpha,
                dA, OFFSET_A(i_diag),         lda, stride_a,
                dB, OFFSET_B(i_diag),         ldb, stride_b,
                dC, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;

        // clang-format off
        rocblas_internal_syr2k_her2k_non_recursive_template<MIN_NB, BATCHED, TWOK, HERK>(
                handle, uplo, trans, n_diag, k, alpha,
                dA, OFFSET_A(i_diag),         lda, stride_a,
                dB, OFFSET_B(i_diag),         ldb, stride_b,
                dC, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    rocblas_operation trans_orig
        = rocblas_operation_none == trans
              ? rocblas_operation_none
              : (HERK ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose);
    rocblas_operation trans_opp
        = rocblas_operation_none == trans
              ? (HERK ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose)
              : rocblas_operation_none;

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
        // n_nb gemm blocks of size nb x nb
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_orig, trans_opp, nb, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                        handle, trans_orig, trans_opp, nb, nb, k, (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                        dB, OFFSET_B(i1),     ldb, stride_b,
                        dA, OFFSET_A(i2),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_orig, trans_opp, nb, nb, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                        handle, trans_orig, trans_opp, nb, nb, k,(HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                        dB, OFFSET_B(i2),     ldb, stride_b,
                        dA, OFFSET_A(i1),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                    // clang-format on
                }
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
                     handle, trans_orig, trans_opp, n1, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                        handle, trans_orig, trans_opp, n1, nb, k,  (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                        dB, OFFSET_B(i1),     ldb, stride_b,
                        dA, OFFSET_A(i2),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                     handle, trans_orig, trans_opp, nb, n1, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_template<BATCHED, T>(
                        handle, trans_orig, trans_opp, nb, n1, k, (HERK? handle->is_stream_in_capture_mode()? alpha_conj_h : &alpha_conj : alpha),
                        dB, OFFSET_B(i2),     ldb, stride_b,
                        dA, OFFSET_A(i1),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
        }
    }

    return rocblas_status_success;
}

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her2k_syr2k_check_numerics(const char*       function_name,
                                                  rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation trans,
                                                  rocblas_int       n,
                                                  rocblas_int       k,
                                                  TConstPtr         A,
                                                  rocblas_int       lda,
                                                  rocblas_stride    strideA,
                                                  TConstPtr         B,
                                                  rocblas_int       ldb,
                                                  rocblas_stride    strideB,
                                                  TPtr              C,
                                                  rocblas_int       ldc,
                                                  rocblas_stride    strideC,
                                                  rocblas_int       batch_count,
                                                  const int         check_numerics,
                                                  bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;

    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              n,
                                                              k,
                                                              A,
                                                              0,
                                                              lda,
                                                              strideA,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              n,
                                                              k,
                                                              B,
                                                              0,
                                                              ldb,
                                                              strideB,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status = rocblas_internal_check_numerics_matrix_template(
        function_name,
        handle,
        rocblas_operation_none,
        uplo,
        HERM ? rocblas_client_hermitian_matrix : rocblas_client_symmetric_matrix,
        n,
        n,
        C,
        0,
        ldc,
        strideC,
        batch_count,
        check_numerics,
        is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syr2k*.cpp or her2k*.cpp

// clang-format off

#ifdef INSTANTIATE_SYR2K_HER2K_TEMPLATE
#error INSTANTIATE_SYR2K_HER2K_TEMPLATE already defined
#endif

#define INSTANTIATE_SYR2K_HER2K_TEMPLATE(MIN_NB_, BATCHED_, TWOK_, HERK_, T_, U_, TConstPtr_, TPtr_) \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syr2k_her2k_template       \
                                <MIN_NB_, BATCHED_, TWOK_, HERK_, T_, U_, TConstPtr_, TPtr_>         \
                                (rocblas_handle handle,                                              \
                                 rocblas_fill   uplo,                                                \
                                 rocblas_operation trans,                                            \
                                 rocblas_int       n,                                                \
                                 rocblas_int       k,                                                \
                                 const T_*         alpha,                                            \
                                 TConstPtr_        dA_in,                                            \
                                 rocblas_stride    offset_a,                                         \
                                 rocblas_int       lda,                                              \
                                 rocblas_stride    stride_a,                                         \
                                 TConstPtr_        dB_in,                                            \
                                 rocblas_stride    offset_b,                                         \
                                 rocblas_int       ldb,                                              \
                                 rocblas_stride    stride_b,                                         \
                                 const U_*         beta,                                             \
                                 TPtr_             dC_in,                                            \
                                 rocblas_stride    offset_c,                                         \
                                 rocblas_int       ldc,                                              \
                                 rocblas_stride    stride_c,                                         \
                                 rocblas_int       batch_count);

// syrk instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDZSYRK_NB, false, false, false, float, float, float const*, float*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDZSYRK_NB, false, false, false, double, double, double const*, double*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_CSYRK_NB,   false, false, false, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDZSYRK_NB, false, false, false, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)

// syrk_batched instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDSYRK_BATCHED_NB, true, false, false, float, float, float const* const*, float* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDSYRK_BATCHED_NB, true, false, false, double, double, double const* const*, double* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_CZSYRK_BATCHED_NB, true, false, false, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_CZSYRK_BATCHED_NB, true, false, false, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const* const*, rocblas_double_complex* const*)

// herk instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_CHERK_NB, false, false, true, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_ZHERK_NB, false, false, true, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)

// herk_batched instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HERK_BATCHED_NB, true, false, true, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HERK_BATCHED_NB, true, false, true, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const* const*, rocblas_double_complex* const*)

// syrkx instantiations (d, z unneeded as same block size as syrk)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SSYRKX_NB,   false, false, false, float, float, float const*, float*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_DCZSYRKX_NB, false, false, false, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)

// syrkx_batched instantiations (none needed as all have same block size as syrk_batched)

// herkx instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HERKX_NB, false, false, true, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HERKX_NB, false, false, true, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)

// herkx_batched instantiations (none needed as all have same block size as herk_batched)

// syr2k instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SSYR2K_NB, false, true, false, float, float, float const*, float*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_DCZSYR2K_NB, false, true, false, double, double, double const*, double*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_DCZSYR2K_NB,   false, true, false, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_DCZSYR2K_NB, false, true, false, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)

// syr2k_batched instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDSYR2K_BATCHED_NB, true, true, false, float, float, float const* const*, float* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_SDSYR2K_BATCHED_NB, true, true, false, double, double, double const* const*, double* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_CZSYR2K_BATCHED_NB, true, true, false, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_CZSYR2K_BATCHED_NB, true, true, false, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex const* const*, rocblas_double_complex* const*)

// her2k instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HER2K_NB, false, true, true, rocblas_float_complex, float, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HER2K_NB, false, true, true, rocblas_double_complex, double, rocblas_double_complex const*, rocblas_double_complex*)

// her2k_batched instantiations
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HER2K_BATCHED_NB, true, true, true, rocblas_float_complex, float, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_HER2K_TEMPLATE(ROCBLAS_HER2K_BATCHED_NB, true, true, true, rocblas_double_complex, double, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYR2K_HER2K_TEMPLATE

#ifdef INSTANTIATE_HER2K_SYR2K_NUMERICS
#error INSTANTIATE_HER2K_SYR2K_NUMERICS already defined
#endif

#define INSTANTIATE_HER2K_SYR2K_NUMERICS(HERM_, TConstPtr_, TPtr_)                        \
template rocblas_status rocblas_her2k_syr2k_check_numerics                                \
                                  <HERM_, TConstPtr_, TPtr_>                            \
                                  (const char*       function_name,                     \
                                   rocblas_handle handle,                               \
                                   rocblas_fill   uplo,                                 \
                                   rocblas_operation trans,                             \
                                   rocblas_int    n,                                    \
                                   rocblas_int    k,                                    \
                                   TConstPtr_     A,                                    \
                                   rocblas_int    lda,                                  \
                                   rocblas_stride strideA,                              \
                                   TConstPtr_     B,                                    \
                                   rocblas_int    ldb,                                  \
                                   rocblas_stride strideB,                              \
                                   TPtr_          C,                                    \
                                   rocblas_int    ldc,                                  \
                                   rocblas_stride strideC,                              \
                                   rocblas_int    batch_count,                          \
                                   const int      check_numerics,                       \
                                   bool           is_input);

// instantiate for rocblas_Xher2k_Xsyr2k and rocblas_Xher2k_Xsyr2k_strided_batched
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, float const*, float*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, double const*, double*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true, rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xher2k_Xsyr2k_batched
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, float const* const*, float* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, double const* const*, double* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS(false, rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_HER2K_SYR2K_NUMERICS( true, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HER2K_SYR2K_NUMERICS
// clang-format on
