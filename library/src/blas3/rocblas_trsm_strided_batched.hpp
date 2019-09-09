/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "trsm_host.hpp"
#include "trtri_trsm.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

template <typename>
constexpr char rocblas_trsm_name[] = "unknown";
template <>
constexpr char rocblas_trsm_name<float>[] = "rocblas_strided_batched_strsm";
template <>
constexpr char rocblas_trsm_name<double>[] = "rocblas_strided_batched_dtrsm";

/* ============================================================================================ */

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsm_strided_batched_ex_impl(rocblas_handle    handle,
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
                                                    const T*          supplied_invA = nullptr,
                                                    rocblas_int       supplied_invA_size = 0,
                                                    rocblas_int       stride_invA        = 0)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    /////////////
    // LOGGING //
    /////////////
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
                            stride_A,
                            B,
                            ldb,
                            stride_B,
                            batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                log_bench(handle,
                            "./rocblas-bench -f trsm_strided_batched -r",
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
                            "--stride_A",
                            stride_A,
                            "--ldb",
                            ldb,
                            "--stride_B",
                            stride_B,
                            "--batch",
                            batch_count);
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
                            stride_A,
                            B,
                            ldb,
                            stride_B,
                            batch_count);
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
                        "stride_A",
                        stride_A,
                        "ldb",
                        ldb,
                        "stride_B",
                        stride_B,
                        "batch",
                        batch_count);
        }
    }

    /////////////////////
    // ARGUMENT CHECKS //
    /////////////////////
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
    if(batch_count < 0)
        return rocblas_status_invalid_size;

    if(batch_count > 1)
    {
        if(stride_A < lda * k || stride_B < ldb * n)
            return rocblas_status_invalid_size;
    }

    //////////////////////
    // MEMORY MANAGEMENT//
    //////////////////////
    // quick return if possible.
    // return status_size_unchanged if device memory size query
    if(!m || !n)
        return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                        : rocblas_status_success;

    return rocblas_trsm_strided_batched_template<BLOCK, T>(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            m,
                                                            n,
                                                            alpha,
                                                            A,
                                                            lda,
                                                            stride_A,
                                                            B,
                                                            ldb,
                                                            stride_B,
                                                            batch_count,
                                                            supplied_invA,
                                                            supplied_invA_size,
                                                            stride_invA);
}
