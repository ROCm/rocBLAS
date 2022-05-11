/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_trsv.hpp"
#include "utility.hpp"

namespace
{
    constexpr rocblas_int STRSV_BLOCK = 64;
    constexpr rocblas_int DTRSV_BLOCK = 64;
    constexpr rocblas_int CTRSV_BLOCK = 64;
    constexpr rocblas_int ZTRSV_BLOCK = 32;

    template <typename>
    constexpr char rocblas_trsv_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_trsv_batched_name<float>[] = "rocblas_strsv_batched";
    template <>
    constexpr char rocblas_trsv_batched_name<double>[] = "rocblas_dtrsv_batched";
    template <>
    constexpr char rocblas_trsv_batched_name<rocblas_float_complex>[] = "rocblas_ctrsv_batched";
    template <>
    constexpr char rocblas_trsv_batched_name<rocblas_double_complex>[] = "rocblas_ztrsv_batched";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_batched_impl(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             T* const          B[],
                                             rocblas_int       incx,
                                             rocblas_int       batch_count,
                                             const T* const    supplied_invA[]    = nullptr,
                                             rocblas_int       supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_trsv_batched_name<T>,
                          uplo,
                          transA,
                          diag,
                          m,
                          A,
                          lda,
                          B,
                          incx,
                          batch_count);

            if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transA_letter = rocblas_transpose_letter(transA);
                auto diag_letter   = rocblas_diag_letter(diag);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    if(handle->pointer_mode == rocblas_pointer_mode_host)
                        log_bench(handle,
                                  "./rocblas-bench -f trsv_batched -r",
                                  rocblas_precision_string<T>,
                                  "--uplo",
                                  uplo_letter,
                                  "--transposeA",
                                  transA_letter,
                                  "--diag",
                                  diag_letter,
                                  "-m",
                                  m,
                                  "--lda",
                                  lda,
                                  "--incx",
                                  incx,
                                  "--batch_count",
                                  batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                    log_profile(handle,
                                rocblas_trsv_batched_name<T>,
                                "uplo",
                                uplo_letter,
                                "transA",
                                transA_letter,
                                "diag",
                                diag_letter,
                                "M",
                                m,
                                "lda",
                                lda,
                                "incx",
                                incx,
                                "batch_count",
                                batch_count);
            }
        }

        size_t         dev_bytes;
        rocblas_status arg_status = rocblas_trsv_arg_check(
            handle, uplo, transA, diag, m, A, lda, B, incx, batch_count, dev_bytes);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        auto w_completed_sec = w_mem[0];

        auto check_numerics = handle->check_numerics;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trsv_check_numerics_status
                = rocblas_internal_trsv_check_numerics(rocblas_trsv_batched_name<T>,
                                                       handle,
                                                       m,
                                                       A,
                                                       0,
                                                       lda,
                                                       0,
                                                       B,
                                                       0,
                                                       incx,
                                                       0,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(trsv_check_numerics_status != rocblas_status_success)
                return trsv_check_numerics_status;
        }

        rocblas_status status
            = rocblas_internal_trsv_substitution_template<BLOCK, T>(handle,
                                                                    uplo,
                                                                    transA,
                                                                    diag,
                                                                    m,
                                                                    A,
                                                                    0,
                                                                    lda,
                                                                    0,
                                                                    nullptr,
                                                                    B,
                                                                    0,
                                                                    incx,
                                                                    0,
                                                                    batch_count,
                                                                    (rocblas_int*)w_completed_sec);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trsv_check_numerics_status
                = rocblas_internal_trsv_check_numerics(rocblas_trsv_batched_name<T>,
                                                       handle,
                                                       m,
                                                       A,
                                                       0,
                                                       lda,
                                                       0,
                                                       B,
                                                       0,
                                                       incx,
                                                       0,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(trsv_check_numerics_status != rocblas_status_success)
                return trsv_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv_batched(rocblas_handle     handle,
                                     rocblas_fill       uplo,
                                     rocblas_operation  transA,
                                     rocblas_diagonal   diag,
                                     rocblas_int        m,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     float* const       x[],
                                     rocblas_int        incx,

                                     rocblas_int batch_count)
try
{
    return rocblas_trsv_batched_impl<STRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrsv_batched(rocblas_handle      handle,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         m,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     double* const       x[],
                                     rocblas_int         incx,
                                     rocblas_int         batch_count)
try
{
    return rocblas_trsv_batched_impl<DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrsv_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_operation                  transA,
                                     rocblas_diagonal                   diag,
                                     rocblas_int                        m,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        lda,
                                     rocblas_float_complex* const       x[],
                                     rocblas_int                        incx,

                                     rocblas_int batch_count)
try
{
    return rocblas_trsv_batched_impl<CTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrsv_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_operation                   transA,
                                     rocblas_diagonal                    diag,
                                     rocblas_int                         m,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         lda,
                                     rocblas_double_complex* const       x[],
                                     rocblas_int                         incx,
                                     rocblas_int                         batch_count)
try
{
    return rocblas_trsv_batched_impl<ZTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
