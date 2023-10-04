/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "logging.hpp"
#include "rocblas_tbmv.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_tbmv_name[] = "unknown";
    template <>
    constexpr char rocblas_tbmv_name<float>[] = "rocblas_stbmv_batched";
    template <>
    constexpr char rocblas_tbmv_name<double>[] = "rocblas_dtbmv_batched";
    template <>
    constexpr char rocblas_tbmv_name<rocblas_float_complex>[] = "rocblas_ctbmv_batched";
    template <>
    constexpr char rocblas_tbmv_name<rocblas_double_complex>[] = "rocblas_ztbmv_batched";

    template <typename T>
    rocblas_status rocblas_tbmv_batched_impl(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             T* const          x[],
                                             rocblas_int       incx,
                                             rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transA_letter = rocblas_transpose_letter(transA);
                auto diag_letter   = rocblas_diag_letter(diag);

                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_tbmv_name<T>,
                              uplo,
                              transA,
                              diag,
                              n,
                              k,
                              A,
                              lda,
                              x,
                              incx,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f tbmv_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-n",
                              n,
                              "-k",
                              k,
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              "--batch_count",
                              batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                    log_profile(handle,
                                rocblas_tbmv_name<T>,
                                "uplo",
                                uplo_letter,
                                "transA",
                                transA_letter,
                                "diag",
                                diag_letter,
                                "N",
                                n,
                                "k",
                                k,
                                "lda",
                                lda,
                                "incx",
                                incx,
                                "batch_count",
                                batch_count);
            }
        }

        rocblas_status arg_status = rocblas_tbmv_arg_check<T>(
            handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(!A || !x)
            return rocblas_status_invalid_pointer;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(T) * n * batch_count,
                                                          sizeof(T*) * batch_count);
        auto w_mem = handle->device_malloc(sizeof(T) * n * batch_count, sizeof(T*) * batch_count);
        if(!w_mem)
            return rocblas_status_memory_error;

        void* w_mem_x_copy     = w_mem[0];
        void* w_mem_x_copy_arr = w_mem[1];

        setup_batched_array<256>(
            handle->get_stream(), (T*)w_mem_x_copy, n, (T**)w_mem_x_copy_arr, batch_count);

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tbmv_check_numerics_status
                = rocblas_tbmv_check_numerics(rocblas_tbmv_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tbmv_check_numerics_status != rocblas_status_success)
                return tbmv_check_numerics_status;
        }

        rocblas_status status = rocblas_tbmv_template(handle,
                                                      uplo,
                                                      transA,
                                                      diag,
                                                      n,
                                                      k,
                                                      A,
                                                      0,
                                                      lda,
                                                      0,
                                                      x,
                                                      0,
                                                      incx,
                                                      0,
                                                      batch_count,
                                                      (T* const*)w_mem_x_copy_arr);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tbmv_check_numerics_status
                = rocblas_tbmv_check_numerics(rocblas_tbmv_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tbmv_check_numerics_status != rocblas_status_success)
                return tbmv_check_numerics_status;
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

rocblas_status rocblas_stbmv_batched(rocblas_handle     handle,
                                     rocblas_fill       uplo,
                                     rocblas_operation  transA,
                                     rocblas_diagonal   diag,
                                     rocblas_int        n,
                                     rocblas_int        k,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     float* const       x[],
                                     rocblas_int        incx,
                                     rocblas_int        batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}

rocblas_status rocblas_dtbmv_batched(rocblas_handle      handle,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         n,
                                     rocblas_int         k,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     double* const       x[],
                                     rocblas_int         incx,
                                     rocblas_int         batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}

rocblas_status rocblas_ctbmv_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_operation                  transA,
                                     rocblas_diagonal                   diag,
                                     rocblas_int                        n,
                                     rocblas_int                        k,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        lda,
                                     rocblas_float_complex* const       x[],
                                     rocblas_int                        incx,
                                     rocblas_int                        batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}

rocblas_status rocblas_ztbmv_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_operation                   transA,
                                     rocblas_diagonal                    diag,
                                     rocblas_int                         n,
                                     rocblas_int                         k,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         lda,
                                     rocblas_double_complex* const       x[],
                                     rocblas_int                         incx,
                                     rocblas_int                         batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, n, k, A, lda, x, incx, batch_count);
}

} // extern "C"
