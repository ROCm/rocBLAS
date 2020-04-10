/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
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
                                             rocblas_int       m,
                                             rocblas_int       k,
                                             const T* const    A[],
                                             rocblas_int       lda,
                                             T* const          x[],
                                             rocblas_int       incx,
                                             rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

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
                          m,
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
                          "-m",
                          m,
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
                            "M",
                            m,
                            "k",
                            k,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "batch_count",
                            batch_count);
        }

        if(m < 0 || k < 0 || lda < k + 1 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;
        if(!m || !batch_count)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;
        if(!A || !x)
            return rocblas_status_invalid_pointer;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(T) * m * batch_count,
                                                          sizeof(T*) * batch_count);
        auto mem = handle->device_malloc(sizeof(T) * m * batch_count, sizeof(T*) * batch_count);

        void* mem_x_copy;
        void* mem_x_copy_arr;
        std::tie(mem_x_copy, mem_x_copy_arr) = mem;

        setup_batched_array<256>(
            handle->rocblas_stream, (T*)mem_x_copy, m, (T**)mem_x_copy_arr, batch_count);

        return rocblas_tbmv_template(handle,
                                     uplo,
                                     transA,
                                     diag,
                                     m,
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
                                     (T* const*)mem_x_copy_arr);
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
                                     rocblas_int        m,
                                     rocblas_int        k,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     float* const       x[],
                                     rocblas_int        incx,
                                     rocblas_int        batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

rocblas_status rocblas_dtbmv_batched(rocblas_handle      handle,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         m,
                                     rocblas_int         k,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     double* const       x[],
                                     rocblas_int         incx,
                                     rocblas_int         batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

rocblas_status rocblas_ctbmv_batched(rocblas_handle                     handle,
                                     rocblas_fill                       uplo,
                                     rocblas_operation                  transA,
                                     rocblas_diagonal                   diag,
                                     rocblas_int                        m,
                                     rocblas_int                        k,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        lda,
                                     rocblas_float_complex* const       x[],
                                     rocblas_int                        incx,
                                     rocblas_int                        batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

rocblas_status rocblas_ztbmv_batched(rocblas_handle                      handle,
                                     rocblas_fill                        uplo,
                                     rocblas_operation                   transA,
                                     rocblas_diagonal                    diag,
                                     rocblas_int                         m,
                                     rocblas_int                         k,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         lda,
                                     rocblas_double_complex* const       x[],
                                     rocblas_int                         incx,
                                     rocblas_int                         batch_count)
{
    return rocblas_tbmv_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

} // extern "C"
