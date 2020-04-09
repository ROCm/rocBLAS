/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_tbmv.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_tbmv_name[] = "unknown";
    template <>
    constexpr char rocblas_tbmv_name<float>[] = "rocblas_stbmv_strided_batched";
    template <>
    constexpr char rocblas_tbmv_name<double>[] = "rocblas_dtbmv_strided_batched";
    template <>
    constexpr char rocblas_tbmv_name<rocblas_float_complex>[] = "rocblas_ctbmv_strided_batched";
    template <>
    constexpr char rocblas_tbmv_name<rocblas_double_complex>[] = "rocblas_ztbmv_strided_batched";

    template <typename T>
    rocblas_status rocblas_tbmv_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transA,
                                                     rocblas_diagonal  diag,
                                                     rocblas_int       m,
                                                     rocblas_int       k,
                                                     const T*          A,
                                                     rocblas_int       lda,
                                                     rocblas_stride    stride_A,
                                                     T*                x,
                                                     rocblas_int       incx,
                                                     rocblas_stride    stride_x,
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
                          stride_A,
                          x,
                          incx,
                          stride_x,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                log_bench(handle,
                          "./rocblas-bench -f tbmv_strided_batched -r",
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
                          "--stride_a",
                          stride_A,
                          "--incx",
                          incx,
                          "--stride_x",
                          stride_x,
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
                            "stride_a",
                            stride_A,
                            "incx",
                            incx,
                            "stride_x",
                            stride_x,
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
            return handle->set_optimal_device_memory_size(sizeof(T) * m * batch_count);

        auto mem_x_copy = handle->device_malloc(sizeof(T) * m * batch_count);

        return rocblas_tbmv_template(handle,
                                     uplo,
                                     transA,
                                     diag,
                                     m,
                                     k,
                                     A,
                                     0,
                                     lda,
                                     stride_A,
                                     x,
                                     0,
                                     incx,
                                     stride_x,
                                     batch_count,
                                     (T*)mem_x_copy);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_stbmv_strided_batched(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       k,
                                             const float*      A,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_A,
                                             float*            x,
                                             rocblas_int       incx,
                                             rocblas_stride    stride_x,
                                             rocblas_int       batch_count)
{
    return rocblas_tbmv_strided_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, stride_A, x, incx, stride_x, batch_count);
}

rocblas_status rocblas_dtbmv_strided_batched(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       k,
                                             const double*     A,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_A,
                                             double*           x,
                                             rocblas_int       incx,
                                             rocblas_stride    stride_x,
                                             rocblas_int       batch_count)
{
    return rocblas_tbmv_strided_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, stride_A, x, incx, stride_x, batch_count);
}

rocblas_status rocblas_ctbmv_strided_batched(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            transA,
                                             rocblas_diagonal             diag,
                                             rocblas_int                  m,
                                             rocblas_int                  k,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_stride               stride_A,
                                             rocblas_float_complex*       x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stride_x,
                                             rocblas_int                  batch_count)
{
    return rocblas_tbmv_strided_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, stride_A, x, incx, stride_x, batch_count);
}

rocblas_status rocblas_ztbmv_strided_batched(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             transA,
                                             rocblas_diagonal              diag,
                                             rocblas_int                   m,
                                             rocblas_int                   k,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_stride                stride_A,
                                             rocblas_double_complex*       x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stride_x,
                                             rocblas_int                   batch_count)
{
    return rocblas_tbmv_strided_batched_impl(
        handle, uplo, transA, diag, m, k, A, lda, stride_A, x, incx, stride_x, batch_count);
}

} // extern "C"
