/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_tbmv.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"
#include <limits>

namespace
{
    template <typename>
    constexpr char rocblas_tbmv_name[] = "unknown";
    template <>
    constexpr char rocblas_tbmv_name<float>[] = "rocblas_stbmv";
    template <>
    constexpr char rocblas_tbmv_name<double>[] = "rocblas_dtbmv";
    template <>
    constexpr char rocblas_tbmv_name<rocblas_float_complex>[] = "rocblas_ctbmv";
    template <>
    constexpr char rocblas_tbmv_name<rocblas_double_complex>[] = "rocblas_ztbmv";

    template <typename T>
    rocblas_status rocblas_tbmv_impl(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       k,
                                     const T*          A,
                                     rocblas_int       lda,
                                     T*                x,
                                     rocblas_int       incx)
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

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(
                        handle, rocblas_tbmv_name<T>, uplo, transA, diag, m, k, A, lda, x, incx);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f tbmv -r",
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
                              incx);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(
                        handle, rocblas_tbmv_name<T>, uplo, transA, diag, m, k, A, lda, x, incx);
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
                            incx);
        }

        if(m < 0 || k < 0 || lda < m || lda < 1 || !incx || k >= lda)
            return rocblas_status_invalid_size;

        if(!m)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        if(!A || !x)
            return rocblas_status_invalid_pointer;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(T) * m);

        auto mem = handle->device_malloc(sizeof(T) * m);

        return rocblas_tbmv_template<T>(
            handle, uplo, transA, diag, m, k, A, 0, lda, 0, x, 0, incx, 0, 1, (T*)mem);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_stbmv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             rocblas_int       k,
                             const float*      A,
                             rocblas_int       lda,
                             float*            x,
                             rocblas_int       incx)
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

rocblas_status rocblas_dtbmv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             rocblas_int       k,
                             const double*     A,
                             rocblas_int       lda,
                             double*           x,
                             rocblas_int       incx)
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

rocblas_status rocblas_ctbmv(rocblas_handle               handle,
                             rocblas_fill                 uplo,
                             rocblas_operation            transA,
                             rocblas_diagonal             diag,
                             rocblas_int                  m,
                             rocblas_int                  k,
                             const rocblas_float_complex* A,
                             rocblas_int                  lda,
                             rocblas_float_complex*       x,
                             rocblas_int                  incx)
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

rocblas_status rocblas_ztbmv(rocblas_handle                handle,
                             rocblas_fill                  uplo,
                             rocblas_operation             transA,
                             rocblas_diagonal              diag,
                             rocblas_int                   m,
                             rocblas_int                   k,
                             const rocblas_double_complex* A,
                             rocblas_int                   lda,
                             rocblas_double_complex*       x,
                             rocblas_int                   incx)
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

} // extern "C"
