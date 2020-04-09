/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemv.hpp"
#include "rocblas_trsv.hpp"
#include "utility.h"

namespace
{
    constexpr rocblas_int STRSV_BLOCK = 128;
    constexpr rocblas_int DTRSV_BLOCK = 128;

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
    rocblas_status rocblas_trsv_batched_ex_impl(rocblas_handle    handle,
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

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(m < 0 || lda < m || lda < 1 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;

        // quick return if possible.
        // return rocblas_status_size_unchanged if device memory size query
        if(!m || !batch_count)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        if(!A || !B)
            return rocblas_status_invalid_pointer;

        void* mem_x_temp;
        void* mem_x_temp_arr;
        void* mem_invA;
        void* mem_invA_arr;

        rocblas_status status = rocblas_trsv_template_mem<BLOCK, true, T>(handle,
                                                                          m,
                                                                          batch_count,
                                                                          &mem_x_temp,
                                                                          &mem_x_temp_arr,
                                                                          &mem_invA,
                                                                          &mem_invA_arr,
                                                                          supplied_invA,
                                                                          supplied_invA_size);

        rocblas_status status2 = rocblas_trsv_template<BLOCK, true, T>(handle,
                                                                       uplo,
                                                                       transA,
                                                                       diag,
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
                                                                       mem_x_temp,
                                                                       mem_x_temp_arr,
                                                                       mem_invA,
                                                                       mem_invA_arr,
                                                                       supplied_invA,
                                                                       supplied_invA_size);

        return (status2 == rocblas_status_success) ? status : status2;
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
    return rocblas_trsv_batched_ex_impl<STRSV_BLOCK>(
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
    return rocblas_trsv_batched_ex_impl<DTRSV_BLOCK>(
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
    return rocblas_trsv_batched_ex_impl<STRSV_BLOCK>(
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
    return rocblas_trsv_batched_ex_impl<DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_trsv_batched_ex(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_diagonal  diag,
                                       rocblas_int       m,
                                       const void* const A[],
                                       rocblas_int       lda,
                                       void* const       x[],
                                       rocblas_int       incx,
                                       rocblas_int       batch_count,
                                       const void*       invA,
                                       rocblas_int       invA_size,
                                       rocblas_datatype  compute_type)
try
{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsv_batched_ex_impl<DTRSV_BLOCK>(handle,
                                                         uplo,
                                                         transA,
                                                         diag,
                                                         m,
                                                         (const double* const*)(A),
                                                         lda,
                                                         (double* const*)(x),
                                                         incx,
                                                         batch_count,
                                                         (const double* const*)(invA),
                                                         invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsv_batched_ex_impl<STRSV_BLOCK>(handle,
                                                         uplo,
                                                         transA,
                                                         diag,
                                                         m,
                                                         (const float* const*)(A),
                                                         lda,
                                                         (float* const*)(x),
                                                         incx,
                                                         batch_count,
                                                         (const float* const*)(invA),
                                                         invA_size);

    case rocblas_datatype_f64_c:
        return rocblas_trsv_batched_ex_impl<DTRSV_BLOCK>(
            handle,
            uplo,
            transA,
            diag,
            m,
            (const rocblas_double_complex* const*)(A),
            lda,
            (rocblas_double_complex* const*)(x),
            incx,
            batch_count,
            (const rocblas_double_complex* const*)(invA),
            invA_size);

    case rocblas_datatype_f32_c:
        return rocblas_trsv_batched_ex_impl<STRSV_BLOCK>(
            handle,
            uplo,
            transA,
            diag,
            m,
            (const rocblas_float_complex* const*)(A),
            lda,
            (rocblas_float_complex* const*)(x),
            incx,
            batch_count,
            (const rocblas_float_complex* const*)(invA),
            invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
