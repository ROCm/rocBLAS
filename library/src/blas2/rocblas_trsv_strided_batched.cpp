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
    constexpr char rocblas_trsv_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_trsv_strided_batched_name<float>[] = "rocblas_strsv_strided_batched";
    template <>
    constexpr char rocblas_trsv_strided_batched_name<double>[] = "rocblas_dtrsv_strided_batched";
    template <>
    constexpr char rocblas_trsv_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_ctrsv_strided_batched";
    template <>
    constexpr char rocblas_trsv_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_ztrsv_strided_batched";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_strided_batched_ex_impl(rocblas_handle    handle,
                                                        rocblas_fill      uplo,
                                                        rocblas_operation transA,
                                                        rocblas_diagonal  diag,
                                                        rocblas_int       m,
                                                        const T*          A,
                                                        rocblas_int       lda,
                                                        rocblas_stride    stride_A,
                                                        T*                B,
                                                        rocblas_int       incx,
                                                        rocblas_stride    stride_x,
                                                        rocblas_int       batch_count,
                                                        const T*          supplied_invA = nullptr,
                                                        rocblas_int       supplied_invA_size = 0,
                                                        rocblas_stride    stride_invA        = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_trsv_strided_batched_name<T>,
                      uplo,
                      transA,
                      diag,
                      m,
                      A,
                      lda,
                      stride_A,
                      B,
                      incx,
                      stride_x,
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
                              "./rocblas-bench -f trsv_strided_batched -r",
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
                              "--stride_A",
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
                            rocblas_trsv_strided_batched_name<T>,
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
                            "stride_A",
                            stride_A,
                            "incx",
                            incx,
                            "stride_x",
                            stride_x,
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

        rocblas_status status = rocblas_trsv_template_mem<BLOCK, false, T>(handle,
                                                                           m,
                                                                           batch_count,
                                                                           &mem_x_temp,
                                                                           &mem_x_temp_arr,
                                                                           &mem_invA,
                                                                           &mem_invA_arr,
                                                                           supplied_invA,
                                                                           supplied_invA_size);

        rocblas_status status2 = rocblas_trsv_template<BLOCK, false, T>(handle,
                                                                        uplo,
                                                                        transA,
                                                                        diag,
                                                                        m,
                                                                        A,
                                                                        0,
                                                                        lda,
                                                                        stride_A,
                                                                        B,
                                                                        0,
                                                                        incx,
                                                                        stride_x,
                                                                        batch_count,
                                                                        mem_x_temp,
                                                                        mem_x_temp_arr,
                                                                        mem_invA,
                                                                        mem_invA_arr,
                                                                        supplied_invA,
                                                                        supplied_invA_size,
                                                                        0,
                                                                        stride_invA);

        return (status2 == rocblas_status_success) ? status : status2;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv_strided_batched(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             const float*      A,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_A,
                                             float*            x,
                                             rocblas_int       incx,
                                             rocblas_stride    stride_x,
                                             rocblas_int       batch_count)
try
{
    return rocblas_trsv_strided_batched_ex_impl<STRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, stride_A, x, incx, stride_x, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrsv_strided_batched(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             const double*     A,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_A,
                                             double*           x,
                                             rocblas_int       incx,
                                             rocblas_stride    stride_x,
                                             rocblas_int       batch_count)
try
{
    return rocblas_trsv_strided_batched_ex_impl<DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, stride_A, x, incx, stride_x, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrsv_strided_batched(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            transA,
                                             rocblas_diagonal             diag,
                                             rocblas_int                  m,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_stride               stride_A,
                                             rocblas_float_complex*       x,
                                             rocblas_int                  incx,
                                             rocblas_stride               stride_x,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_trsv_strided_batched_ex_impl<STRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, stride_A, x, incx, stride_x, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrsv_strided_batched(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             transA,
                                             rocblas_diagonal              diag,
                                             rocblas_int                   m,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_stride                stride_A,
                                             rocblas_double_complex*       x,
                                             rocblas_int                   incx,
                                             rocblas_stride                stride_x,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_trsv_strided_batched_ex_impl<DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, stride_A, x, incx, stride_x, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_trsv_strided_batched_ex(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_diagonal  diag,
                                               rocblas_int       m,
                                               const void*       A,
                                               rocblas_int       lda,
                                               rocblas_stride    stride_A,
                                               void*             x,
                                               rocblas_int       incx,
                                               rocblas_stride    stride_x,
                                               rocblas_int       batch_count,
                                               const void*       invA,
                                               rocblas_int       invA_size,
                                               rocblas_datatype  compute_type)
try
{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsv_strided_batched_ex_impl<DTRSV_BLOCK>(handle,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 m,
                                                                 static_cast<const double*>(A),
                                                                 lda,
                                                                 stride_A,
                                                                 static_cast<double*>(x),
                                                                 incx,
                                                                 stride_x,
                                                                 batch_count,
                                                                 static_cast<const double*>(invA),
                                                                 invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsv_strided_batched_ex_impl<STRSV_BLOCK>(handle,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 m,
                                                                 static_cast<const float*>(A),
                                                                 lda,
                                                                 stride_A,
                                                                 static_cast<float*>(x),
                                                                 incx,
                                                                 stride_x,
                                                                 batch_count,
                                                                 static_cast<const float*>(invA),
                                                                 invA_size);

    case rocblas_datatype_f64_c:
        return rocblas_trsv_strided_batched_ex_impl<DTRSV_BLOCK>(
            handle,
            uplo,
            transA,
            diag,
            m,
            static_cast<const rocblas_double_complex*>(A),
            lda,
            stride_A,
            static_cast<rocblas_double_complex*>(x),
            incx,
            stride_x,
            batch_count,
            static_cast<const rocblas_double_complex*>(invA),
            invA_size);

    case rocblas_datatype_f32_c:
        return rocblas_trsv_strided_batched_ex_impl<STRSV_BLOCK>(
            handle,
            uplo,
            transA,
            diag,
            m,
            static_cast<const rocblas_float_complex*>(A),
            lda,
            stride_A,
            static_cast<rocblas_float_complex*>(x),
            incx,
            stride_x,
            batch_count,
            static_cast<const rocblas_float_complex*>(invA),
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
