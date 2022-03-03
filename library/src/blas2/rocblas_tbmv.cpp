/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_tbmv.hpp"
#include "logging.hpp"

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
        }

        if(m < 0 || k < 0 || lda < k + 1 || !incx)
            return rocblas_status_invalid_size;

        if(!m)
        {
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return rocblas_status_success;
        }

        if(!A || !x)
            return rocblas_status_invalid_pointer;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(T) * m);

        auto w_mem = handle->device_malloc(sizeof(T) * m);
        if(!w_mem)
            return rocblas_status_memory_error;

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tbmv_check_numerics_status
                = rocblas_tbmv_check_numerics(rocblas_tbmv_name<T>,
                                              handle,
                                              m,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(tbmv_check_numerics_status != rocblas_status_success)
                return tbmv_check_numerics_status;
        }

        rocblas_status status = rocblas_tbmv_template(
            handle, uplo, transA, diag, m, k, A, 0, lda, 0, x, 0, incx, 0, 1, (T*)w_mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tbmv_check_numerics_status
                = rocblas_tbmv_check_numerics(rocblas_tbmv_name<T>,
                                              handle,
                                              m,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              1,
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
try
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
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
try
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
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
try
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
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
try
{
    return rocblas_tbmv_impl(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
