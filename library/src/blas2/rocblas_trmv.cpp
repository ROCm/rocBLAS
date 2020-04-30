/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_trmv.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_trmv_name[] = "unknown";
    template <>
    constexpr char rocblas_trmv_name<float>[] = "rocblas_strmv";
    template <>
    constexpr char rocblas_trmv_name<double>[] = "rocblas_dtrmv";
    template <>
    constexpr char rocblas_trmv_name<rocblas_float_complex>[] = "rocblas_ctrmv";
    template <>
    constexpr char rocblas_trmv_name<rocblas_double_complex>[] = "rocblas_ztrmv";

    template <typename T>
    rocblas_status rocblas_trmv_impl(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     const T*          A,
                                     rocblas_int       lda,
                                     T*                x,
                                     rocblas_int       incx)
    {
        if(!handle)
        {
            return rocblas_status_invalid_handle;
        }

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);
            if(layer_mode & rocblas_layer_mode_log_trace)
            {
                log_trace(handle, rocblas_trmv_name<T>, uplo, transA, diag, m, A, lda, x, incx);
            }

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                log_bench(handle,
                          "./rocblas-bench",
                          "-f",
                          "trmv",
                          "-r",
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
                          incx);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_trmv_name<T>,
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
                            incx);
            }
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        {
            return rocblas_status_invalid_value;
        }

        if(m < 0 || lda < m || lda < 1 || !incx)
        {
            return rocblas_status_invalid_size;
        }

        //
        // quick return if possible.
        // return rocblas_status_size_unchanged if device memory size query
        //
        if(!m)
        {
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;
        }

        size_t dev_bytes = m * sizeof(T);
        if(handle->is_device_memory_size_query())
        {
            return handle->set_optimal_device_memory_size(dev_bytes);
        }

        if(!A || !x)
        {
            return rocblas_status_invalid_pointer;
        }

        T* w = (T*)handle->device_malloc(dev_bytes);
        if(!w)
        {
            return rocblas_status_memory_error;
        }

        return rocblas_trmv_nobatch_template(handle, uplo, transA, diag, m, A, lda, x, incx, w);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                                   \
    rocblas_status routine_name_(rocblas_handle    handle,                        \
                                 rocblas_fill      uplo,                          \
                                 rocblas_operation transA,                        \
                                 rocblas_diagonal  diag,                          \
                                 rocblas_int       m,                             \
                                 const T_*         A,                             \
                                 rocblas_int       lda,                           \
                                 T_*               x,                             \
                                 rocblas_int       incx)                          \
    try                                                                           \
    {                                                                             \
        return rocblas_trmv_impl(handle, uplo, transA, diag, m, A, lda, x, incx); \
    }                                                                             \
    catch(...)                                                                    \
    {                                                                             \
        return exception_to_rocblas_status();                                     \
    }

IMPL(rocblas_strmv, float);
IMPL(rocblas_dtrmv, double);
IMPL(rocblas_ctrmv, rocblas_float_complex);
IMPL(rocblas_ztrmv, rocblas_double_complex);

#undef IMPL

} // extern "C"
