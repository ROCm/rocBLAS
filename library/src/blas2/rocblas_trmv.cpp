/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_trmv.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

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
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;

        if(m < 0 || lda < m || lda < 1 || !incx)
            return rocblas_status_invalid_size;

        // quick return if possible.
        if(!m)
        {
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return rocblas_status_success;
        }

        size_t dev_bytes = m * sizeof(T);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        if(!A || !x)
            return rocblas_status_invalid_pointer;

        auto workspace = handle->device_malloc(dev_bytes);
        if(!workspace)
            return rocblas_status_memory_error;

        auto check_numerics = handle->check_numerics;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trmv_check_numerics_status
                = rocblas_trmv_check_numerics(rocblas_trmv_name<T>,
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
            if(trmv_check_numerics_status != rocblas_status_success)
                return trmv_check_numerics_status;
        }

        constexpr rocblas_int    batch_count_1 = 1;
        constexpr rocblas_int    offset_a = 0, offset_x = 0;
        constexpr rocblas_stride stride_a = 0, stride_x = 0, stride_w = 0;
        rocblas_status           status = rocblas_internal_trmv_template(handle,
                                                                         uplo,
                                                                         transA,
                                                                         diag,
                                                                         m,
                                                                         A,
                                                                         offset_a,
                                                                         lda,
                                                                         stride_a,
                                                                         x,
                                                                         offset_x,
                                                                         incx,
                                                                         stride_x,
                                                                         (T*)workspace,
                                                                         stride_w,
                                                                         batch_count_1);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trmv_check_numerics_status
                = rocblas_trmv_check_numerics(rocblas_trmv_name<T>,
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
            if(trmv_check_numerics_status != rocblas_status_success)
                return trmv_check_numerics_status;
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
