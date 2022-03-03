/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_trsv_inverse.hpp"
#include "utility.hpp"

namespace
{
    constexpr rocblas_int TRSV_EX_BLOCK = 128;

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

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          "rocblas_trsv_batched_ex",
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
                                  "./rocblas-bench -f trsv_batched_ex -r",
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
                                "rocblas_trsv_batched_ex",
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

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(m < 0 || lda < m || lda < 1 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;

        // quick return if possible.
        if(!m || !batch_count)
        {
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return rocblas_status_success;
        }

        if(!A || !B)
            return rocblas_status_invalid_pointer;

        // Proxy object holds the allocation. It must stay alive as long as mem_* pointers below are alive.
        auto  w_mem = handle->device_malloc(0);
        void* w_mem_x_temp;
        void* w_mem_x_temp_arr;
        void* w_mem_invA;
        void* w_mem_invA_arr;

        rocblas_status perf_status
            = rocblas_internal_trsv_inverse_template_mem<BLOCK, true, T>(handle,
                                                                         m,
                                                                         batch_count,
                                                                         w_mem,
                                                                         w_mem_x_temp,
                                                                         w_mem_x_temp_arr,
                                                                         w_mem_invA,
                                                                         w_mem_invA_arr,
                                                                         supplied_invA,
                                                                         supplied_invA_size);

        // If this was a device memory query or an error occurred, return status
        if(perf_status != rocblas_status_success && perf_status != rocblas_status_perf_degraded)
            return perf_status;

        auto check_numerics = handle->check_numerics;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trsv_check_numerics_status
                = rocblas_internal_trsv_ex_check_numerics("rocblas_trsv_batched_ex",
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
            = rocblas_internal_trsv_inverse_template<BLOCK, true, T>(handle,
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
                                                                     w_mem_x_temp,
                                                                     w_mem_x_temp_arr,
                                                                     w_mem_invA,
                                                                     w_mem_invA_arr,
                                                                     supplied_invA,
                                                                     supplied_invA_size);

        status = (status != rocblas_status_success) ? status : perf_status;
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trsv_check_numerics_status
                = rocblas_internal_trsv_ex_check_numerics("rocblas_trsv_batched_ex",
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
        return rocblas_trsv_batched_ex_impl<TRSV_EX_BLOCK>(handle,
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
        return rocblas_trsv_batched_ex_impl<TRSV_EX_BLOCK>(handle,
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
        return rocblas_trsv_batched_ex_impl<TRSV_EX_BLOCK>(
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
        return rocblas_trsv_batched_ex_impl<TRSV_EX_BLOCK>(
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
