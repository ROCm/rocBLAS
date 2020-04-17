/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_trsm.hpp"
#include "trtri_trsm.hpp"
#include "utility.h"

// Shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
// you can use all 64K, but in practice no.
constexpr rocblas_int STRSM_BLOCK = 128;
constexpr rocblas_int DTRSM_BLOCK = 128;

namespace
{
    template <typename>
    constexpr char rocblas_trsm_name[] = "unknown";
    template <>
    constexpr char rocblas_trsm_name<float>[] = "rocblas_strided_batched_strsm";
    template <>
    constexpr char rocblas_trsm_name<double>[] = "rocblas_strided_batched_dtrsm";
    template <>
    constexpr char rocblas_trsm_name<rocblas_float_complex>[] = "rocblas_strided_batched_ctrsm";
    template <>
    constexpr char rocblas_trsm_name<rocblas_double_complex>[] = "rocblas_strided_batched_ztrsm";

    /* ============================================================================================ */

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_strided_batched_ex_impl(rocblas_handle    handle,
                                                        rocblas_side      side,
                                                        rocblas_fill      uplo,
                                                        rocblas_operation transA,
                                                        rocblas_diagonal  diag,
                                                        rocblas_int       m,
                                                        rocblas_int       n,
                                                        const T*          alpha,
                                                        const T*          A,
                                                        rocblas_int       lda,
                                                        rocblas_stride    stride_A,
                                                        T*                B,
                                                        rocblas_int       ldb,
                                                        rocblas_stride    stride_B,
                                                        rocblas_int       batch_count,
                                                        const T*          supplied_invA = nullptr,
                                                        rocblas_int       supplied_invA_size = 0,
                                                        rocblas_stride    stride_invA        = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        if(handle->is_device_memory_size_query())
        {
            if(!m || !n || !batch_count)
                return rocblas_status_size_unchanged;
            // Regular case is being handled after logging for now since logic is complex.
        }

        /////////////
        // LOGGING //
        /////////////
        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto side_letter   = rocblas_side_letter(side);
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transA_letter = rocblas_transpose_letter(transA);
                auto diag_letter   = rocblas_diag_letter(diag);

                if(handle->pointer_mode == rocblas_pointer_mode_host)
                {
                    if(layer_mode & rocblas_layer_mode_log_trace)
                        log_trace(handle,
                                  rocblas_trsm_name<T>,
                                  side,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  n,
                                  *alpha,
                                  A,
                                  lda,
                                  stride_A,
                                  B,
                                  ldb,
                                  stride_B,
                                  batch_count);

                    if(layer_mode & rocblas_layer_mode_log_bench)
                    {
                        log_bench(handle,
                                  "./rocblas-bench -f trsm_strided_batched -r",
                                  rocblas_precision_string<T>,
                                  "--side",
                                  side_letter,
                                  "--uplo",
                                  uplo_letter,
                                  "--transposeA",
                                  transA_letter,
                                  "--diag",
                                  diag_letter,
                                  "-m",
                                  m,
                                  "-n",
                                  n,
                                  "--alpha",
                                  *alpha,
                                  "--lda",
                                  lda,
                                  "--stride_A",
                                  stride_A,
                                  "--ldb",
                                  ldb,
                                  "--stride_B",
                                  stride_B,
                                  "--batch_count",
                                  batch_count);
                    }
                }
                else
                {
                    if(layer_mode & rocblas_layer_mode_log_trace)
                        log_trace(handle,
                                  rocblas_trsm_name<T>,
                                  side,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  n,
                                  alpha,
                                  A,
                                  lda,
                                  stride_A,
                                  B,
                                  ldb,
                                  stride_B,
                                  batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    log_profile(handle,
                                rocblas_trsm_name<T>,
                                "side",
                                side_letter,
                                "uplo",
                                uplo_letter,
                                "transA",
                                transA_letter,
                                "diag",
                                diag_letter,
                                "m",
                                m,
                                "n",
                                n,
                                "lda",
                                lda,
                                "stride_A",
                                stride_A,
                                "ldb",
                                ldb,
                                "stride_B",
                                stride_B,
                                "batch_count",
                                batch_count);
                }
            }
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;

        // A is of size lda*k
        rocblas_int k = side == rocblas_side_left ? m : n;
        if(batch_count < 0 || m < 0 || n < 0 || lda < k || ldb < m)
            return rocblas_status_invalid_size;

        // quick return if possible.
        if(!m || !n || !batch_count)
            return rocblas_status_success;

        if(!alpha || !A || !B)
            return rocblas_status_invalid_pointer;

        //////////////////////
        // MEMORY MANAGEMENT//
        //////////////////////
        void*          mem_x_temp;
        void*          mem_x_temp_arr;
        void*          mem_invA;
        void*          mem_invA_arr;
        bool           optimal_mem;
        rocblas_status perf_status = rocblas_trsm_template_mem<BLOCK, false, T>(handle,
                                                                                side,
                                                                                m,
                                                                                n,
                                                                                batch_count,
                                                                                mem_x_temp,
                                                                                mem_x_temp_arr,
                                                                                mem_invA,
                                                                                mem_invA_arr,
                                                                                supplied_invA,
                                                                                supplied_invA_size);

        if(perf_status != rocblas_status_success && perf_status != rocblas_status_perf_degraded)
            return perf_status;

        optimal_mem = perf_status == rocblas_status_success;

        rocblas_status status = rocblas_trsm_template<BLOCK, false, T>(handle,
                                                                       side,
                                                                       uplo,
                                                                       transA,
                                                                       diag,
                                                                       m,
                                                                       n,
                                                                       alpha,
                                                                       (const T*)A,
                                                                       0,
                                                                       lda,
                                                                       stride_A,
                                                                       (T*)B,
                                                                       0,
                                                                       ldb,
                                                                       stride_B,
                                                                       batch_count,
                                                                       optimal_mem,
                                                                       mem_x_temp,
                                                                       mem_x_temp_arr,
                                                                       mem_invA,
                                                                       mem_invA_arr,
                                                                       (const T*)supplied_invA,
                                                                       supplied_invA_size,
                                                                       0,
                                                                       stride_invA);

        return status != rocblas_status_success ? status : perf_status;
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsm_strided_batched(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const float*      alpha,
                                             const float*      A,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_A,
                                             float*            B,
                                             rocblas_int       ldb,
                                             rocblas_stride    stride_B,
                                             rocblas_int       batch_count)
try
{
    return rocblas_trsm_strided_batched_ex_impl<STRSM_BLOCK>(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             m,
                                                             n,
                                                             alpha,
                                                             A,
                                                             lda,
                                                             stride_A,
                                                             B,
                                                             ldb,
                                                             stride_B,
                                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrsm_strided_batched(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const double*     alpha,
                                             const double*     A,
                                             rocblas_int       lda,
                                             rocblas_stride    stride_A,
                                             double*           B,
                                             rocblas_int       ldb,
                                             rocblas_stride    stride_B,
                                             rocblas_int       batch_count)
try
{
    return rocblas_trsm_strided_batched_ex_impl<DTRSM_BLOCK>(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             m,
                                                             n,
                                                             alpha,
                                                             A,
                                                             lda,
                                                             stride_A,
                                                             B,
                                                             ldb,
                                                             stride_B,
                                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrsm_strided_batched(rocblas_handle               handle,
                                             rocblas_side                 side,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            transA,
                                             rocblas_diagonal             diag,
                                             rocblas_int                  m,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_stride               stride_A,
                                             rocblas_float_complex*       B,
                                             rocblas_int                  ldb,
                                             rocblas_stride               stride_B,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_trsm_strided_batched_ex_impl<STRSM_BLOCK>(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             m,
                                                             n,
                                                             alpha,
                                                             A,
                                                             lda,
                                                             stride_A,
                                                             B,
                                                             ldb,
                                                             stride_B,
                                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrsm_strided_batched(rocblas_handle                handle,
                                             rocblas_side                  side,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             transA,
                                             rocblas_diagonal              diag,
                                             rocblas_int                   m,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_stride                stride_A,
                                             rocblas_double_complex*       B,
                                             rocblas_int                   ldb,
                                             rocblas_stride                stride_B,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_trsm_strided_batched_ex_impl<DTRSM_BLOCK>(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             m,
                                                             n,
                                                             alpha,
                                                             A,
                                                             lda,
                                                             stride_A,
                                                             B,
                                                             ldb,
                                                             stride_B,
                                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_trsm_strided_batched_ex(rocblas_handle    handle,
                                               rocblas_side      side,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_diagonal  diag,
                                               rocblas_int       m,
                                               rocblas_int       n,
                                               const void*       alpha,
                                               const void*       A,
                                               rocblas_int       lda,
                                               rocblas_stride    stride_A,
                                               void*             B,
                                               rocblas_int       ldb,
                                               rocblas_stride    stride_B,
                                               rocblas_int       batch_count,
                                               const void*       invA,
                                               rocblas_int       invA_size,
                                               rocblas_stride    stride_invA,
                                               rocblas_datatype  compute_type)
try
{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsm_strided_batched_ex_impl<DTRSM_BLOCK>(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 m,
                                                                 n,
                                                                 static_cast<const double*>(alpha),
                                                                 static_cast<const double*>(A),
                                                                 lda,
                                                                 stride_A,
                                                                 static_cast<double*>(B),
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 static_cast<const double*>(invA),
                                                                 invA_size,
                                                                 stride_invA);

    case rocblas_datatype_f32_r:
        return rocblas_trsm_strided_batched_ex_impl<STRSM_BLOCK>(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 m,
                                                                 n,
                                                                 static_cast<const float*>(alpha),
                                                                 static_cast<const float*>(A),
                                                                 lda,
                                                                 stride_A,
                                                                 static_cast<float*>(B),
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 static_cast<const float*>(invA),
                                                                 invA_size,
                                                                 stride_invA);

    case rocblas_datatype_f32_c:
        return rocblas_trsm_strided_batched_ex_impl<STRSM_BLOCK>(
            handle,
            side,
            uplo,
            transA,
            diag,
            m,
            n,
            static_cast<const rocblas_float_complex*>(alpha),
            static_cast<const rocblas_float_complex*>(A),
            lda,
            stride_A,
            static_cast<rocblas_float_complex*>(B),
            ldb,
            stride_B,
            batch_count,
            static_cast<const rocblas_float_complex*>(invA),
            invA_size,
            stride_invA);

    case rocblas_datatype_f64_c:
        return rocblas_trsm_strided_batched_ex_impl<STRSM_BLOCK>(
            handle,
            side,
            uplo,
            transA,
            diag,
            m,
            n,
            static_cast<const rocblas_double_complex*>(alpha),
            static_cast<const rocblas_double_complex*>(A),
            lda,
            stride_A,
            static_cast<rocblas_double_complex*>(B),
            ldb,
            stride_B,
            batch_count,
            static_cast<const rocblas_double_complex*>(invA),
            invA_size,
            stride_invA);

    default:
        return rocblas_status_not_implemented;
    }
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
