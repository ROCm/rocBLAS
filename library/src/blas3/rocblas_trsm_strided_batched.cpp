/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_trsm.hpp"
#include "trtri_trsm.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

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

        /////////////
        // LOGGING //
        /////////////
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
                              "--batch",
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
                            "batch",
                            batch_count);
            }
        }

        /////////////////////
        // ARGUMENT CHECKS //
        /////////////////////
        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!alpha || !A)
            return rocblas_status_invalid_pointer;
        if(!B)
            return rocblas_status_invalid_pointer;

        // A is of size lda*k
        rocblas_int k = side == rocblas_side_left ? m : n;

        if(batch_count < 0)
            return rocblas_status_invalid_size;
        // TODO: Should these return invalid_size even if batch_count == 0?
        if(lda < k && batch_count > 0)
            return rocblas_status_invalid_size;
        if(ldb < m && batch_count > 0)
            return rocblas_status_invalid_size;
        if((m < 0 || n < 0) && batch_count > 0)
            return rocblas_status_invalid_size;

        //////////////////////
        // MEMORY MANAGEMENT//
        //////////////////////
        // quick return if possible.
        // return status_size_unchanged if device memory size query
        if(!m || !n)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        rocblas_status perf_status = rocblas_status_success;
        if(supplied_invA && supplied_invA_size / BLOCK < k)
        {
            static int msg = fputs("WARNING: TRSM invA_size argument is too small; invA argument "
                                   "is being ignored; TRSM performance is degraded\n",
                                   stderr);
            perf_status    = rocblas_status_perf_degraded;
            supplied_invA  = nullptr;
        }

        void*          mem_x_temp;
        void*          mem_x_temp_arr;
        void*          mem_invA;
        void*          mem_invA_arr;
        bool           optimal_mem;
        rocblas_status status = rocblas_trsm_template_mem<BLOCK, false, T>(handle,
                                                                           side,
                                                                           m,
                                                                           n,
                                                                           batch_count,
                                                                           &mem_x_temp,
                                                                           &mem_x_temp_arr,
                                                                           &mem_invA,
                                                                           &mem_invA_arr,
                                                                           &optimal_mem,
                                                                           supplied_invA,
                                                                           supplied_invA_size);

        rocblas_status status2 = rocblas_trsm_template<BLOCK, false, T>(handle,
                                                                        side,
                                                                        uplo,
                                                                        transA,
                                                                        diag,
                                                                        m,
                                                                        n,
                                                                        alpha,
                                                                        0,
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

        return (perf_status != rocblas_status_success)
                   ? perf_status
                   : (status2 == rocblas_status_success) ? status : status2;
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

    default:
        return rocblas_status_not_implemented;
    }
}

} // extern "C"
