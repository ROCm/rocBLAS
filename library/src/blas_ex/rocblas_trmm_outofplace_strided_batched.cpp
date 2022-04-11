/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "../blas3/rocblas_trmm.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

#define Strmm_outofplace_strided_batched_NB 32
#define Dtrmm_outofplace_strided_batched_NB 32
#define Ctrmm_outofplace_strided_batched_NB 32
#define Ztrmm_outofplace_strided_batched_NB 32

namespace
{
    template <typename>
    constexpr char rocblas_trmm_outofplace_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_trmm_outofplace_strided_batched_name<float>[]
        = "rocblas_strmm_outofplace_strided_batched";
    template <>
    constexpr char rocblas_trmm_outofplace_strided_batched_name<double>[]
        = "rocblas_dtrmm_outofplace_strided_batched";
    template <>
    constexpr char rocblas_trmm_outofplace_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_ctrmm_outofplace_strided_batched";
    template <>
    constexpr char rocblas_trmm_outofplace_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_ztrmm_outofplace_strided_batched";

    template <int NB, typename T>
    rocblas_status rocblas_trmm_outofplace_strided_batched_impl(rocblas_handle    handle,
                                                                rocblas_side      side,
                                                                rocblas_fill      uplo,
                                                                rocblas_operation transa,
                                                                rocblas_diagonal  diag,
                                                                rocblas_int       m,
                                                                rocblas_int       n,
                                                                const T*          alpha,
                                                                const T*          a,
                                                                rocblas_int       lda,
                                                                rocblas_stride    stride_a,
                                                                const T*          b,
                                                                rocblas_int       ldb,
                                                                rocblas_stride    stride_b,
                                                                T*                c,
                                                                rocblas_int       ldc,
                                                                rocblas_stride    stride_c,
                                                                rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile)
           && (!handle->is_device_memory_size_query()))
        {
            auto side_letter   = rocblas_side_letter(side);
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transa_letter = rocblas_transpose_letter(transa);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_trmm_outofplace_strided_batched_name<T>,
                          side,
                          uplo,
                          transa,
                          diag,
                          m,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          a,
                          lda,
                          stride_a,
                          b,
                          ldb,
                          stride_b,
                          c,
                          ldc,
                          stride_c,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f trmm_outofplace_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--side",
                          side_letter,
                          "--uplo",
                          uplo_letter,
                          "--transposeA",
                          transa_letter,
                          "--diag",
                          diag_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          LOG_BENCH_SCALAR_VALUE(handle, alpha),
                          "--lda",
                          lda,
                          "--stride_a",
                          stride_a,
                          "--ldb",
                          ldb,
                          "--stride_b",
                          stride_b,
                          "--ldc",
                          ldc,
                          "--stride_c",
                          stride_c,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_trmm_outofplace_strided_batched_name<T>,
                            "side",
                            side_letter,
                            "uplo",
                            uplo_letter,
                            "transa",
                            transa_letter,
                            "diag",
                            diag_letter,
                            "m",
                            m,
                            "n",
                            n,
                            "lda",
                            lda,
                            "stride_a",
                            stride_a,
                            "ldb",
                            ldb,
                            "stride_b",
                            stride_b,
                            "ldc",
                            ldc,
                            "stride_c",
                            stride_c,
                            "batch_count",
                            batch_count);
        }

        rocblas_int nrowa = rocblas_side_left == side ? m : n;

        if(m < 0 || n < 0 || lda < nrowa || ldb < m || ldc < m || batch_count < 0)
            return rocblas_status_invalid_size;

        if(m == 0 || n == 0 || batch_count == 0)
            return rocblas_status_success;

        if(!b || !c || !alpha)
            return rocblas_status_invalid_pointer;

        rocblas_stride offset_a     = 0;
        rocblas_stride offset_b     = 0;
        rocblas_stride offset_c     = 0;
        rocblas_stride stride_alpha = 0;

        if(rocblas_pointer_mode_host == handle->pointer_mode && 0 == *alpha)
        {
            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(set_matrix_zero_if_alpha_zero_template(
                handle, m, n, alpha, 0, c, ldc, stride_c, batch_count));
            return rocblas_status_success;
        }
        else if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // set matrix to zero and continue calculation. This will give
            // the same functionality as Legacy BLAS. alpha is on device and
            // it should not be copied from device to host because this is
            // an asynchronous function and the copy would make it synchronous.
            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(set_matrix_zero_if_alpha_zero_template(
                handle, m, n, alpha, 0, c, ldc, stride_c, batch_count));
        }

        if(rocblas_pointer_mode_host == handle->pointer_mode && !a)
            return rocblas_status_invalid_pointer;

        rocblas_int a_col       = rocblas_side_left == side ? m : n;
        bool        i64_indices = (a_col * size_t(lda) > std::numeric_limits<rocblas_int>::max())
                           || (n * size_t(ldb) > std::numeric_limits<rocblas_int>::max())
                           || (n * size_t(ldc) > std::numeric_limits<rocblas_int>::max());

        if(i64_indices)
        {
            rocblas_internal_trmm_template<NB, false, T>(handle,
                                                         side,
                                                         uplo,
                                                         transa,
                                                         diag,
                                                         m,
                                                         n,
                                                         alpha,
                                                         stride_alpha,
                                                         a,
                                                         size_t(offset_a),
                                                         size_t(lda),
                                                         stride_a,
                                                         b,
                                                         size_t(offset_b),
                                                         size_t(ldb),
                                                         stride_b,
                                                         c,
                                                         size_t(offset_c),
                                                         size_t(ldc),
                                                         stride_c,
                                                         batch_count);
        }
        else
        {
            rocblas_internal_trmm_template<NB, false, T>(handle,
                                                         side,
                                                         uplo,
                                                         transa,
                                                         diag,
                                                         m,
                                                         n,
                                                         alpha,
                                                         stride_alpha,
                                                         a,
                                                         offset_a,
                                                         lda,
                                                         stride_a,
                                                         b,
                                                         offset_b,
                                                         ldb,
                                                         stride_b,
                                                         c,
                                                         offset_c,
                                                         ldc,
                                                         stride_c,
                                                         batch_count);
        }
        return rocblas_status_success;
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

#define IMPL(routine_name_, T_, STRIDED_BATCHED_NB_)                                           \
    rocblas_status routine_name_(rocblas_handle    handle,                                     \
                                 rocblas_side      side,                                       \
                                 rocblas_fill      uplo,                                       \
                                 rocblas_operation transa,                                     \
                                 rocblas_diagonal  diag,                                       \
                                 rocblas_int       m,                                          \
                                 rocblas_int       n,                                          \
                                 const T_*         alpha,                                      \
                                 const T_*         a,                                          \
                                 rocblas_int       lda,                                        \
                                 rocblas_stride    stride_a,                                   \
                                 const T_*         b,                                          \
                                 rocblas_int       ldb,                                        \
                                 rocblas_stride    stride_b,                                   \
                                 T_*               c,                                          \
                                 rocblas_int       ldc,                                        \
                                 rocblas_stride    stride_c,                                   \
                                 rocblas_int       batch_count)                                \
    try                                                                                        \
    {                                                                                          \
        return rocblas_trmm_outofplace_strided_batched_impl<STRIDED_BATCHED_NB_>(handle,       \
                                                                                 side,         \
                                                                                 uplo,         \
                                                                                 transa,       \
                                                                                 diag,         \
                                                                                 m,            \
                                                                                 n,            \
                                                                                 alpha,        \
                                                                                 a,            \
                                                                                 lda,          \
                                                                                 stride_a,     \
                                                                                 b,            \
                                                                                 ldb,          \
                                                                                 stride_b,     \
                                                                                 c,            \
                                                                                 ldc,          \
                                                                                 stride_c,     \
                                                                                 batch_count); \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        return exception_to_rocblas_status();                                                  \
    }

IMPL(rocblas_strmm_outofplace_strided_batched, float, Strmm_outofplace_strided_batched_NB);
IMPL(rocblas_dtrmm_outofplace_strided_batched, double, Dtrmm_outofplace_strided_batched_NB);
IMPL(rocblas_ctrmm_outofplace_strided_batched,
     rocblas_float_complex,
     Ctrmm_outofplace_strided_batched_NB);
IMPL(rocblas_ztrmm_outofplace_strided_batched,
     rocblas_double_complex,
     Ztrmm_outofplace_strided_batched_NB);

#undef IMPL

} // extern "C"

/* ============================================================================================ */
