/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_trmm.hpp"
#include "utility.hpp"

#define STRMM_BATCHED_STOPPING_NB 32
#define DTRMM_BATCHED_STOPPING_NB 32
#define CTRMM_BATCHED_STOPPING_NB 16
#define ZTRMM_BATCHED_STOPPING_NB 16

namespace
{
    template <typename>
    constexpr char rocblas_trmm_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_trmm_batched_name<float>[] = "rocblas_strmm_batched";
    template <>
    constexpr char rocblas_trmm_batched_name<double>[] = "rocblas_dtrmm_batched";
    template <>
    constexpr char rocblas_trmm_batched_name<rocblas_float_complex>[] = "rocblas_ctrmm_batched";
    template <>
    constexpr char rocblas_trmm_batched_name<rocblas_double_complex>[] = "rocblas_ztrmm_batched";

    template <int STOPPING_NB, typename T>
    rocblas_status rocblas_trmm_batched_impl(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transa,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             const T*          alpha,
                                             const T* const    a[],
                                             rocblas_int       lda,
                                             T* const          b[],
                                             rocblas_int       ldb,
                                             rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device. This is because gemm is called and it
        // requires alpha and beta to be on host
        T        alpha_h, beta_h;
        const T* beta = nullptr;
        RETURN_IF_ROCBLAS_ERROR(
            copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, m && n));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

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
                          rocblas_trmm_batched_name<T>,
                          side,
                          uplo,
                          transa,
                          diag,
                          m,
                          n,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          a,
                          lda,
                          b,
                          ldb,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f trmm_batched -r",
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
                          "--ldb",
                          ldb,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_trmm_batched_name<T>,
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
                            "ldb",
                            ldb,
                            "batch_count",
                            batch_count);
        }

        rocblas_int nrowa = rocblas_side_left == side ? m : n;

        if(m < 0 || n < 0 || lda < nrowa || ldb < m || batch_count < 0)
            return rocblas_status_invalid_size;

        if(m == 0 || n == 0 || batch_count == 0)
            return rocblas_status_success;

        if(!b || !alpha)
            return rocblas_status_invalid_pointer;

        rocblas_int    offset_a     = 0;
        rocblas_int    offset_b     = 0;
        rocblas_stride stride_a     = 0;
        rocblas_stride stride_b     = 0;
        rocblas_stride stride_alpha = 0;

        if(rocblas_pointer_mode_host == handle->pointer_mode && 0 == *alpha)
        {
            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(set_matrix_zero_if_alpha_zero_template(
                handle, m, n, alpha, 0, b, offset_b, ldb, stride_b, batch_count));
            return rocblas_status_success;
        }
        else if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // set matrix to zero and continue calculation. This will give
            // the same functionality as Legacy BLAS. alpha is on device and
            // it should not be copied from device to host because this is
            // an asynchronous function and the copy would make it synchronous.
            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(set_matrix_zero_if_alpha_zero_template(
                handle, m, n, alpha, 0, b, offset_b, ldb, stride_b, batch_count));
        }

        if(rocblas_pointer_mode_host == handle->pointer_mode && !a)
            return rocblas_status_invalid_pointer;

        rocblas_int a_col       = rocblas_side_left == side ? m : n;
        bool        i64_indices = (a_col * size_t(lda) > std::numeric_limits<rocblas_int>::max())
                           || (n * size_t(ldb) > std::numeric_limits<rocblas_int>::max());

        if(i64_indices)
        {
            rocblas_internal_trmm_template<STOPPING_NB, true, T>(handle,
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
                                                                 (const T* const*)b,
                                                                 size_t(offset_b),
                                                                 size_t(ldb),
                                                                 stride_b,
                                                                 b, // in place, c == b
                                                                 size_t(offset_b),
                                                                 size_t(ldb),
                                                                 stride_b,
                                                                 batch_count);
        }
        else
        {
            rocblas_internal_trmm_template<STOPPING_NB, true, T>(handle,
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
                                                                 (const T* const*)b,
                                                                 offset_b,
                                                                 ldb,
                                                                 stride_b,
                                                                 b,
                                                                 offset_b,
                                                                 ldb,
                                                                 stride_b,
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

#define IMPL(routine_name_, T_, BATCHED_STOPPING_NB_)                                    \
    rocblas_status routine_name_(rocblas_handle    handle,                               \
                                 rocblas_side      side,                                 \
                                 rocblas_fill      uplo,                                 \
                                 rocblas_operation transa,                               \
                                 rocblas_diagonal  diag,                                 \
                                 rocblas_int       m,                                    \
                                 rocblas_int       n,                                    \
                                 const T_*         alpha,                                \
                                 const T_* const   a[],                                  \
                                 rocblas_int       lda,                                  \
                                 T_* const         b[],                                  \
                                 rocblas_int       ldb,                                  \
                                 rocblas_int       batch_count)                          \
    try                                                                                  \
    {                                                                                    \
        return rocblas_trmm_batched_impl<BATCHED_STOPPING_NB_>(                          \
            handle, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, batch_count); \
    }                                                                                    \
    catch(...)                                                                           \
    {                                                                                    \
        return exception_to_rocblas_status();                                            \
    }

IMPL(rocblas_strmm_batched, float, STRMM_BATCHED_STOPPING_NB);
IMPL(rocblas_dtrmm_batched, double, DTRMM_BATCHED_STOPPING_NB);
IMPL(rocblas_ctrmm_batched, rocblas_float_complex, CTRMM_BATCHED_STOPPING_NB);
IMPL(rocblas_ztrmm_batched, rocblas_double_complex, ZTRMM_BATCHED_STOPPING_NB);

#undef IMPL

} // extern "C"

/* ============================================================================================ */
