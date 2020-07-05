/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "gemm.hpp"
#include "logging.h"

namespace
{
    template <typename>
    constexpr char rocblas_gemm_strided_batched_name[] = "unknown";

    template <>
    constexpr char rocblas_gemm_strided_batched_name<rocblas_half>[]
        = "rocblas_hgemm_strided_batched";

    template <>
    constexpr char rocblas_gemm_strided_batched_name<float>[] = "rocblas_sgemm_strided_batched";

    template <>
    constexpr char rocblas_gemm_strided_batched_name<double>[] = "rocblas_dgemm_strided_batched";

    template <>
    constexpr char rocblas_gemm_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_cgemm_strided_batched";

    template <>
    constexpr char rocblas_gemm_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zgemm_strided_batched";

    /*******************************************************************************
    * Strided / Batched GEMM implementation
    ******************************************************************************/
    template <typename T>
    auto rocblas_gemm_strided_batched_impl(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T*          A,
                                           rocblas_int       ld_a,
                                           rocblas_stride    stride_a,
                                           const T*          B,
                                           rocblas_int       ld_b,
                                           rocblas_stride    stride_b,
                                           const T*          beta,
                                           T*                C,
                                           rocblas_int       ld_c,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count)

    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device
        T alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(
            copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        auto layer_mode = handle->layer_mode;

        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto trans_a_letter = rocblas_transpose_letter(trans_a);
            auto trans_b_letter = rocblas_transpose_letter(trans_b);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          log_trace_scalar_value(alpha),
                          A,
                          ld_a,
                          stride_a,
                          B,
                          ld_b,
                          stride_b,
                          log_trace_scalar_value(beta),
                          C,
                          ld_c,
                          stride_c,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                log_bench(handle,
                          "./rocblas-bench -f gemm_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          trans_a_letter,
                          "--transposeB",
                          trans_b_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "-k",
                          k,
                          LOG_BENCH_SCALAR_VALUE(alpha),
                          "--lda",
                          ld_a,
                          "--stride_a",
                          stride_a,
                          "--ldb",
                          ld_b,
                          "--stride_b",
                          stride_b,
                          LOG_BENCH_SCALAR_VALUE(beta),
                          "--ldc",
                          ld_c,
                          "--stride_c",
                          stride_c,
                          "--batch_count",
                          batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_gemm_strided_batched_name<T>,
                            "transA",
                            trans_a_letter,
                            "transB",
                            trans_b_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "K",
                            k,
                            "alpha",
                            value_category(*alpha),
                            "lda",
                            ld_a,
                            "stride_a",
                            stride_a,
                            "ldb",
                            ld_b,
                            "stride_b",
                            stride_b,
                            "beta",
                            value_category(*beta),
                            "ldc",
                            ld_c,
                            "stride_c",
                            stride_c,
                            "batch_count",
                            batch_count);
            }
        }

        auto validArgs = validateArgs(
            handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c, batch_count);

        if(validArgs != rocblas_status_continue)
            return validArgs;

        return rocblas_gemm_template<false>(handle,
                                            trans_a,
                                            trans_b,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            0,
                                            ld_a,
                                            stride_a,
                                            B,
                                            0,
                                            ld_b,
                                            stride_b,
                                            beta,
                                            C,
                                            0,
                                            ld_c,
                                            stride_c,
                                            batch_count);
    }
}

extern "C" {

/*******************************************************************************
 * Strided_Batched GEMM APIs
 ******************************************************************************/

rocblas_status rocblas_hgemm_strided_batched(rocblas_handle      handle,
                                             rocblas_operation   trans_a,
                                             rocblas_operation   trans_b,
                                             rocblas_int         m,
                                             rocblas_int         n,
                                             rocblas_int         k,
                                             const rocblas_half* alpha,
                                             const rocblas_half* A,
                                             rocblas_int         ld_a,
                                             rocblas_stride      stride_a,
                                             const rocblas_half* B,
                                             rocblas_int         ld_b,
                                             rocblas_stride      stride_b,
                                             const rocblas_half* beta,
                                             rocblas_half*       C,
                                             rocblas_int         ld_c,
                                             rocblas_stride      stride_c,
                                             rocblas_int         batch_count)
try
{
    return rocblas_gemm_strided_batched_impl(handle,
                                             trans_a,
                                             trans_b,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             A,
                                             ld_a,
                                             stride_a,
                                             B,
                                             ld_b,
                                             stride_b,
                                             beta,
                                             C,
                                             ld_c,
                                             stride_c,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}
rocblas_status rocblas_sgemm_strided_batched(rocblas_handle    handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const float*      alpha,
                                             const float*      A,
                                             rocblas_int       ld_a,
                                             rocblas_stride    stride_a,
                                             const float*      B,
                                             rocblas_int       ld_b,
                                             rocblas_stride    stride_b,
                                             const float*      beta,
                                             float*            C,
                                             rocblas_int       ld_c,
                                             rocblas_stride    stride_c,
                                             rocblas_int       batch_count)
try
{
    return rocblas_gemm_strided_batched_impl(handle,
                                             trans_a,
                                             trans_b,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             A,
                                             ld_a,
                                             stride_a,
                                             B,
                                             ld_b,
                                             stride_b,
                                             beta,
                                             C,
                                             ld_c,
                                             stride_c,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dgemm_strided_batched(rocblas_handle    handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const double*     alpha,
                                             const double*     A,
                                             rocblas_int       ld_a,
                                             rocblas_stride    stride_a,
                                             const double*     B,
                                             rocblas_int       ld_b,
                                             rocblas_stride    stride_b,
                                             const double*     beta,
                                             double*           C,
                                             rocblas_int       ld_c,
                                             rocblas_stride    stride_c,
                                             rocblas_int       batch_count)
try
{
    return rocblas_gemm_strided_batched_impl(handle,
                                             trans_a,
                                             trans_b,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             A,
                                             ld_a,
                                             stride_a,
                                             B,
                                             ld_b,
                                             stride_b,
                                             beta,
                                             C,
                                             ld_c,
                                             stride_c,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cgemm_strided_batched(rocblas_handle               handle,
                                             rocblas_operation            trans_a,
                                             rocblas_operation            trans_b,
                                             rocblas_int                  m,
                                             rocblas_int                  n,
                                             rocblas_int                  k,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  ld_a,
                                             rocblas_stride               stride_a,
                                             const rocblas_float_complex* B,
                                             rocblas_int                  ld_b,
                                             rocblas_stride               stride_b,
                                             const rocblas_float_complex* beta,
                                             rocblas_float_complex*       C,
                                             rocblas_int                  ld_c,
                                             rocblas_stride               stride_c,
                                             rocblas_int                  batch_count)
try
{
    return rocblas_gemm_strided_batched_impl(handle,
                                             trans_a,
                                             trans_b,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             A,
                                             ld_a,
                                             stride_a,
                                             B,
                                             ld_b,
                                             stride_b,
                                             beta,
                                             C,
                                             ld_c,
                                             stride_c,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zgemm_strided_batched(rocblas_handle                handle,
                                             rocblas_operation             trans_a,
                                             rocblas_operation             trans_b,
                                             rocblas_int                   m,
                                             rocblas_int                   n,
                                             rocblas_int                   k,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   ld_a,
                                             rocblas_stride                stride_a,
                                             const rocblas_double_complex* B,
                                             rocblas_int                   ld_b,
                                             rocblas_stride                stride_b,
                                             const rocblas_double_complex* beta,
                                             rocblas_double_complex*       C,
                                             rocblas_int                   ld_c,
                                             rocblas_stride                stride_c,
                                             rocblas_int                   batch_count)
try
{
    return rocblas_gemm_strided_batched_impl(handle,
                                             trans_a,
                                             trans_b,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             A,
                                             ld_a,
                                             stride_a,
                                             B,
                                             ld_b,
                                             stride_b,
                                             beta,
                                             C,
                                             ld_c,
                                             stride_c,
                                             batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Strided Batched GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_strided_batched_kernel_name(rocblas_handle      handle,
                                                         rocblas_operation   trans_a,
                                                         rocblas_operation   trans_b,
                                                         rocblas_int         m,
                                                         rocblas_int         n,
                                                         rocblas_int         k,
                                                         const rocblas_half* alpha,
                                                         const rocblas_half* A,
                                                         rocblas_int         ld_a,
                                                         rocblas_stride      stride_a,
                                                         const rocblas_half* B,
                                                         rocblas_int         ld_b,
                                                         rocblas_stride      stride_b,
                                                         const rocblas_half* beta,
                                                         rocblas_half*       C,
                                                         rocblas_int         ld_c,
                                                         rocblas_stride      stride_c,
                                                         rocblas_int         batch_count)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_sgemm_strided_batched_kernel_name(rocblas_handle    handle,
                                                         rocblas_operation trans_a,
                                                         rocblas_operation trans_b,
                                                         rocblas_int       m,
                                                         rocblas_int       n,
                                                         rocblas_int       k,
                                                         const float*      alpha,
                                                         const float*      A,
                                                         rocblas_int       ld_a,
                                                         rocblas_stride    stride_a,
                                                         const float*      B,
                                                         rocblas_int       ld_b,
                                                         rocblas_stride    stride_b,
                                                         const float*      beta,
                                                         float*            C,
                                                         rocblas_int       ld_c,
                                                         rocblas_stride    stride_c,
                                                         rocblas_int       batch_count)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_dgemm_strided_batched_kernel_name(rocblas_handle    handle,
                                                         rocblas_operation trans_a,
                                                         rocblas_operation trans_b,
                                                         rocblas_int       m,
                                                         rocblas_int       n,
                                                         rocblas_int       k,
                                                         const double*     alpha,
                                                         const double*     A,
                                                         rocblas_int       ld_a,
                                                         rocblas_stride    stride_a,
                                                         const double*     B,
                                                         rocblas_int       ld_b,
                                                         rocblas_stride    stride_b,
                                                         const double*     beta,
                                                         double*           C,
                                                         rocblas_int       ld_c,
                                                         rocblas_stride    stride_c,
                                                         rocblas_int       batch_count)
{
    return rocblas_status_not_implemented;
}

} // extern "C"
