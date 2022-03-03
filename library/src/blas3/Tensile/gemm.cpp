/**************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 ************************************************************************** */

#include "gemm.hpp"
#include "logging.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_gemm_name[] = "unknown";
    template <>
    constexpr char rocblas_gemm_name<rocblas_half>[] = "rocblas_hgemm";
    template <>
    constexpr char rocblas_gemm_name<float>[] = "rocblas_sgemm";
    template <>
    constexpr char rocblas_gemm_name<double>[] = "rocblas_dgemm";
    template <>
    constexpr char rocblas_gemm_name<rocblas_float_complex>[] = "rocblas_cgemm";
    template <>
    constexpr char rocblas_gemm_name<rocblas_double_complex>[] = "rocblas_zgemm";

    /*******************************************************************************
    * GEMM implementation
    ******************************************************************************/
    template <typename T>
    auto rocblas_gemm_impl(rocblas_handle    handle,
                           rocblas_operation trans_a,
                           rocblas_operation trans_b,
                           rocblas_int       m,
                           rocblas_int       n,
                           rocblas_int       k,
                           const T*          alpha,
                           const T*          A,
                           rocblas_int       lda,
                           const T*          B,
                           rocblas_int       ldb,
                           const T*          beta,
                           T*                C,
                           rocblas_int       ldc)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device
        T alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(
            copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Perform logging
        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto trans_a_letter = rocblas_transpose_letter(trans_a);
            auto trans_b_letter = rocblas_transpose_letter(trans_b);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          A,
                          lda,
                          B,
                          ldb,
                          LOG_TRACE_SCALAR_VALUE(handle, beta),
                          C,
                          ldc);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                log_bench(handle,
                          "./rocblas-bench -f gemm -r",
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
                          LOG_BENCH_SCALAR_VALUE(handle, alpha),
                          "--lda",
                          lda,
                          "--ldb",
                          ldb,
                          LOG_BENCH_SCALAR_VALUE(handle, beta),
                          "--ldc",
                          ldc);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_gemm_name<T>,
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
                            lda,
                            "ldb",
                            ldb,
                            "beta",
                            value_category(*beta),
                            "ldc",
                            ldc);
        }

        auto validArgs
            = validateArgs(handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        if(validArgs != rocblas_status_continue)
            return validArgs;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status gemm_check_numerics_status
                = rocblas_gemm_check_numerics(rocblas_gemm_name<T>,
                                              handle,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              A,
                                              lda,
                                              0,
                                              B,
                                              ldb,
                                              0,
                                              C,
                                              ldc,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(gemm_check_numerics_status != rocblas_status_success)
                return gemm_check_numerics_status;
        }

        rocblas_status status = rocblas_status_success;

        rocblas_int a_n2 = rocblas_operation_none == trans_a ? k : m;
        rocblas_int b_n2 = rocblas_operation_none == trans_b ? n : k;

        status = rocblas_internal_gemm_template<false>(handle,
                                                       trans_a,
                                                       trans_b,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       0,
                                                       lda,
                                                       0,
                                                       B,
                                                       0,
                                                       ldb,
                                                       0,
                                                       beta,
                                                       C,
                                                       0,
                                                       ldc,
                                                       0,
                                                       1);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status gemm_check_numerics_status
                = rocblas_gemm_check_numerics(rocblas_gemm_name<T>,
                                              handle,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              A,
                                              lda,
                                              0,
                                              B,
                                              ldb,
                                              0,
                                              C,
                                              ldc,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(gemm_check_numerics_status != rocblas_status_success)
                return gemm_check_numerics_status;
        }
        return status;
    }
}

/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
extern "C" {

rocblas_status rocblas_hgemm(rocblas_handle      handle,
                             rocblas_operation   trans_a,
                             rocblas_operation   trans_b,
                             rocblas_int         m,
                             rocblas_int         n,
                             rocblas_int         k,
                             const rocblas_half* alpha,
                             const rocblas_half* A,
                             rocblas_int         lda,
                             const rocblas_half* B,
                             rocblas_int         ldb,
                             const rocblas_half* beta,
                             rocblas_half*       C,
                             rocblas_int         ldc)
try
{
    return rocblas_gemm_impl(
        handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_sgemm(rocblas_handle    handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const float*      alpha,
                             const float*      A,
                             rocblas_int       lda,
                             const float*      B,
                             rocblas_int       ldb,
                             const float*      beta,
                             float*            C,
                             rocblas_int       ldc)
try
{
    return rocblas_gemm_impl(
        handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dgemm(rocblas_handle    handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const double*     alpha,
                             const double*     A,
                             rocblas_int       lda,
                             const double*     B,
                             rocblas_int       ldb,
                             const double*     beta,
                             double*           C,
                             rocblas_int       ldc)
try
{
    return rocblas_gemm_impl(
        handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cgemm(rocblas_handle               handle,
                             rocblas_operation            trans_a,
                             rocblas_operation            trans_b,
                             rocblas_int                  m,
                             rocblas_int                  n,
                             rocblas_int                  k,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* A,
                             rocblas_int                  lda,
                             const rocblas_float_complex* B,
                             rocblas_int                  ldb,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       C,
                             rocblas_int                  ldc)
try
{
    return rocblas_gemm_impl(
        handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zgemm(rocblas_handle                handle,
                             rocblas_operation             trans_a,
                             rocblas_operation             trans_b,
                             rocblas_int                   m,
                             rocblas_int                   n,
                             rocblas_int                   k,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* A,
                             rocblas_int                   lda,
                             const rocblas_double_complex* B,
                             rocblas_int                   ldb,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       C,
                             rocblas_int                   ldc)
try
{
    return rocblas_gemm_impl(
        handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_kernel_name(rocblas_handle      handle,
                                         rocblas_operation   trans_a,
                                         rocblas_operation   trans_b,
                                         rocblas_int         m,
                                         rocblas_int         n,
                                         rocblas_int         k,
                                         const rocblas_half* alpha,
                                         const rocblas_half* A,
                                         rocblas_int         lda,
                                         rocblas_stride      stride_a,
                                         const rocblas_half* B,
                                         rocblas_int         ldb,
                                         rocblas_stride      stride_b,
                                         const rocblas_half* beta,
                                         rocblas_half*       C,
                                         rocblas_int         ldc,
                                         rocblas_stride      stride_c,
                                         rocblas_int         b_c)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_sgemm_kernel_name(rocblas_handle    handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         rocblas_int       k,
                                         const float*      alpha,
                                         const float*      A,
                                         rocblas_int       lda,
                                         rocblas_stride    stride_a,
                                         const float*      B,
                                         rocblas_int       ldb,
                                         rocblas_stride    stride_b,
                                         const float*      beta,
                                         float*            C,
                                         rocblas_int       ldc,
                                         rocblas_stride    stride_c,
                                         rocblas_int       b_c)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_dgemm_kernel_name(rocblas_handle    handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         rocblas_int       k,
                                         const double*     alpha,
                                         const double*     A,
                                         rocblas_int       lda,
                                         rocblas_stride    stride_a,
                                         const double*     B,
                                         rocblas_int       ldb,
                                         rocblas_stride    stride_b,
                                         const double*     beta,
                                         double*           C,
                                         rocblas_int       ldc,
                                         rocblas_stride    stride_c,
                                         rocblas_int       b_c)
{
    return rocblas_status_not_implemented;
}

} // extern "C"
