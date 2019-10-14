/**************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 ************************************************************************** */
#include "gemm.hpp"
#include "Tensile.h"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"
#include <sys/time.h>

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
    rocblas_status rocblas_gemm_impl(rocblas_handle    handle,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const T*          alpha,
                                     const T*          A,
                                     rocblas_int       ld_a,
                                     const T*          B,
                                     rocblas_int       ld_b,
                                     const T*          beta,
                                     T*                C,
                                     rocblas_int       ld_c)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Perform logging
        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto trans_a_letter = rocblas_transpose_letter(trans_a);
            auto trans_b_letter = rocblas_transpose_letter(trans_b);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemm_name<T>,
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              ld_a,
                              B,
                              ld_b,
                              log_trace_scalar_value(beta),
                              C,
                              ld_c);

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
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              ld_a,
                              "--ldb",
                              ld_b,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--ldc",
                              ld_c);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemm_name<T>,
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              ld_a,
                              B,
                              ld_b,
                              beta,
                              C,
                              ld_c);
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
                            "lda",
                            ld_a,
                            "ldb",
                            ld_b,
                            "ldc",
                            ld_c);
        }

#ifdef USE_TENSILE_HOST

        rocblas_status validArgs = validateArgs(
            handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, 0, B, ld_b, 0, beta, C, ld_c, 0, 1);

        if(validArgs != rocblas_status_success)
            return validArgs;

        T alpha_h;
        T beta_h;
        if(rocblas_pointer_mode_host == handle->pointer_mode)
        {
            alpha_h = *alpha;
            beta_h  = *beta;
        }
        else
        {
            hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost);
            hipMemcpy(&beta_h, beta, sizeof(T), hipMemcpyDeviceToHost);
        }

        TensileHostCall<T>           hostCall;
        RocblasContractionProblem<T> problem(ContractionProblemType::GEMM,
                                             trans_a,
                                             trans_b,
                                             m,
                                             n,
                                             k,
                                             alpha_h,
                                             A,
                                             ld_a,
                                             B,
                                             ld_b,
                                             beta_h,
                                             C,
                                             ld_c);

        return callTensileContraction(&problem, handle->host);

#else

        rocblas_status validArgs = validateArgs(
            handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, 0, B, ld_b, 0, beta, C, ld_c, 0, 1);

        if(validArgs != rocblas_status_success)
            return validArgs;

        return rocblas_gemm_template<false, false>(handle,
                                                   trans_a,
                                                   trans_b,
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   0,
                                                   A,
                                                   0,
                                                   ld_a,
                                                   0,
                                                   B,
                                                   0,
                                                   ld_b,
                                                   0,
                                                   beta,
                                                   0,
                                                   C,
                                                   0,
                                                   ld_c,
                                                   0,
                                                   1);
#endif
    }

#ifndef USE_TENSILE_HOST
    template <typename T>
    rocblas_status rocblas_gemm_kernel_name_impl(rocblas_handle    handle,
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
                                                 rocblas_int       b_c)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto trans_a_letter = rocblas_transpose_letter(trans_a);
            auto trans_b_letter = rocblas_transpose_letter(trans_b);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemm_name<T>,
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              ld_a,
                              B,
                              ld_b,
                              log_trace_scalar_value(beta),
                              C,
                              ld_c);

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
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              ld_a,
                              "--ldb",
                              ld_b,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--ldc",
                              ld_c);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemm_name<T>,
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              ld_a,
                              B,
                              ld_b,
                              beta,
                              C,
                              ld_c);
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
                            "lda",
                            ld_a,
                            "ldb",
                            ld_b,
                            "ldc",
                            ld_c);
        }

        rocblas_status validArgs = validateArgs(handle,
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
                                                b_c);

        if(validArgs != rocblas_status_success)
            return validArgs;

        rocblas_gemm_kernel_name_template<false, T>(
            trans_a, trans_b, m, n, k, ld_a, stride_a, ld_b, stride_b, ld_c, stride_c, b_c);

        return validArgs;
    }
#endif

}

extern "C" {
/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm(rocblas_handle      handle,
                             rocblas_operation   trans_a,
                             rocblas_operation   trans_b,
                             rocblas_int         m,
                             rocblas_int         n,
                             rocblas_int         k,
                             const rocblas_half* alpha,
                             const rocblas_half* A,
                             rocblas_int         ld_a,
                             const rocblas_half* B,
                             rocblas_int         ld_b,
                             const rocblas_half* beta,
                             rocblas_half*       C,
                             rocblas_int         ld_c)
{
    return rocblas_gemm_impl<rocblas_half>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_sgemm(rocblas_handle    handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const float*      alpha,
                             const float*      A,
                             rocblas_int       ld_a,
                             const float*      B,
                             rocblas_int       ld_b,
                             const float*      beta,
                             float*            C,
                             rocblas_int       ld_c)
{
    return rocblas_gemm_impl<float>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_dgemm(rocblas_handle    handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const double*     alpha,
                             const double*     A,
                             rocblas_int       ld_a,
                             const double*     B,
                             rocblas_int       ld_b,
                             const double*     beta,
                             double*           C,
                             rocblas_int       ld_c)
{
    return rocblas_gemm_impl<double>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_cgemm(rocblas_handle               handle,
                             rocblas_operation            trans_a,
                             rocblas_operation            trans_b,
                             rocblas_int                  m,
                             rocblas_int                  n,
                             rocblas_int                  k,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* A,
                             rocblas_int                  ld_a,
                             const rocblas_float_complex* B,
                             rocblas_int                  ld_b,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       C,
                             rocblas_int                  ld_c)
{
    return rocblas_gemm_impl<rocblas_float_complex>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_zgemm(rocblas_handle                handle,
                             rocblas_operation             trans_a,
                             rocblas_operation             trans_b,
                             rocblas_int                   m,
                             rocblas_int                   n,
                             rocblas_int                   k,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* A,
                             rocblas_int                   ld_a,
                             const rocblas_double_complex* B,
                             rocblas_int                   ld_b,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       C,
                             rocblas_int                   ld_c)
{
    return rocblas_gemm_impl<rocblas_double_complex>(
        handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, B, ld_b, beta, C, ld_c);
}

#ifndef USE_TENSILE_HOST

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
                                         rocblas_int         ld_a,
                                         rocblas_stride      stride_a,
                                         const rocblas_half* B,
                                         rocblas_int         ld_b,
                                         rocblas_stride      stride_b,
                                         const rocblas_half* beta,
                                         rocblas_half*       C,
                                         rocblas_int         ld_c,
                                         rocblas_stride      stride_c,
                                         rocblas_int         b_c)
{
    return rocblas_gemm_kernel_name_impl<rocblas_half>(handle,
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
                                                       b_c);
}

rocblas_status rocblas_sgemm_kernel_name(rocblas_handle    handle,
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
                                         rocblas_int       b_c)
{
    return rocblas_gemm_kernel_name_impl<float>(handle,
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
                                                b_c);
}

rocblas_status rocblas_dgemm_kernel_name(rocblas_handle    handle,
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
                                         rocblas_int       b_c)
{
    return rocblas_gemm_kernel_name_impl<double>(handle,
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
                                                 b_c);
}
#endif
}
