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

namespace {

    template <typename>
    static constexpr char rocblas_gemm_name[] = "unknown";
    template <>
    static constexpr char rocblas_gemm_name<rocblas_half>[] = "rocblas_hgemm";
    template <>
    static constexpr char rocblas_gemm_name<float>[] = "rocblas_sgemm";
    template <>
    static constexpr char rocblas_gemm_name<double>[] = "rocblas_dgemm";
    template <>
    static constexpr char rocblas_gemm_name<rocblas_float_complex>[] = "rocblas_cgemm";
    template <>
    static constexpr char rocblas_gemm_name<rocblas_double_complex>[] = "rocblas_zgemm";

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
        // clang-format off
        // Perform logging
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        if(!alpha || !beta)
            return rocblas_status_invalid_pointer;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                        rocblas_layer_mode_log_profile))
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
                            *alpha,
                            A,
                            ld_a,
                            B,
                            ld_b,
                            *beta,
                            C,
                            ld_c);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::stringstream alphass;
                    alphass << "--alpha " << std::real(*alpha);
                    if (std::imag(*alpha) != 0)
                        alphass << " --alphai " << std::imag(*alpha);

                    std::stringstream betass;
                    betass << "--beta " << std::real(*beta);
                    if (std::imag(*beta) != 0)
                        betass << " --betai " << std::imag(*beta);

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
                            alphass.str(),
                            "--lda",
                            ld_a,
                            "--ldb",
                            ld_b,
                            betass.str(),
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

        rocblas_int b_c = 1;
        if(m == 0 || n == 0 || k == 0 || b_c == 0)
        {
            return rocblas_status_success;
        }

        rocblas_int stride_a;
        rocblas_int stride_b;
        rocblas_int stride_c;
        infer_batch_strides(trans_a, trans_b, m, n, k, ld_a,
                            &stride_a, ld_b, &stride_b, ld_c, &stride_c);

        rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                            m, n, k, alpha,
                                            A, ld_a, stride_a,
                                            B, ld_b, stride_b, beta,
                                            C, ld_c, stride_c, b_c);

        if(validArgs != rocblas_status_success)
            return validArgs;

        return rocblas_gemm_strided_batched_template<T>(handle, trans_a, trans_b, m, n, k, alpha, A, ld_a, stride_a, B, ld_b, stride_b, beta, C, ld_c, stride_c, 1);
    }

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
                                                rocblas_int       stride_a,
                                                const T*          B,
                                                rocblas_int       ld_b,
                                                rocblas_int       stride_b,
                                                const T*          beta,
                                                T*                C,
                                                rocblas_int       ld_c,
                                                rocblas_int       stride_c,
                                                rocblas_int       b_c)
    {
        // clang-format off
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                        rocblas_layer_mode_log_profile))
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
                            *alpha,
                            A,
                            ld_a,
                            B,
                            ld_b,
                            *beta,
                            C,
                            ld_c);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::stringstream alphass;
                    alphass << "--alpha " << std::real(*alpha);
                    if (std::imag(*alpha) != 0)
                        alphass << " --alphai " << std::imag(*alpha);

                    std::stringstream betass;
                    betass << "--beta " << std::real(*beta);
                    if (std::imag(*beta) != 0)
                        betass << " --betai " << std::imag(*beta);

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
                            alphass.str(),
                            "--lda",
                            ld_a,
                            "--ldb",
                            ld_b,
                            betass.str(),
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

        rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                                m, n, k, alpha,
                                                A, ld_a, stride_a,
                                                B, ld_b, stride_b, beta,
                                                C, ld_c, stride_c, b_c);

        if(validArgs != rocblas_status_success)
            return validArgs;

        rocblas_gemm_strided_batched_kernel_name_template<T>(trans_a, trans_b, m, n, k, ld_a, stride_a, ld_b, stride_b, ld_c, stride_c, b_c);

        return validArgs;
    }

}


extern "C" {
/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const rocblas_half *alpha,
                             const rocblas_half *A,
                             rocblas_int ld_a,
                             const rocblas_half *B,
                             rocblas_int ld_b,
                             const rocblas_half *beta,
                             rocblas_half *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<rocblas_half>(handle, trans_a, trans_b,
                                           m, n, k, alpha, A, ld_a,
                                           B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_sgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const float *alpha,
                             const float *A,
                             rocblas_int ld_a,
                             const float *B,
                             rocblas_int ld_b,
                             const float *beta,
                             float *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<float>(handle, trans_a, trans_b,
                                    m, n, k, alpha, A, ld_a,
                                    B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_dgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const double *alpha,
                             const double *A,
                             rocblas_int ld_a,
                             const double *B,
                             rocblas_int ld_b,
                             const double *beta,
                             double *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<double>(handle, trans_a, trans_b,
                                     m, n, k, alpha, A, ld_a,
                                     B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_cgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A,
                             rocblas_int ld_a,
                             const rocblas_float_complex *B,
                             rocblas_int ld_b,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<rocblas_float_complex>(handle, trans_a, trans_b,
                                                    m, n, k, alpha, A, ld_a,
                                                    B, ld_b, beta, C, ld_c);
}


rocblas_status rocblas_zgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A,
                             rocblas_int ld_a,
                             const rocblas_double_complex *B,
                             rocblas_int ld_b,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<rocblas_double_complex>(handle, trans_a, trans_b,
                                                    m, n, k, alpha, A, ld_a,
                                                    B, ld_b, beta, C, ld_c);
}

/*******************************************************************************
 * GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const rocblas_half *alpha,
                                         const rocblas_half *A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const rocblas_half *B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const rocblas_half *beta,
                                         rocblas_half *C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    return rocblas_gemm_kernel_name_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_sgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const float *alpha,
                                         const float *A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const float *B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const float *beta,
                                         float *C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    return rocblas_gemm_kernel_name_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_dgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const double *alpha,
                                         const double *A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const double *B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const double *beta,
                                         double *C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    return rocblas_gemm_kernel_name_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}


}

