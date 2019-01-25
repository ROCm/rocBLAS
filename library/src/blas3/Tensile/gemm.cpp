/**************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 ************************************************************************** */

#include <hip/hip_runtime.h>
#include <sys/time.h>
#include "rocblas.h"
#include "Tensile.h"
#include "gemm.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

/*******************************************************************************
 * Helper enumeration over different transpose combinations
 ******************************************************************************/
typedef enum transpose_mode_ {
    // First letter refers to A, second letter refers to B
    NN,
    NT,
    TN,
    TT
} transpose_mode;

transpose_mode GetTransposeMode(rocblas_operation trans_a, rocblas_operation trans_b)
{
    if(trans_a == rocblas_operation_none)
    {
        if(trans_b == rocblas_operation_none)
            return NN;
        return NT;
    }
    else
    {
        if(trans_b == rocblas_operation_none)
            return TN;
        return TT;
    }
}

/*******************************************************************************
 * Tensile Solution Name (debug only)
 ******************************************************************************/
template <typename T>
const char* tensileGetSolutionName(rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int strideC1,
                                   rocblas_int strideC2,
                                   rocblas_int strideA1,
                                   rocblas_int strideA2,
                                   rocblas_int strideB1,
                                   rocblas_int strideB2,
                                   rocblas_int sizeI,
                                   rocblas_int sizeJ,
                                   rocblas_int sizeK,
                                   rocblas_int sizeL)
{
// This macro condenses all the identical arguments to the various
// tensileGetSolutionName function calls for consistency / brevity
#define TENSILE_ARG_NAMES \
    strideC1, strideC2, strideA1, strideA2, strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL

    transpose_mode transposeMode = GetTransposeMode(trans_a, trans_b);

    if(std::is_same<T, rocblas_half>::value)
    {
        switch(transposeMode)
        {
        case NN: return tensileGetSolutionName_Cijk_Ailk_Bljk_HB(TENSILE_ARG_NAMES);
        case NT: return tensileGetSolutionName_Cijk_Ailk_Bjlk_HB(TENSILE_ARG_NAMES);
        case TN: return tensileGetSolutionName_Cijk_Alik_Bljk_HB(TENSILE_ARG_NAMES);
        case TT: return tensileGetSolutionName_Cijk_Alik_Bjlk_HB(TENSILE_ARG_NAMES);
        }
    }
    else if(std::is_same<T, float>::value)
    {
        switch(transposeMode)
        {
        case NN: return tensileGetSolutionName_Cijk_Ailk_Bljk_SB(TENSILE_ARG_NAMES);
        case NT: return tensileGetSolutionName_Cijk_Ailk_Bjlk_SB(TENSILE_ARG_NAMES);
        case TN: return tensileGetSolutionName_Cijk_Alik_Bljk_SB(TENSILE_ARG_NAMES);
        case TT: return tensileGetSolutionName_Cijk_Alik_Bjlk_SB(TENSILE_ARG_NAMES);
        }
    }
    else if(std::is_same<T, double>::value)
    {
        switch(transposeMode)
        {
        case NN: return tensileGetSolutionName_Cijk_Ailk_Bljk_DB(TENSILE_ARG_NAMES);
        case NT: return tensileGetSolutionName_Cijk_Ailk_Bjlk_DB(TENSILE_ARG_NAMES);
        case TN: return tensileGetSolutionName_Cijk_Alik_Bljk_DB(TENSILE_ARG_NAMES);
        case TT: return tensileGetSolutionName_Cijk_Alik_Bjlk_DB(TENSILE_ARG_NAMES);
        }
    }
    return "";

#undef TENSILE_ARG_NAMES
}

/*******************************************************************************
 * Tensile Function call
 ******************************************************************************/
template <typename T>
hipError_t callTensile(const T* alpha,
                       const T* beta,
                       const T* A,
                       const T* B,
                       T* C,
                       rocblas_operation trans_a,
                       rocblas_operation trans_b,
                       rocblas_int strideC1,
                       rocblas_int strideC2,
                       rocblas_int strideA1,
                       rocblas_int strideA2,
                       rocblas_int strideB1,
                       rocblas_int strideB2,
                       rocblas_int sizeI,
                       rocblas_int sizeJ,
                       rocblas_int sizeK,
                       rocblas_int sizeL,
                       rocblas_handle handle)
{
#ifndef NDEBUG
    std::cout << "Solution Name: " << tensileGetSolutionName<T>(trans_a,
                                                                trans_b,
                                                                strideC1,
                                                                strideC2,
                                                                strideA1,
                                                                strideA2,
                                                                strideB1,
                                                                strideB2,
                                                                sizeI,
                                                                sizeJ,
                                                                sizeK,
                                                                sizeL)
              << std::endl;
#endif

    // Collect alpha / beta (either from host or device)
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

// Helper macros for function call brevity
#define TENSILE_ARGS(T)                                                                    \
    reinterpret_cast<T*>(C), reinterpret_cast<const T*>(C), reinterpret_cast<const T*>(A), \
        reinterpret_cast<const T*>(B), *reinterpret_cast<T*>(&alpha_h),                    \
        *reinterpret_cast<T*>(&beta_h), strideC1, strideC2, strideA1, strideA2, strideB1,  \
        strideB2, sizeI, sizeJ, sizeK, sizeL, handle->rocblas_stream, 0, nullptr, nullptr

    hipError_t status;
    transpose_mode transposeMode = GetTransposeMode(trans_a, trans_b);
    if(std::is_same<T, rocblas_half>::value)
    {
        switch(transposeMode)
        {
        case NN: status = tensile_Cijk_Ailk_Bljk_HB(TENSILE_ARGS(_Float16)); break;
        case NT: status = tensile_Cijk_Ailk_Bjlk_HB(TENSILE_ARGS(_Float16)); break;
        case TN: status = tensile_Cijk_Alik_Bljk_HB(TENSILE_ARGS(_Float16)); break;
        case TT: status = tensile_Cijk_Alik_Bjlk_HB(TENSILE_ARGS(_Float16)); break;
        }
    }
    else if(std::is_same<T, float>::value)
    {
        switch(transposeMode)
        {
        case NN: status = tensile_Cijk_Ailk_Bljk_SB(TENSILE_ARGS(float)); break;
        case NT: status = tensile_Cijk_Ailk_Bjlk_SB(TENSILE_ARGS(float)); break;
        case TN: status = tensile_Cijk_Alik_Bljk_SB(TENSILE_ARGS(float)); break;
        case TT: status = tensile_Cijk_Alik_Bjlk_SB(TENSILE_ARGS(float)); break;
        }
    }
    else if(std::is_same<T, double>::value)
    {
        switch(transposeMode)
        {
        case NN: status = tensile_Cijk_Ailk_Bljk_DB(TENSILE_ARGS(double)); break;
        case NT: status = tensile_Cijk_Ailk_Bjlk_DB(TENSILE_ARGS(double)); break;
        case TN: status = tensile_Cijk_Alik_Bljk_DB(TENSILE_ARGS(double)); break;
        case TT: status = tensile_Cijk_Alik_Bjlk_DB(TENSILE_ARGS(double)); break;
        }
    }
    else
    {
        std::cerr << "Unsupported input format" << std::endl;
    }

#ifndef NDEBUG
    std::cout << "Return Status: " << status << std::endl;
#endif

    return status;
}

template <typename>
constexpr char rocblas_gemm_name[] = "unknown";
template <>
constexpr char rocblas_gemm_name<rocblas_half>[] = "rocblas_hgemm";
template <>
constexpr char rocblas_gemm_name<float>[] = "rocblas_sgemm";
template <>
constexpr char rocblas_gemm_name<double>[] = "rocblas_dgemm";

/*******************************************************************************
 * GEMM implementation
 ******************************************************************************/
template <typename T>
rocblas_status rocblas_gemm_impl(rocblas_handle handle,
                                 rocblas_operation trans_a,
                                 rocblas_operation trans_b,
                                 rocblas_int m,
                                 rocblas_int n,
                                 rocblas_int k,
                                 const T* alpha,
                                 const T* A,
                                 rocblas_int ld_a,
                                 const T* B,
                                 rocblas_int ld_b,
                                 const T* beta,
                                 T* C,
                                 rocblas_int ld_c)
{
    // clang-format off
    // Perform logging
    if(!handle)
        return rocblas_status_invalid_handle;
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
                          "--alpha",
                          *alpha,
                          "--lda",
                          ld_a,
                          "--ldb",
                          ld_b,
                          "--beta",
                          *beta,
                          "--ldc",
                          ld_c);
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

    unsigned int strideC1 = static_cast<unsigned int>(ld_c);
    unsigned int strideC2 = static_cast<unsigned int>(stride_c);
    unsigned int strideA1 = static_cast<unsigned int>(ld_a);
    unsigned int strideA2 = static_cast<unsigned int>(stride_a);
    unsigned int strideB1 = static_cast<unsigned int>(ld_b);
    unsigned int strideB2 = static_cast<unsigned int>(stride_b);
    unsigned int sizeI    = static_cast<unsigned int>(m);
    unsigned int sizeJ    = static_cast<unsigned int>(n);
    unsigned int sizeK    = b_c;
    unsigned int sizeL    = static_cast<unsigned int>(k);

    hipError_t status = callTensile<T>(alpha, beta, A, B, C,
                                       trans_a, trans_b,
                                       strideC1, strideC2,
                                       strideA1, strideA2,
                                       strideB1, strideB2,
                                       sizeI, sizeJ, sizeK, sizeL,
                                       handle);
    // clang-format on

    return get_rocblas_status_for_hip_status(status);
}

template <typename>
constexpr char rocblas_gemm_strided_batched_name[] = "unknown";
template <>
constexpr char rocblas_gemm_strided_batched_name<rocblas_half>[] = "rocblas_hgemm_strided_batched";
template <>
constexpr char rocblas_gemm_strided_batched_name<float>[] = "rocblas_sgemm_strided_batched";
template <>
constexpr char rocblas_gemm_strided_batched_name<double>[] = "rocblas_dgemm_strided_batched";

/*******************************************************************************
 * Strided / Batched GEMM implementation
 ******************************************************************************/
template <typename T>
rocblas_status rocblas_gemm_strided_batched_impl(rocblas_handle handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_int k,
                                                 const T* alpha,
                                                 const T* A,
                                                 rocblas_int ld_a,
                                                 rocblas_int stride_a,
                                                 const T* B,
                                                 rocblas_int ld_b,
                                                 rocblas_int stride_b,
                                                 const T* beta,
                                                 T* C,
                                                 rocblas_int ld_c,
                                                 rocblas_int stride_c,
                                                 rocblas_int b_c)

{
    // clang-format off
    if(!handle)
        return rocblas_status_invalid_handle;

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
                          rocblas_gemm_strided_batched_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          *alpha,
                          A,
                          ld_a,
                          stride_a,
                          B,
                          ld_b,
                          stride_b,
                          *beta,
                          C,
                          ld_c,
                          stride_c,
                          b_c);

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
                          "--alpha",
                          *alpha,
                          "--lda",
                          ld_a,
                          "--stride_a",
                          stride_a,
                          "--ldb",
                          ld_b,
                          "--stride_b",
                          stride_b,
                          "--beta",
                          *beta,
                          "--ldc",
                          ld_c,
                          "--stride_c",
                          stride_c,
                          "--batch",
                          b_c);
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
            {
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
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
                        "lda",
                        ld_a,
                        "stride_a",
                        stride_a,
                        "ldb",
                        ld_b,
                        "stride_b",
                        stride_b,
                        "ldc",
                        ld_c,
                        "stride_c",
                        stride_c,
                        "batch_count",
                        b_c);
        }
    }

    if(m == 0 || n == 0 || k == 0 || b_c == 0)
    {
        return rocblas_status_success;
    }

    rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                            m, n, k, alpha,
                                            A, ld_a, stride_a,
                                            B, ld_b, stride_b, beta,
                                            C, ld_c, stride_c, b_c);

    if(validArgs != rocblas_status_success)
        return validArgs;

    unsigned int strideC1 = static_cast<unsigned int>(ld_c);
    unsigned int strideC2 = static_cast<unsigned int>(stride_c);
    unsigned int strideA1 = static_cast<unsigned int>(ld_a);
    unsigned int strideA2 = static_cast<unsigned int>(stride_a);
    unsigned int strideB1 = static_cast<unsigned int>(ld_b);
    unsigned int strideB2 = static_cast<unsigned int>(stride_b);
    unsigned int sizeI    = static_cast<unsigned int>(m);
    unsigned int sizeJ    = static_cast<unsigned int>(n);
    unsigned int sizeK    = static_cast<unsigned int>(b_c);
    unsigned int sizeL    = static_cast<unsigned int>(k);

    hipError_t status = callTensile<T>(alpha, beta, A, B, C,
                                       trans_a, trans_b,
                                       strideC1, strideC2,
                                       strideA1, strideA2,
                                       strideB1, strideB2,
                                       sizeI, sizeJ, sizeK, sizeL,
                                       handle);
    return get_rocblas_status_for_hip_status(status);

    // clang-format on
}

/*******************************************************************************
 * Batched / Strided GEMM Kernel name implementation
 ******************************************************************************/
template <typename T>
rocblas_status rocblas_gemm_kernel_name_impl(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const T* alpha,
                                             const T* A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const T* B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const T* beta,
                                             T* C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    // clang-format off
    if(!handle)
        return rocblas_status_invalid_handle;

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
                          rocblas_gemm_strided_batched_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          *alpha,
                          A,
                          ld_a,
                          stride_a,
                          B,
                          ld_b,
                          stride_b,
                          *beta,
                          C,
                          ld_c,
                          stride_c,
                          b_c);

            if(layer_mode & rocblas_layer_mode_log_bench)
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
                          "--alpha",
                          *alpha,
                          "--lda",
                          ld_a,
                          "--bsa",
                          stride_a,
                          "--ldb",
                          ld_b,
                          "--bsb",
                          stride_b,
                          "--beta",
                          *beta,
                          "--ldc",
                          ld_c,
                          "--bsc",
                          stride_c,
                          "--batch",
                          b_c);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
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

        if(layer_mode & rocblas_layer_mode_log_profile)
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
                        "lda",
                        ld_a,
                        "stride_a",
                        stride_a,
                        "ldb",
                        ld_b,
                        "stride_b",
                        stride_b,
                        "ldc",
                        ld_c,
                        "stride_c",
                        stride_c,
                        "batch_count",
                        b_c);
    }

    rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                            m, n, k, alpha,
                                            A, ld_a, stride_a,
                                            B, ld_b, stride_b, beta,
                                            C, ld_c, stride_c, b_c);

    if(validArgs != rocblas_status_success)
        return validArgs;

    unsigned int strideC1 = static_cast<unsigned int>(ld_c);
    unsigned int strideC2 = static_cast<unsigned int>(stride_c);
    unsigned int strideA1 = static_cast<unsigned int>(ld_a);
    unsigned int strideA2 = static_cast<unsigned int>(stride_a);
    unsigned int strideB1 = static_cast<unsigned int>(ld_b);
    unsigned int strideB2 = static_cast<unsigned int>(stride_b);
    unsigned int sizeI    = static_cast<unsigned int>(m);
    unsigned int sizeJ    = static_cast<unsigned int>(n);
    unsigned int sizeK    = static_cast<unsigned int>(b_c);
    unsigned int sizeL    = static_cast<unsigned int>(k);

    std::cout << "gemm kernel Name: ";


    const char* solution_name = tensileGetSolutionName<T>(trans_a, trans_b,
                                                          strideC1, strideC2,
                                                          strideA1, strideA2,
                                                          strideB1, strideB2,
                                                          sizeI, sizeJ, sizeK, sizeL);

    std::cout << solution_name << std::endl;

    return validArgs;
}

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


/*******************************************************************************
 * Batched / Strided GEMM APIs
 ******************************************************************************/

rocblas_status rocblas_hgemm_strided_batched(rocblas_handle handle,
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
    return rocblas_gemm_strided_batched_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle,
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
    return rocblas_gemm_strided_batched_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle,
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
    return rocblas_gemm_strided_batched_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

/*******************************************************************************
 * Batched / Strided GEMM Kernel name APIs
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
    rocblas_status status = rocblas_gemm_kernel_name_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
    return status;
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
    rocblas_status status = rocblas_gemm_kernel_name_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
    return status;
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
    rocblas_status status = rocblas_gemm_kernel_name_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
    return status;
}
// clang-format on
