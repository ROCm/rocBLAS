/**************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
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
 * API Args
 ******************************************************************************/
#define ARGS(TYPE)                                                                              \
    rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b, rocblas_int m, \
        rocblas_int n, rocblas_int k, const TYPE *alpha, const TYPE *A, rocblas_int ld_a,       \
        const TYPE *B, rocblas_int ld_b, const TYPE *beta, TYPE *C, rocblas_int ld_c

#define ARGS_BATCHED(TYPE)                                                                      \
    rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b, rocblas_int m, \
        rocblas_int n, rocblas_int k, const TYPE *alpha, const TYPE *A, rocblas_int ld_a,       \
        rocblas_int stride_a, const TYPE *B, rocblas_int ld_b, rocblas_int stride_b,            \
        const TYPE *beta, TYPE *C, rocblas_int ld_c, rocblas_int stride_c, rocblas_int b_c

/*******************************************************************************
 * Preamble Code
 ******************************************************************************/
#define PREAMBLE(TYPE)                                                                 \
                                                                                       \
    if(nullptr != handle)                                                              \
    {                                                                                  \
        if(handle->pointer_mode == rocblas_pointer_mode_host)                          \
        {                                                                              \
            log_trace(handle,                                                          \
                      replaceX<TYPE>("rocblas_Xgemm"),                                 \
                      trans_a,                                                         \
                      trans_b,                                                         \
                      m,                                                               \
                      n,                                                               \
                      k,                                                               \
                      *alpha,                                                          \
                      (const void*&)A,                                                 \
                      ld_a,                                                            \
                      (const void*&)B,                                                 \
                      ld_b,                                                            \
                      *beta,                                                           \
                      (const void*&)C,                                                 \
                      ld_c);                                                           \
                                                                                       \
            std::string trans_a_letter = rocblas_transpose_letter(trans_a);            \
            std::string trans_b_letter = rocblas_transpose_letter(trans_b);            \
                                                                                       \
            log_bench(handle,                                                          \
                      "./rocblas-bench -f gemm -r",                                    \
                      replaceX<TYPE>("X"),                                             \
                      "--transposeA",                                                  \
                      trans_a_letter,                                                  \
                      "--transposeB",                                                  \
                      trans_b_letter,                                                  \
                      "-m",                                                            \
                      m,                                                               \
                      "-n",                                                            \
                      n,                                                               \
                      "-k",                                                            \
                      k,                                                               \
                      "--alpha",                                                       \
                      *alpha,                                                          \
                      "--lda",                                                         \
                      ld_a,                                                            \
                      "--ldb",                                                         \
                      ld_b,                                                            \
                      "--beta",                                                        \
                      *beta,                                                           \
                      "--ldc",                                                         \
                      ld_c);                                                           \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            log_trace(handle,                                                          \
                      replaceX<TYPE>("rocblas_Xgemm"),                                 \
                      trans_a,                                                         \
                      trans_b,                                                         \
                      m,                                                               \
                      n,                                                               \
                      k,                                                               \
                      (const void*&)alpha,                                             \
                      (const void*&)A,                                                 \
                      ld_a,                                                            \
                      (const void*&)B,                                                 \
                      ld_b,                                                            \
                      (const void*&)beta,                                              \
                      (const void*&)C,                                                 \
                      ld_c);                                                           \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    rocblas_int b_c = 1;                                                               \
    rocblas_int stride_c;                                                              \
    rocblas_int stride_a;                                                              \
    rocblas_int stride_b;                                                              \
    infer_batch_strides(                                                               \
        trans_a, trans_b, m, n, k, ld_a, &stride_a, ld_b, &stride_b, ld_c, &stride_c); \
    rocblas_status validArgs = validateArgs(handle,                                    \
                                            trans_a,                                   \
                                            trans_b,                                   \
                                            m,                                         \
                                            n,                                         \
                                            k,                                         \
                                            alpha,                                     \
                                            A,                                         \
                                            ld_a,                                      \
                                            stride_a,                                  \
                                            B,                                         \
                                            ld_b,                                      \
                                            stride_b,                                  \
                                            beta,                                      \
                                            C,                                         \
                                            ld_c,                                      \
                                            stride_c,                                  \
                                            b_c);                                      \
    if(m == 0 || n == 0 || k == 0 || b_c == 0)                                         \
    {                                                                                  \
        return rocblas_status_success;                                                 \
    }                                                                                  \
                                                                                       \
    if(validArgs != rocblas_status_success)                                            \
        return validArgs;                                                              \
                                                                                       \
    unsigned int strideC1 = static_cast<unsigned int>(ld_c);                           \
    unsigned int strideC2 = static_cast<unsigned int>(stride_c);                       \
    unsigned int strideA1 = static_cast<unsigned int>(ld_a);                           \
    unsigned int strideA2 = static_cast<unsigned int>(stride_a);                       \
    unsigned int strideB1 = static_cast<unsigned int>(ld_b);                           \
    unsigned int strideB2 = static_cast<unsigned int>(stride_b);                       \
    unsigned int sizeI    = static_cast<unsigned int>(m);                              \
    unsigned int sizeJ    = static_cast<unsigned int>(n);                              \
    unsigned int sizeK    = b_c;                                                       \
    unsigned int sizeL    = static_cast<unsigned int>(k);

#define PREAMBLE_BATCHED(TYPE)                                          \
    rocblas_status validArgs = validateArgs(handle,                     \
                                            trans_a,                    \
                                            trans_b,                    \
                                            m,                          \
                                            n,                          \
                                            k,                          \
                                            alpha,                      \
                                            A,                          \
                                            ld_a,                       \
                                            stride_a,                   \
                                            B,                          \
                                            ld_b,                       \
                                            stride_b,                   \
                                            beta,                       \
                                            C,                          \
                                            ld_c,                       \
                                            stride_c,                   \
                                            b_c);                       \
                                                                        \
    if(handle->pointer_mode == rocblas_pointer_mode_host)               \
    {                                                                   \
        log_trace(handle,                                               \
                  replaceX<TYPE>("rocblas_Xgemm_strided_batched"),      \
                  trans_a,                                              \
                  trans_b,                                              \
                  m,                                                    \
                  n,                                                    \
                  k,                                                    \
                  *alpha,                                               \
                  (const void*&)A,                                      \
                  ld_a,                                                 \
                  stride_a,                                             \
                  (const void*&)B,                                      \
                  ld_b,                                                 \
                  stride_b,                                             \
                  *beta,                                                \
                  (const void*&)C,                                      \
                  ld_c,                                                 \
                  stride_c,                                             \
                  b_c);                                                 \
                                                                        \
        std::string trans_a_letter = rocblas_transpose_letter(trans_a); \
        std::string trans_b_letter = rocblas_transpose_letter(trans_b); \
                                                                        \
        log_bench(handle,                                               \
                  "./rocblas-bench -f gemm_strided_batched -r",         \
                  replaceX<TYPE>("X"),                                  \
                  "--transposeA",                                       \
                  trans_a_letter,                                       \
                  "--transposeB",                                       \
                  trans_b_letter,                                       \
                  "-m",                                                 \
                  m,                                                    \
                  "-n",                                                 \
                  n,                                                    \
                  "-k",                                                 \
                  k,                                                    \
                  "--alpha",                                            \
                  *alpha,                                               \
                  "--lda",                                              \
                  ld_a,                                                 \
                  "--stride_a",                                         \
                  stride_a,                                             \
                  "--ldb",                                              \
                  ld_b,                                                 \
                  "--stride_b",                                         \
                  stride_b,                                             \
                  "--beta",                                             \
                  *beta,                                                \
                  "--ldc",                                              \
                  ld_c,                                                 \
                  "--stride_c",                                         \
                  stride_c,                                             \
                  "--batch",                                            \
                  b_c);                                                 \
    }                                                                   \
    else                                                                \
    {                                                                   \
        log_trace(handle,                                               \
                  replaceX<TYPE>("rocblas_Xgemm_strided_batched"),      \
                  trans_a,                                              \
                  trans_b,                                              \
                  m,                                                    \
                  n,                                                    \
                  k,                                                    \
                  (const void*&)alpha,                                  \
                  (const void*&)A,                                      \
                  ld_a,                                                 \
                  stride_a,                                             \
                  (const void*&)B,                                      \
                  ld_b,                                                 \
                  stride_b,                                             \
                  (const void*&)beta,                                   \
                  (const void*&)C,                                      \
                  ld_c,                                                 \
                  stride_c,                                             \
                  b_c);                                                 \
    }                                                                   \
                                                                        \
    if(m == 0 || n == 0 || k == 0 || b_c == 0)                          \
    {                                                                   \
        return rocblas_status_success;                                  \
    }                                                                   \
                                                                        \
    if(validArgs != rocblas_status_success)                             \
        return validArgs;                                               \
                                                                        \
    unsigned int strideC1 = static_cast<unsigned int>(ld_c);            \
    unsigned int strideC2 = static_cast<unsigned int>(stride_c);        \
    unsigned int strideA1 = static_cast<unsigned int>(ld_a);            \
    unsigned int strideA2 = static_cast<unsigned int>(stride_a);        \
    unsigned int strideB1 = static_cast<unsigned int>(ld_b);            \
    unsigned int strideB2 = static_cast<unsigned int>(stride_b);        \
    unsigned int sizeI    = static_cast<unsigned int>(m);               \
    unsigned int sizeJ    = static_cast<unsigned int>(n);               \
    unsigned int sizeK    = static_cast<unsigned int>(b_c);             \
    unsigned int sizeL    = static_cast<unsigned int>(k);

/*******************************************************************************
 * Calling Tensile
 ******************************************************************************/
#ifndef NDEBUG

#define PRINT_SOLUTION_NAME(PREC, TRANS)                                            \
    std::cout << "Solution Name: "                                                  \
              << tensileGetSolutionName_##TRANS##_##PREC##B(strideC1,               \
                                                            strideC2,               \
                                                            strideA1,               \
                                                            strideA2,               \
                                                            strideB1,               \
                                                            strideB2,               \
                                                            sizeI,                  \
                                                            sizeJ,                  \
                                                            sizeK,                  \
                                                            sizeL,                  \
                                                            handle->rocblas_stream) \
              << std::endl;

#define PRINT_RETURN_STATUS std::cout << "Return Status: " << status << std::endl;

#else
#define PRINT_SOLUTION_NAME(PREC, TRANS)
#define PRINT_RETURN_STATUS
#endif

#define CALL_TENSILE(PREC, TYPE, TRANS)                                                              \
    PRINT_SOLUTION_NAME(PREC, TRANS)                                                                 \
    TYPE alpha_h;                                                                                    \
    TYPE beta_h;                                                                                     \
    if(rocblas_pointer_mode_host == handle->pointer_mode)                                            \
    {                                                                                                \
        alpha_h = *alpha;                                                                            \
        beta_h  = *beta;                                                                             \
    }                                                                                                \
    else                                                                                             \
    {                                                                                                \
        hipMemcpy(&alpha_h, alpha, sizeof(TYPE), hipMemcpyDeviceToHost);                             \
        hipMemcpy(&beta_h, beta, sizeof(TYPE), hipMemcpyDeviceToHost);                               \
    }                                                                                                \
    unsigned int int_limit      = std::numeric_limits<int>::max() / sizeof(TYPE);                    \
    unsigned int n_chunk_size_c = int_limit / strideC1;                                              \
    unsigned int n_chunk_size_b = int_limit / strideB1;                                              \
    unsigned int n_chunk_size   = n_chunk_size_c < n_chunk_size_b ? n_chunk_size_c : n_chunk_size_b; \
    unsigned int m_chunk_size   = int_limit / strideA1;                                              \
    unsigned int n_chunk_count  = ((sizeJ - 1) / n_chunk_size) + 1;                                  \
    unsigned int m_chunk_count  = ((sizeI - 1) / m_chunk_size) + 1;                                  \
    for(int n_chunk_iterator = 0; n_chunk_iterator < n_chunk_count; n_chunk_iterator++)              \
    {                                                                                                \
        unsigned int n_chunk_remaining = sizeJ - (n_chunk_size * n_chunk_iterator);                  \
        unsigned int n_chunk_sizeJ =                                                                 \
            n_chunk_size < n_chunk_remaining ? n_chunk_size : n_chunk_remaining;                     \
        for(int m_chunk_iterator = 0; m_chunk_iterator < m_chunk_count; m_chunk_iterator++)          \
        {                                                                                            \
            unsigned int m_chunk_remaining = sizeI - (m_chunk_size * m_chunk_iterator);              \
            unsigned int m_chunk_sizeI =                                                             \
                m_chunk_size < m_chunk_remaining ? m_chunk_size : m_chunk_remaining;                 \
                                                                                                     \
            if(trans_a == rocblas_operation_none)                                                    \
            {                                                                                        \
                if(strideA1 * sizeL > int_limit)                                                     \
                {                                                                                    \
                    std::cerr << "rocBLAS ERROR: lda*k exceeds address limit" << std::endl;          \
                }                                                                                    \
            }                                                                                        \
            else                                                                                     \
            {                                                                                        \
                if(strideA1 * m_chunk_sizeI > int_limit)                                             \
                {                                                                                    \
                    std::cerr << "rocBLAS ERROR: lda*m exceeds address limit" << std::endl;          \
                }                                                                                    \
            }                                                                                        \
            if(trans_b == rocblas_operation_none)                                                    \
            {                                                                                        \
                if(strideB1 * n_chunk_sizeJ > int_limit)                                             \
                {                                                                                    \
                    std::cerr << "rocBLAS ERROR: ldb*n exceeds address limit" << std::endl;          \
                }                                                                                    \
            }                                                                                        \
            else                                                                                     \
            {                                                                                        \
                if(strideB1 * sizeL > int_limit)                                                     \
                {                                                                                    \
                    std::cerr << "rocBLAS ERROR: ldb*k exceeds address limit" << std::endl;          \
                }                                                                                    \
            }                                                                                        \
            if(strideC1 * n_chunk_sizeJ > int_limit)                                                 \
            {                                                                                        \
                std::cerr << "rocBLAS ERROR: ldc*n exceeds address limit" << std::endl;              \
            }                                                                                        \
                                                                                                     \
            size_t C_offset =                                                                        \
                n_chunk_iterator * n_chunk_size * strideC1 + m_chunk_iterator * m_chunk_size;        \
            size_t B_offset = n_chunk_iterator * n_chunk_size;                                       \
            size_t A_offset = m_chunk_iterator * m_chunk_size;                                       \
            if(trans_b == rocblas_operation_none)                                                    \
                B_offset *= strideB1;                                                                \
            if(trans_a != rocblas_operation_none)                                                    \
                A_offset *= strideA1;                                                                \
                                                                                                     \
            status = tensile_##TRANS##_##PREC##B(C + C_offset,                                       \
                                                 A + A_offset,                                       \
                                                 B + B_offset,                                       \
                                                 alpha_h,                                            \
                                                 beta_h,                                             \
                                                 0,                                                  \
                                                 0,                                                  \
                                                 0,                                                  \
                                                 strideC1,                                           \
                                                 strideC2,                                           \
                                                 strideA1,                                           \
                                                 strideA2,                                           \
                                                 strideB1,                                           \
                                                 strideB2,                                           \
                                                 m_chunk_sizeI,                                      \
                                                 n_chunk_sizeJ,                                      \
                                                 sizeK,                                              \
                                                 sizeL,                                              \
                                                 handle->rocblas_stream,                             \
                                                 0,                                                  \
                                                 nullptr,                                            \
                                                 nullptr);                                           \
        }                                                                                            \
    }                                                                                                \
    PRINT_RETURN_STATUS

#define CALL_HTENSILE(PREC, TYPE, TRANS)                                         \
    PRINT_SOLUTION_NAME(PREC, TRANS)                                             \
    TYPE alpha_h;                                                                \
    TYPE beta_h;                                                                 \
    if(rocblas_pointer_mode_host == handle->pointer_mode)                        \
    {                                                                            \
        alpha_h = *alpha;                                                        \
        beta_h  = *beta;                                                         \
    }                                                                            \
    else                                                                         \
    {                                                                            \
        hipMemcpy(&alpha_h, alpha, sizeof(TYPE), hipMemcpyDeviceToHost);         \
        hipMemcpy(&beta_h, beta, sizeof(TYPE), hipMemcpyDeviceToHost);           \
    }                                                                            \
    status = tensile_##TRANS##_##PREC##B(reinterpret_cast<_Float16*>(C),         \
                                         reinterpret_cast<const _Float16*>(A),   \
                                         reinterpret_cast<const _Float16*>(B),   \
                                         *reinterpret_cast<_Float16*>(&alpha_h), \
                                         *reinterpret_cast<_Float16*>(&beta_h),  \
                                         0,                                      \
                                         0,                                      \
                                         0,                                      \
                                         strideC1,                               \
                                         strideC2,                               \
                                         strideA1,                               \
                                         strideA2,                               \
                                         strideB1,                               \
                                         strideB2,                               \
                                         sizeI,                                  \
                                         sizeJ,                                  \
                                         sizeK,                                  \
                                         sizeL,                                  \
                                         handle->rocblas_stream,                 \
                                         0,                                      \
                                         nullptr,                                \
                                         nullptr);                               \
    PRINT_RETURN_STATUS

/*******************************************************************************
 * Handle Transposes
 ******************************************************************************/
#define TENSILE_TRANSPOSES(PREC, TYPE)               \
    hipError_t status;                               \
    if(trans_a == rocblas_operation_none)            \
    {                                                \
        if(trans_b == rocblas_operation_none)        \
        { /*NN*/                                     \
            CALL_TENSILE(PREC, TYPE, Cijk_Ailk_Bljk) \
        }                                            \
        else                                         \
        { /*NT*/                                     \
            CALL_TENSILE(PREC, TYPE, Cijk_Ailk_Bjlk) \
        }                                            \
    }                                                \
    else                                             \
    { /*TN*/                                         \
        if(trans_b == rocblas_operation_none)        \
        {                                            \
            CALL_TENSILE(PREC, TYPE, Cijk_Alik_Bljk) \
        }                                            \
        else                                         \
        { /*TT*/                                     \
            CALL_TENSILE(PREC, TYPE, Cijk_Alik_Bjlk) \
        }                                            \
    }                                                \
    return get_rocblas_status_for_hip_status(status);

#define HTENSILE_TRANSPOSES(PREC, TYPE)               \
    hipError_t status;                                \
    if(trans_a == rocblas_operation_none)             \
    {                                                 \
        if(trans_b == rocblas_operation_none)         \
        { /*NN*/                                      \
            CALL_HTENSILE(PREC, TYPE, Cijk_Ailk_Bljk) \
        }                                             \
        else                                          \
        { /*NT*/                                      \
            CALL_HTENSILE(PREC, TYPE, Cijk_Ailk_Bjlk) \
        }                                             \
    }                                                 \
    else                                              \
    { /*TN*/                                          \
        if(trans_b == rocblas_operation_none)         \
        {                                             \
            CALL_HTENSILE(PREC, TYPE, Cijk_Alik_Bljk) \
        }                                             \
        else                                          \
        { /*TT*/                                      \
            CALL_HTENSILE(PREC, TYPE, Cijk_Alik_Bjlk) \
        }                                             \
    }                                                 \
    return get_rocblas_status_for_hip_status(status);

/*******************************************************************************
 * Batched vs Non
 ******************************************************************************/
#define GEMM_API(prec, PREC, TYPE)                  \
    rocblas_status rocblas_##prec##gemm(ARGS(TYPE)) \
    {                                               \
        PREAMBLE(TYPE)                              \
        TENSILE_TRANSPOSES(PREC, TYPE)              \
    }

#define GEMM_API_BATCHED(prec, PREC, TYPE)                                  \
    rocblas_status rocblas_##prec##gemm_strided_batched(ARGS_BATCHED(TYPE)) \
    {                                                                       \
        PREAMBLE_BATCHED(TYPE)                                              \
        TENSILE_TRANSPOSES(PREC, TYPE)                                      \
    }
#define HGEMM_API(prec, PREC, TYPE)                 \
    rocblas_status rocblas_##prec##gemm(ARGS(TYPE)) \
    {                                               \
        PREAMBLE(TYPE)                              \
        HTENSILE_TRANSPOSES(PREC, TYPE)             \
    }

#define HGEMM_API_BATCHED(prec, PREC, TYPE)                                 \
    rocblas_status rocblas_##prec##gemm_strided_batched(ARGS_BATCHED(TYPE)) \
    {                                                                       \
        PREAMBLE_BATCHED(TYPE)                                              \
        HTENSILE_TRANSPOSES(PREC, TYPE)                                     \
    }

/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
HGEMM_API(h, H, rocblas_half)
GEMM_API(s, S, float)
GEMM_API(d, D, double)
HGEMM_API_BATCHED(h, H, rocblas_half)
GEMM_API_BATCHED(s, S, float)
GEMM_API_BATCHED(d, D, double)

template <typename T>
const char* tensileGetSolutionName_Cijk_Ailk_Bjlk(rocblas_int strideC1,
                                                  rocblas_int strideC2,
                                                  rocblas_int strideA1,
                                                  rocblas_int strideA2,
                                                  rocblas_int strideB1,
                                                  rocblas_int strideB2,
                                                  rocblas_int sizeI,
                                                  rocblas_int sizeJ,
                                                  rocblas_int sizeK,
                                                  rocblas_int sizeL,
                                                  rocblas_handle handle);

template <>
const char* tensileGetSolutionName_Cijk_Ailk_Bjlk<rocblas_half>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Ailk_Bjlk_HB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Ailk_Bjlk<float>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Ailk_Bjlk_SB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Ailk_Bjlk<double>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Ailk_Bjlk_DB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <typename T>
const char* tensileGetSolutionName_Cijk_Ailk_Bljk(rocblas_int strideC1,
                                                  rocblas_int strideC2,
                                                  rocblas_int strideA1,
                                                  rocblas_int strideA2,
                                                  rocblas_int strideB1,
                                                  rocblas_int strideB2,
                                                  rocblas_int sizeI,
                                                  rocblas_int sizeJ,
                                                  rocblas_int sizeK,
                                                  rocblas_int sizeL,
                                                  rocblas_handle handle);

template <>
const char* tensileGetSolutionName_Cijk_Ailk_Bljk<rocblas_half>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Ailk_Bljk_HB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Ailk_Bljk<float>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Ailk_Bljk_SB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Ailk_Bljk<double>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Ailk_Bljk_DB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <typename T>
const char* tensileGetSolutionName_Cijk_Alik_Bjlk(rocblas_int strideC1,
                                                  rocblas_int strideC2,
                                                  rocblas_int strideA1,
                                                  rocblas_int strideA2,
                                                  rocblas_int strideB1,
                                                  rocblas_int strideB2,
                                                  rocblas_int sizeI,
                                                  rocblas_int sizeJ,
                                                  rocblas_int sizeK,
                                                  rocblas_int sizeL,
                                                  rocblas_handle handle);

template <>
const char* tensileGetSolutionName_Cijk_Alik_Bjlk<rocblas_half>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Alik_Bjlk_HB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Alik_Bjlk<float>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Alik_Bjlk_SB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Alik_Bjlk<double>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Alik_Bjlk_DB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <typename T>
const char* tensileGetSolutionName_Cijk_Alik_Bljk(rocblas_int strideC1,
                                                  rocblas_int strideC2,
                                                  rocblas_int strideA1,
                                                  rocblas_int strideA2,
                                                  rocblas_int strideB1,
                                                  rocblas_int strideB2,
                                                  rocblas_int sizeI,
                                                  rocblas_int sizeJ,
                                                  rocblas_int sizeK,
                                                  rocblas_int sizeL,
                                                  rocblas_handle handle);

template <>
const char* tensileGetSolutionName_Cijk_Alik_Bljk<rocblas_half>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Alik_Bljk_HB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Alik_Bljk<float>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Alik_Bljk_SB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <>
const char* tensileGetSolutionName_Cijk_Alik_Bljk<double>(rocblas_int strideC1,
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
    return tensileGetSolutionName_Cijk_Alik_Bljk_DB(strideC1,
                                                    strideC2,
                                                    strideA1,
                                                    strideA2,
                                                    strideB1,
                                                    strideB2,
                                                    sizeI,
                                                    sizeJ,
                                                    sizeK,
                                                    sizeL,
                                                    handle->rocblas_stream);
};

template <typename T>
rocblas_status rocblas_gemm_kernel_name_template(rocblas_handle handle,
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

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xgemm_strided_batched"),
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)A,
                  ld_a,
                  stride_a,
                  (const void*&)B,
                  ld_b,
                  stride_b,
                  *beta,
                  (const void*&)C,
                  ld_c,
                  stride_c,
                  b_c);

        std::string trans_a_letter = rocblas_transpose_letter(trans_a);
        std::string trans_b_letter = rocblas_transpose_letter(trans_b);

        log_bench(handle,
                  "./rocblas-bench -f gemm_strided_batched -r",
                  replaceX<T>("X"),
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
        log_trace(handle,
                  replaceX<T>("rocblas_Xgemm_strided_batched"),
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)A,
                  ld_a,
                  stride_a,
                  (const void*&)B,
                  ld_b,
                  stride_b,
                  (const void*&)beta,
                  (const void*&)C,
                  ld_c,
                  stride_c,
                  b_c);
    }

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

    const char* solution_name;
    if(trans_a == rocblas_operation_none)
    {
        if(trans_b == rocblas_operation_none)
        { /*NN*/
            solution_name = tensileGetSolutionName_Cijk_Ailk_Bljk<T>(strideC1,
                                                                     strideC2,
                                                                     strideA1,
                                                                     strideA2,
                                                                     strideB1,
                                                                     strideB2,
                                                                     sizeI,
                                                                     sizeJ,
                                                                     sizeK,
                                                                     sizeL,
                                                                     handle);
        }
        else
        { /*NT*/
            solution_name = tensileGetSolutionName_Cijk_Ailk_Bjlk<T>(strideC1,
                                                                     strideC2,
                                                                     strideA1,
                                                                     strideA2,
                                                                     strideB1,
                                                                     strideB2,
                                                                     sizeI,
                                                                     sizeJ,
                                                                     sizeK,
                                                                     sizeL,
                                                                     handle);
        }
    }
    else
    {
        if(trans_b == rocblas_operation_none)
        { /*TN*/
            solution_name = tensileGetSolutionName_Cijk_Alik_Bljk<T>(strideC1,
                                                                     strideC2,
                                                                     strideA1,
                                                                     strideA2,
                                                                     strideB1,
                                                                     strideB2,
                                                                     sizeI,
                                                                     sizeJ,
                                                                     sizeK,
                                                                     sizeL,
                                                                     handle);
        }
        else
        { /*TT*/
            solution_name = tensileGetSolutionName_Cijk_Alik_Bjlk<T>(strideC1,
                                                                     strideC2,
                                                                     strideA1,
                                                                     strideA2,
                                                                     strideB1,
                                                                     strideB2,
                                                                     sizeI,
                                                                     sizeJ,
                                                                     sizeK,
                                                                     sizeL,
                                                                     handle);
        }
    }
    std::cout << solution_name << std::endl;

    return validArgs;
}

rocblas_status rocblas_hgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const rocblas_half* alpha,
                                         const rocblas_half* A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const rocblas_half* B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const rocblas_half* beta,
                                         rocblas_half* C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    rocblas_status status = rocblas_gemm_kernel_name_template<rocblas_half>(handle,
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

    return status;
}

rocblas_status rocblas_sgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const float* alpha,
                                         const float* A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const float* B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const float* beta,
                                         float* C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    rocblas_status status = rocblas_gemm_kernel_name_template<float>(handle,
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

    return status;
}

rocblas_status rocblas_dgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const double* alpha,
                                         const double* A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const double* B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const double* beta,
                                         double* C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    rocblas_status status = rocblas_gemm_kernel_name_template<double>(handle,
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

    return status;
}
