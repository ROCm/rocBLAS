/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "Tensile.h"
#include "TensileTypes.h"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemm_ex.hpp"
#include "utility.h"

extern "C" rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle    handle,
                                                          rocblas_operation trans_a,
                                                          rocblas_operation trans_b,
                                                          rocblas_int       m,
                                                          rocblas_int       n,
                                                          rocblas_int       k,
                                                          const void*       alpha,
                                                          const void*       a,
                                                          rocblas_datatype  a_type,
                                                          rocblas_int       lda,
                                                          rocblas_stride    stride_a,
                                                          const void*       b,
                                                          rocblas_datatype  b_type,
                                                          rocblas_int       ldb,
                                                          rocblas_stride    stride_b,
                                                          const void*       beta,
                                                          const void*       c,
                                                          rocblas_datatype  c_type,
                                                          rocblas_int       ldc,
                                                          rocblas_stride    stride_c,
                                                          void*             d,
                                                          rocblas_datatype  d_type,
                                                          rocblas_int       ldd,
                                                          rocblas_stride    stride_d,
                                                          rocblas_int       batch_count,
                                                          rocblas_datatype  compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t           solution_index,
                                                          uint32_t          flags)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    auto layer_mode = handle->layer_mode;
    if(layer_mode
       & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
          | rocblas_layer_mode_log_profile))
    {
        char trans_a_letter, trans_b_letter;
        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            trans_a_letter = rocblas_transpose_letter(trans_a);
            trans_b_letter = rocblas_transpose_letter(trans_b);
        }
        auto a_type_string       = rocblas_datatype_string(a_type);
        auto b_type_string       = rocblas_datatype_string(b_type);
        auto c_type_string       = rocblas_datatype_string(c_type);
        auto d_type_string       = rocblas_datatype_string(d_type);
        auto compute_type_string = rocblas_datatype_string(compute_type);

        if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench))
        {
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    std::stringstream alphass, betass;

                    if(log_trace_alpha_beta_ex(compute_type, alpha, beta, alphass, betass)
                       == rocblas_status_success)
                    {
                        log_trace(handle,
                                  "rocblas_gemm_strided_batched_ex",
                                  trans_a,
                                  trans_b,
                                  m,
                                  n,
                                  k,
                                  alphass.str(),
                                  a,
                                  a_type_string,
                                  lda,
                                  stride_a,
                                  b,
                                  b_type_string,
                                  ldb,
                                  stride_b,
                                  betass.str(),
                                  c,
                                  c_type_string,
                                  ldc,
                                  stride_c,
                                  d,
                                  d_type_string,
                                  ldd,
                                  stride_d,
                                  batch_count,
                                  compute_type_string,
                                  algo,
                                  solution_index,
                                  flags);
                    }
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    if(log_bench_alpha_beta_ex(compute_type, alpha, beta, alphas, betas)
                       == rocblas_status_success)
                    {
                        log_bench(handle,
                                  "./rocblas-bench -f gemm_strided_batched_ex",
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
                                  alphas,
                                  "--a_type",
                                  a_type_string,
                                  "--lda",
                                  lda,
                                  "--stride_a",
                                  stride_a,
                                  "--b_type",
                                  b_type_string,
                                  "--ldb",
                                  ldb,
                                  "--stride_b",
                                  stride_b,
                                  betas,
                                  "--c_type",
                                  c_type_string,
                                  "--ldc",
                                  ldc,
                                  "--stride_c",
                                  stride_c,
                                  "--d_type",
                                  d_type_string,
                                  "--ldd",
                                  ldd,
                                  "--stride_d",
                                  stride_d,
                                  "--batch_count",
                                  batch_count,
                                  "--compute_type",
                                  compute_type_string,
                                  "--algo",
                                  algo,
                                  "--solution_index",
                                  solution_index,
                                  "--flags",
                                  flags);
                    }
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    log_trace(handle,
                              "rocblas_gemm_strided_batched_ex",
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha,
                              a,
                              a_type,
                              lda,
                              stride_a,
                              b,
                              b_type,
                              ldb,
                              stride_b,
                              beta,
                              c,
                              c_type,
                              ldc,
                              stride_c,
                              d,
                              d_type,
                              ldd,
                              stride_d,
                              batch_count,
                              compute_type,
                              algo,
                              solution_index,
                              flags);
                }
            }
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        "rocblas_gemm_strided_batched_ex",
                        "a_type",
                        a_type_string,
                        "b_type",
                        b_type_string,
                        "c_type",
                        c_type_string,
                        "d_type",
                        d_type_string,
                        "compute_type",
                        compute_type_string,
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
                        "ldd",
                        ldd,
                        "stride_d",
                        stride_d,
                        "batch_count",
                        batch_count,
                        "algo",
                        algo,
                        "solution_index",
                        solution_index,
                        "flags",
                        flags);
        }
    }

    // Quick return
    // NOTE: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // leading dimensions must be valid
    if(ldc < m || ldd < m || lda < (trans_a == rocblas_operation_none ? m : k)
       || ldb < (trans_b == rocblas_operation_none ? k : n))
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b || !c || !d || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_gemm_ex_template<false>(handle,
                                           trans_a,
                                           trans_b,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           a,
                                           a_type,
                                           0,
                                           lda,
                                           stride_a,
                                           b,
                                           b_type,
                                           0,
                                           ldb,
                                           stride_b,
                                           beta,
                                           c,
                                           c_type,
                                           0,
                                           ldc,
                                           stride_c,
                                           d,
                                           d_type,
                                           0,
                                           ldd,
                                           stride_d,
                                           batch_count,
                                           compute_type);
}
