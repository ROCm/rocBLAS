/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
//#include "Tensile.h"
#include "TensileTypes.h"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemm_ex.hpp"
#include "utility.h"

namespace
{

    rocblas_status rocblas_gemm_batched_ex_impl(rocblas_handle    handle,
                                                rocblas_operation trans_a,
                                                rocblas_operation trans_b,
                                                rocblas_int       m,
                                                rocblas_int       n,
                                                rocblas_int       k,
                                                const void*       alpha,
                                                const void*       a,
                                                rocblas_datatype  a_type,
                                                rocblas_int       lda,
                                                const void*       b,
                                                rocblas_datatype  b_type,
                                                rocblas_int       ldb,
                                                const void*       beta,
                                                const void*       c,
                                                rocblas_datatype  c_type,
                                                rocblas_int       ldc,
                                                void*             d,
                                                rocblas_datatype  d_type,
                                                rocblas_int       ldd,
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
                        switch(compute_type)
                        {
                        default:
                            return rocblas_status_not_implemented;
                        case rocblas_datatype_f16_r:
                            alphass << log_trace_scalar_value((const rocblas_half*)alpha);
                            betass << log_trace_scalar_value((const rocblas_half*)beta);
                            break;
                        case rocblas_datatype_f32_r:
                            alphass << log_trace_scalar_value((const float*)alpha);
                            betass << log_trace_scalar_value((const float*)beta);
                            break;
                        case rocblas_datatype_f64_r:
                            alphass << log_trace_scalar_value((const double*)alpha);
                            betass << log_trace_scalar_value((const double*)beta);
                            break;
                        case rocblas_datatype_i32_r:
                            alphass << log_trace_scalar_value((const int32_t*)alpha);
                            betass << log_trace_scalar_value((const int32_t*)beta);
                            break;
                        case rocblas_datatype_f32_c:
                            alphass << log_trace_scalar_value((const rocblas_float_complex*)alpha);
                            betass << log_trace_scalar_value((const rocblas_float_complex*)beta);
                            break;
                        case rocblas_datatype_f64_c:
                            alphass << log_trace_scalar_value((const rocblas_double_complex*)alpha);
                            betass << log_trace_scalar_value((const rocblas_double_complex*)beta);
                            break;
                        }

                        log_trace(handle,
                                  "rocblas_gemm_batched_ex",
                                  trans_a,
                                  trans_b,
                                  m,
                                  n,
                                  k,
                                  alphass.str(),
                                  a,
                                  a_type_string,
                                  lda,
                                  b,
                                  b_type_string,
                                  ldb,
                                  betass.str(),
                                  c,
                                  c_type_string,
                                  ldc,
                                  d,
                                  d_type_string,
                                  ldd,
                                  batch_count,
                                  compute_type_string,
                                  algo,
                                  solution_index,
                                  flags);
                    }

                    if(layer_mode & rocblas_layer_mode_log_bench)
                    {
                        std::string alphas, betas;
                        switch(compute_type)
                        {
                        default:
                            return rocblas_status_not_implemented;
                        case rocblas_datatype_f16_r:
                            alphas = log_bench_scalar_value("alpha", (const rocblas_half*)alpha);
                            betas  = log_bench_scalar_value("beta", (const rocblas_half*)beta);
                            break;
                        case rocblas_datatype_f32_r:
                            alphas = log_bench_scalar_value("alpha", (const float*)alpha);
                            betas  = log_bench_scalar_value("beta", (const float*)beta);
                            break;
                        case rocblas_datatype_f64_r:
                            alphas = log_bench_scalar_value("alpha", (const double*)alpha);
                            betas  = log_bench_scalar_value("beta", (const double*)beta);
                            break;
                        case rocblas_datatype_i32_r:
                            alphas = log_bench_scalar_value("alpha", (const int32_t*)alpha);
                            betas  = log_bench_scalar_value("beta", (const int32_t*)beta);
                            break;
                        case rocblas_datatype_f32_c:
                            alphas = log_bench_scalar_value("alpha",
                                                            (const rocblas_float_complex*)alpha);
                            betas  = log_bench_scalar_value("beta",
                                                           (const rocblas_float_complex*)beta);
                            break;
                        case rocblas_datatype_f64_c:
                            alphas = log_bench_scalar_value("alpha",
                                                            (const rocblas_double_complex*)alpha);
                            betas  = log_bench_scalar_value("beta",
                                                           (const rocblas_double_complex*)beta);
                            break;
                        }

                        log_bench(handle,
                                  "./rocblas-bench -f gemm_batched_ex",
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
                                  "--b_type",
                                  b_type_string,
                                  "--ldb",
                                  ldb,
                                  betas,
                                  "--c_type",
                                  c_type_string,
                                  "--ldc",
                                  "--d_type",
                                  d_type_string,
                                  "--ldd",
                                  ldd,
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
                else
                {
                    if(layer_mode & rocblas_layer_mode_log_trace)
                    {
                        log_trace(handle,
                                  "rocblas_gemm_batched_ex",
                                  trans_a,
                                  trans_b,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  a,
                                  a_type,
                                  lda,
                                  b,
                                  b_type,
                                  ldb,
                                  beta,
                                  c,
                                  c_type,
                                  ldc,
                                  d,
                                  d_type,
                                  ldd,
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
                            "rocblas_gemm_batched_ex",
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
                            "ldb",
                            ldb,
                            "ldc",
                            ldc,
                            "ldd",
                            ldd,
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

        // quick return m,n,k equal to 0 is valid in BLAS
        // Note: k==0 is NOT a quick return, because C must still be multiplied by beta
        if(!m || !n || !batch_count)
            return rocblas_status_success;

        // sizes must not be negative
        if(m < 0 || n < 0 || k < 0 || batch_count < 0)
            return rocblas_status_invalid_size;

        // pointers must be valid
        if(!a || !b || !c || !d || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
        rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
        rocblas_int num_rows_c = m;
        rocblas_int num_rows_d = m;
        rocblas_int stride_a   = trans_a == rocblas_operation_none ? lda * k : lda * m;
        rocblas_int stride_b   = trans_b == rocblas_operation_none ? ldb * n : ldb * k;
        rocblas_int stride_c   = ldc * n;
        rocblas_int stride_d   = ldd * n;

        // leading dimensions must be valid
        if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
            return rocblas_status_invalid_size;

        rocblas_status rb_status = rocblas_status_internal_error;

#define EX_TYPECASTING_PARM                                                                      \
    handle, trans_a, trans_b, m, n, k, alpha, 0, a, 0, lda, stride_a, b, 0, ldb, stride_b, beta, \
        0, c, 0, ldc, stride_c, d, 0, ldd, stride_d, batch_count

        if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
           && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
           && compute_type == rocblas_datatype_f64_r)
        {
            rb_status = gemm_ex_typecasting<true, double, double, double>(EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
                && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status = gemm_ex_typecasting<true, float, float, float>(EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
                && compute_type == rocblas_datatype_f16_r)
        {
            rb_status = gemm_ex_typecasting<true, rocblas_half, rocblas_half, rocblas_half>(
                EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status
                = gemm_ex_typecasting<true, rocblas_half, rocblas_half, float>(EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
                && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status = gemm_ex_typecasting<true, tensile_bfloat16, tensile_bfloat16, float>(
                EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
                && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
                && compute_type == rocblas_datatype_i32_r)
        {
            // For now, K must be a multiple of 4
            if(k % 4 != 0 || ((trans_a == rocblas_operation_transpose) && (lda % 4 != 0))
               || ((trans_b == rocblas_operation_none) && (ldb % 4 != 0)))
            {
                rb_status = rocblas_status_invalid_size;
            }
            else
            {
                // adjust by 4 for Tensile
                lda = (trans_a == rocblas_operation_none) ? lda : lda / 4;
                ldb = (trans_b == rocblas_operation_none) ? ldb / 4 : ldb;
                k   = k / 4;

                rb_status = gemm_ex_typecasting<true, TensileInt8x4, TensileInt32, TensileInt32>(
                    EX_TYPECASTING_PARM);
            }
        }
        else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
                && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
                && compute_type == rocblas_datatype_f32_c)
        {
            rb_status = gemm_ex_typecasting<true,
                                            rocblas_float_complex,
                                            rocblas_float_complex,
                                            rocblas_float_complex>(EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
                && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
                && compute_type == rocblas_datatype_f64_c)
        {
            rb_status = gemm_ex_typecasting<true,
                                            rocblas_double_complex,
                                            rocblas_double_complex,
                                            rocblas_double_complex>(EX_TYPECASTING_PARM);
        }
        else
        {
            rb_status = rocblas_status_not_implemented;
        }
#undef EX_TYPECASTING_PARM

        return rb_status;
    }

}

extern "C" rocblas_status rocblas_gemm_batched_ex(rocblas_handle    handle,
                                                  rocblas_operation trans_a,
                                                  rocblas_operation trans_b,
                                                  rocblas_int       m,
                                                  rocblas_int       n,
                                                  rocblas_int       k,
                                                  const void*       alpha,
                                                  const void*       a,
                                                  rocblas_datatype  a_type,
                                                  rocblas_int       lda,
                                                  const void*       b,
                                                  rocblas_datatype  b_type,
                                                  rocblas_int       ldb,
                                                  const void*       beta,
                                                  const void*       c,
                                                  rocblas_datatype  c_type,
                                                  rocblas_int       ldc,
                                                  void*             d,
                                                  rocblas_datatype  d_type,
                                                  rocblas_int       ldd,
                                                  rocblas_int       batch_count,
                                                  rocblas_datatype  compute_type,
                                                  rocblas_gemm_algo algo,
                                                  int32_t           solution_index,
                                                  uint32_t          flags)
{
    return rocblas_gemm_batched_ex_impl(handle,
                                        trans_a,
                                        trans_b,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        a,
                                        a_type,
                                        lda,
                                        b,
                                        b_type,
                                        ldb,
                                        beta,
                                        c,
                                        c_type,
                                        ldc,
                                        d,
                                        d_type,
                                        ldd,
                                        batch_count,
                                        compute_type,
                                        algo,
                                        solution_index,
                                        flags);
}
