/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_gemm_ex.hpp"
#include "logging.h"

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle    handle,
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
                                          rocblas_datatype  compute_type,
                                          rocblas_gemm_algo algo,
                                          int32_t           solution_index,
                                          uint32_t          flags)
{
    // handle, alpha, beta must not be null pointers for logging
    if(!handle)
        return rocblas_status_invalid_handle;

    // TODO: Compute an optimum size of device memory which can be used as workspace.
    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

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

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_trace))
        {
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                std::stringstream alphass;
                std::stringstream betass;
                std::stringstream bench_alphass;
                std::stringstream bench_betass;

                if(compute_type == rocblas_datatype_f16_r)
                {
                    alphass << *((const _Float16*)alpha);
                    betass << *((const _Float16*)beta);

                    bench_alphass << "--alpha " << *((const _Float16*)alpha);
                    bench_betass << "--beta " << *((const _Float16*)beta);
                }
                else if(compute_type == rocblas_datatype_f32_r)
                {
                    alphass << *((const float*)alpha);
                    betass << *((const float*)beta);

                    bench_alphass << "--alpha " << *((const float*)alpha);
                    bench_betass << "--beta " << *((const float*)beta);
                }
                else if(compute_type == rocblas_datatype_f64_r)
                {
                    alphass << *((const double*)alpha);
                    betass << *((const double*)beta);

                    bench_alphass << "--alpha " << *((const double*)alpha);
                    bench_betass << "--beta " << *((const double*)beta);
                }
                else if(compute_type == rocblas_datatype_i32_r)
                {
                    alphass << *((const int32_t*)alpha);
                    betass << *((const int32_t*)beta);

                    bench_alphass << "--alpha " << *((const int32_t*)alpha);
                    bench_betass << "--beta " << *((const int32_t*)beta);
                }
                else if(compute_type == rocblas_datatype_f32_c)
                {
                    rocblas_float_complex tmpa = *((const rocblas_float_complex*)alpha);
                    rocblas_float_complex tmpb = *((const rocblas_float_complex*)beta);

                    alphass << tmpa;
                    betass << tmpb;

                    bench_alphass << "--alpha " << std::real(tmpa);
                    if(std::imag(tmpa) != 0)
                        bench_alphass << " --alphai " << std::imag(tmpa);
                    bench_betass << "--beta " << std::real(tmpb);
                    if(std::imag(tmpb) != 0)
                        bench_betass << " --betai " << std::imag(tmpb);
                }
                else if(compute_type == rocblas_datatype_f64_c)
                {
                    rocblas_double_complex tmpa = *((const rocblas_double_complex*)alpha);
                    rocblas_double_complex tmpb = *((const rocblas_double_complex*)beta);

                    alphass << tmpa;
                    betass << tmpb;

                    bench_alphass << "--alpha " << std::real(tmpa);
                    if(std::imag(tmpa) != 0)
                        bench_alphass << " --alphai " << std::imag(tmpa);
                    bench_betass << "--beta " << std::real(tmpb);
                    if(std::imag(tmpb) != 0)
                        bench_betass << " --betai " << std::imag(tmpb);
                }

                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    log_trace(handle,
                              "rocblas_gemm_ex",
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
                              compute_type_string,
                              algo,
                              solution_index,
                              flags);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f gemm_ex",
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
                              bench_alphass.str(),
                              "--a_type",
                              a_type_string,
                              "--lda",
                              lda,
                              "--b_type",
                              b_type_string,
                              "--ldb",
                              ldb,
                              bench_betass.str(),
                              "--c_type",
                              c_type_string,
                              "--ldc",
                              ldc,
                              "--d_type",
                              d_type_string,
                              "--ldd",
                              ldd,
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
                    log_trace(handle,
                              "rocblas_gemm_ex",
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha,
                              a,
                              a_type_string,
                              lda,
                              b,
                              b_type_string,
                              ldb,
                              beta,
                              c,
                              c_type_string,
                              ldc,
                              d,
                              d_type_string,
                              ldd,
                              compute_type_string,
                              algo,
                              solution_index,
                              flags);
            }
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        "rocblas_gemm_ex",
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
                        "algo",
                        algo,
                        "solution_index",
                        solution_index,
                        "flags",
                        flags);
        }
    }

    // quick return m,n,k equal to 0 is valid in BLAS
    if(!m || !n || !k)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0)
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b || !c || !d)
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = m;
    rocblas_int num_rows_d = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
        return rocblas_status_invalid_size;

    rocblas_status rb_status   = rocblas_status_internal_error;
    rocblas_int    batch_count = 1;
    rocblas_int    stride_a    = trans_a == rocblas_operation_none ? lda * k : lda * m;
    rocblas_int    stride_b    = trans_b == rocblas_operation_none ? ldb * n : ldb * k;
    rocblas_int    stride_c    = ldc * n;
    rocblas_int    stride_d    = ldd * n;

#define EX_TYPECASTING_PARM                                                                     \
    handle, trans_a, trans_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, \
        stride_c, d, ldd, stride_d, batch_count

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_typecasting<double, double, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<float, float, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, _Float16>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status
            = gemm_ex_typecasting<tensile_bfloat16, tensile_bfloat16, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        // For now, K must be a multiple of 4, and/or LDA/LDB based on transpose mode
        if(k % 4 != 0 || (trans_a == rocblas_operation_transpose && lda % 4 != 0)
           || (trans_b == rocblas_operation_none && ldb % 4 != 0))
        {
            rb_status = rocblas_status_invalid_size;
        }
        else
        {
            // adjust by 4 for Tensile
            lda      = (trans_a == rocblas_operation_none) ? lda : lda / 4;
            ldb      = (trans_b == rocblas_operation_none) ? ldb / 4 : ldb;
            stride_a = stride_a / 4;
            stride_b = stride_b / 4;
            k        = k / 4;

            rb_status = gemm_ex_typecasting<TensileInt8x4, TensileInt32, TensileInt32>(
                EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ex_typecasting<rocblas_float_complex,
                                        rocblas_float_complex,
                                        rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ex_typecasting<rocblas_double_complex,
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
                                                          rocblas_long      stride_a,
                                                          const void*       b,
                                                          rocblas_datatype  b_type,
                                                          rocblas_int       ldb,
                                                          rocblas_long      stride_b,
                                                          const void*       beta,
                                                          const void*       c,
                                                          rocblas_datatype  c_type,
                                                          rocblas_int       ldc,
                                                          rocblas_long      stride_c,
                                                          void*             d,
                                                          rocblas_datatype  d_type,
                                                          rocblas_int       ldd,
                                                          rocblas_long      stride_d,
                                                          rocblas_int       batch_count,
                                                          rocblas_datatype  compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t           solution_index,
                                                          uint32_t          flags)
{
    // handle, alpha, beta must not be null pointers for logging
    if(!handle)
        return rocblas_status_invalid_handle;

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

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
                std::stringstream alphass;
                std::stringstream betass;
                std::stringstream bench_alphass;
                std::stringstream bench_betass;

                if(compute_type == rocblas_datatype_f16_r)
                {
                    alphass << *((const _Float16*)alpha);
                    betass << *((const _Float16*)beta);

                    bench_alphass << "--alpha " << *((const _Float16*)alpha);
                    bench_betass << "--beta " << *((const _Float16*)beta);
                }
                else if(compute_type == rocblas_datatype_f32_r)
                {
                    alphass << *((const float*)alpha);
                    betass << *((const float*)beta);

                    bench_alphass << "--alpha " << *((const float*)alpha);
                    bench_betass << "--beta " << *((const float*)beta);
                }
                else if(compute_type == rocblas_datatype_f64_r)
                {
                    alphass << *((const double*)alpha);
                    betass << *((const double*)beta);

                    bench_alphass << "--alpha " << *((const double*)alpha);
                    bench_betass << "--beta " << *((const double*)beta);
                }
                else if(compute_type == rocblas_datatype_i32_r)
                {
                    alphass << *((const int32_t*)alpha);
                    betass << *((const int32_t*)beta);

                    bench_alphass << "--alpha " << *((const int32_t*)alpha);
                    bench_betass << "--beta " << *((const int32_t*)beta);
                }
                else if(compute_type == rocblas_datatype_f32_c)
                {
                    rocblas_float_complex tmpa = *((const rocblas_float_complex*)alpha);
                    rocblas_float_complex tmpb = *((const rocblas_float_complex*)beta);

                    alphass << tmpa;
                    betass << tmpb;

                    bench_alphass << "--alpha " << std::real(tmpa);
                    if(std::imag(tmpa) != 0)
                        bench_alphass << " --alphai " << std::imag(tmpa);

                    bench_betass << "--beta " << std::real(tmpb);
                    if(std::imag(tmpb) != 0)
                        bench_betass << " --betai " << std::imag(tmpb);
                }
                else if(compute_type == rocblas_datatype_f64_c)
                {
                    rocblas_double_complex tmpa = *((const rocblas_double_complex*)alpha);
                    rocblas_double_complex tmpb = *((const rocblas_double_complex*)beta);

                    alphass << tmpa;
                    betass << tmpb;

                    bench_alphass << "--alpha " << std::real(tmpa);
                    if(std::imag(tmpa) != 0)
                        bench_alphass << " --alphai " << std::imag(tmpa);
                    bench_betass << "--beta " << std::real(tmpb);
                    if(std::imag(tmpb) != 0)
                        bench_betass << " --betai " << std::imag(tmpb);
                }

                if(layer_mode & rocblas_layer_mode_log_trace)
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
                if(layer_mode & rocblas_layer_mode_log_bench)
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
                              bench_alphass.str(),
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
                              bench_betass.str(),
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
                              "--batch",
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

    // quick return m,n,k equal to 0 is valid in BLAS
    if(!m || !n || !k || !batch_count)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b || !c || !d)
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = m;
    rocblas_int num_rows_d = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
        return rocblas_status_invalid_size;

    rocblas_status rb_status = rocblas_status_internal_error;

#define EX_TYPECASTING_PARM                                                                     \
    handle, trans_a, trans_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, \
        stride_c, d, ldd, stride_d, batch_count

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_typecasting<double, double, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<float, float, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, _Float16>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status
            = gemm_ex_typecasting<tensile_bfloat16, tensile_bfloat16, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        // For now, K must be a multiple of 4
        if(k % 4 != 0 || ((trans_a == rocblas_operation_transpose) && (lda % 4 != 0))
           || ((trans_b == rocblas_operation_none) && (ldb % 4 != 0)) || stride_a % 4 != 0
           || stride_b % 4 != 0)
        {
            rb_status = rocblas_status_invalid_size;
        }
        else
        {
            // adjust by 4 for Tensile
            lda      = (trans_a == rocblas_operation_none) ? lda : lda / 4;
            ldb      = (trans_b == rocblas_operation_none) ? ldb / 4 : ldb;
            stride_a = stride_a / 4;
            stride_b = stride_b / 4;
            k        = k / 4;

            rb_status = gemm_ex_typecasting<TensileInt8x4, TensileInt32, TensileInt32>(
                EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ex_typecasting<rocblas_float_complex,
                                        rocblas_float_complex,
                                        rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ex_typecasting<rocblas_double_complex,
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
