/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifdef BUILD_WITH_TENSILE
#include "../blas3/Tensile/gemm_tensile.hpp"
#else
#include "../../src/src64/blas_ex/rocblas_gemm_ex_64.hpp"
#endif

#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_gemm_ex.hpp"

// if no Tensile then we use rocblas_internal_gemm_ex_typecasting_64 and rocblas_internal_gemm_ex_64
// with source kernels so don't even provide rocblas_internal_gemm_ex

#ifdef BUILD_WITH_TENSILE

template <bool BATCHED, typename TScal, typename TiConstPtr, typename ToConstPtr, typename ToPtr>
rocblas_status rocblas_internal_gemm_ex(rocblas_handle     handle,
                                        rocblas_operation  trans_a,
                                        rocblas_operation  trans_b,
                                        rocblas_int        m,
                                        rocblas_int        n,
                                        rocblas_int        k,
                                        TScal              alpha,
                                        TiConstPtr         a,
                                        rocblas_stride     offset_a,
                                        rocblas_int        lda,
                                        rocblas_stride     stride_a,
                                        TiConstPtr         b,
                                        rocblas_stride     offset_b,
                                        rocblas_int        ldb,
                                        rocblas_stride     stride_b,
                                        TScal              beta,
                                        ToConstPtr         c,
                                        rocblas_stride     offset_c,
                                        rocblas_int        ldc,
                                        rocblas_stride     stride_c,
                                        ToPtr              d,
                                        rocblas_stride     offset_d,
                                        rocblas_int        ldd,
                                        rocblas_stride     stride_d,
                                        rocblas_int        batch_count,
                                        rocblas_gemm_algo  algo,
                                        int32_t            solution_index,
                                        rocblas_gemm_flags flags)
{

    // sharing code with gemm
    if(BATCHED)
    {
        return rocblas_call_tensile(handle,
                                    alpha,
                                    beta,
                                    a,
                                    b,
                                    c,
                                    d,
                                    trans_a,
                                    trans_b,
                                    ldd,
                                    stride_d,
                                    offset_d,
                                    ldc,
                                    stride_c,
                                    offset_c,
                                    lda,
                                    stride_a,
                                    offset_a,
                                    ldb,
                                    stride_b,
                                    offset_b,
                                    m,
                                    n,
                                    k,
                                    batch_count,
                                    algo,
                                    solution_index,
                                    flags);
    }
    else
    {
        return rocblas_call_tensile(handle,
                                    alpha,
                                    beta,
                                    a + offset_a,
                                    b + offset_b,
                                    c + offset_c,
                                    d + offset_d,
                                    trans_a,
                                    trans_b,
                                    ldd,
                                    stride_d,
                                    0,
                                    ldc,
                                    stride_c,
                                    0,
                                    lda,
                                    stride_a,
                                    0,
                                    ldb,
                                    stride_b,
                                    0,
                                    m,
                                    n,
                                    k,
                                    batch_count,
                                    algo,
                                    solution_index,
                                    flags);
    }
}

template <bool BATCHED, typename Ti, typename To = Ti, typename Tc = To>
rocblas_status gemm_ex_typecasting(rocblas_handle     handle,
                                   rocblas_operation  trans_a,
                                   rocblas_operation  trans_b,
                                   rocblas_int        m,
                                   rocblas_int        n,
                                   rocblas_int        k,
                                   const void*        alpha,
                                   const void*        a,
                                   rocblas_stride     offsetAin,
                                   rocblas_int        lda,
                                   rocblas_stride     stride_a,
                                   const void*        b,
                                   rocblas_stride     offsetBin,
                                   rocblas_int        ldb,
                                   rocblas_stride     stride_b,
                                   const void*        beta,
                                   const void*        c,
                                   rocblas_stride     offsetCin,
                                   rocblas_int        ldc,
                                   rocblas_stride     stride_c,
                                   void*              d,
                                   rocblas_stride     offsetDin,
                                   rocblas_int        ldd,
                                   rocblas_stride     stride_d,
                                   rocblas_int        batch_count,
                                   rocblas_gemm_algo  algo,
                                   int32_t            solution_index,
                                   rocblas_gemm_flags flags)
{
    Tc alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;

    // check alignment of pointers before casting
    if(BATCHED)
    {
        if(!isAligned(a, sizeof(Ti*)) || !isAligned(b, sizeof(Ti*)) || !isAligned(c, sizeof(To*))
           || !isAligned(d, sizeof(To*)))
            return rocblas_status_invalid_size;

        // Pass alpha and beta as simple array (stride of 1)
        // since Tensile does not have gemm_batched, we will have to iterate
        // over batches either way
        if(check_numerics && !std::is_same_v<Ti, signed char>)
        {
            bool           is_input = true;
            rocblas_status gemm_ex_check_numerics_status
                = rocblas_gemm_check_numerics("rocblas_gemm_batched_ex",
                                              handle,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              (const Ti* const*)a,
                                              offsetAin,
                                              lda,
                                              stride_a,
                                              (const Ti* const*)b,
                                              offsetBin,
                                              ldb,
                                              stride_b,
                                              (const To* const*)c,
                                              offsetCin,
                                              ldc,
                                              stride_c,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(gemm_ex_check_numerics_status != rocblas_status_success)
                return gemm_ex_check_numerics_status;
        }

        status = rocblas_internal_gemm_ex<BATCHED>(handle,
                                                   trans_a,
                                                   trans_b,
                                                   m,
                                                   n,
                                                   k,
                                                   (const Tc*)alpha,
                                                   (const Ti* const*)a,
                                                   offsetAin,
                                                   lda,
                                                   stride_a,
                                                   (const Ti* const*)b,
                                                   offsetBin,
                                                   ldb,
                                                   stride_b,
                                                   (const Tc*)beta,
                                                   (const To* const*)c,
                                                   offsetCin,
                                                   ldc,
                                                   stride_c,
                                                   (To* const*)d,
                                                   offsetDin,
                                                   ldd,
                                                   stride_d,
                                                   batch_count,
                                                   algo,
                                                   solution_index,
                                                   flags);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics && !std::is_same_v<Ti, signed char>)
        {
            bool           is_input = false;
            rocblas_status gemm_ex_check_numerics_status
                = rocblas_gemm_check_numerics("rocblas_gemm_batched_ex",
                                              handle,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              (const Ti* const*)a,
                                              offsetAin,
                                              lda,
                                              stride_a,
                                              (const Ti* const*)b,
                                              offsetBin,
                                              ldb,
                                              stride_b,
                                              (To* const*)d,
                                              offsetDin,
                                              ldd,
                                              stride_d,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(gemm_ex_check_numerics_status != rocblas_status_success)
                return gemm_ex_check_numerics_status;
        }
    }
    else
    {
        if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
           || !isAligned(d, sizeof(To)))
            return rocblas_status_invalid_size;

        if(check_numerics && !std::is_same_v<Ti, signed char>)
        {
            bool           is_input                      = true;
            rocblas_status gemm_ex_check_numerics_status = rocblas_gemm_check_numerics(
                stride_a ? "rocblas_gemm_strided_batched_ex" : "rocblas_gemm_ex",
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (const Ti*)a,
                offsetAin,
                lda,
                stride_a,
                (const Ti*)b,
                offsetBin,
                ldb,
                stride_b,
                (const To*)c,
                offsetCin,
                ldc,
                stride_c,
                batch_count,
                check_numerics,
                is_input);
            if(gemm_ex_check_numerics_status != rocblas_status_success)
                return gemm_ex_check_numerics_status;
        }

        status = rocblas_internal_gemm_ex<BATCHED>(handle,
                                                   trans_a,
                                                   trans_b,
                                                   m,
                                                   n,
                                                   k,
                                                   (const Tc*)alpha,
                                                   (const Ti*)a,
                                                   offsetAin,
                                                   lda,
                                                   stride_a,
                                                   (const Ti*)b,
                                                   offsetBin,
                                                   ldb,
                                                   stride_b,
                                                   (const Tc*)beta,
                                                   (const To*)c,
                                                   offsetCin,
                                                   ldc,
                                                   stride_c,
                                                   (To*)d,
                                                   offsetDin,
                                                   ldd,
                                                   stride_d,
                                                   batch_count,
                                                   algo,
                                                   solution_index,
                                                   flags);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics && !std::is_same_v<Ti, signed char>)
        {
            bool           is_input                      = false;
            rocblas_status gemm_ex_check_numerics_status = rocblas_gemm_check_numerics(
                stride_a ? "rocblas_gemm_strided_batched_ex" : "rocblas_gemm_ex",
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (const Ti*)a,
                offsetAin,
                lda,
                stride_a,
                (const Ti*)b,
                offsetBin,
                ldb,
                stride_b,
                (To*)d,
                offsetDin,
                ldd,
                stride_d,
                batch_count,
                check_numerics,
                is_input);
            if(gemm_ex_check_numerics_status != rocblas_status_success)
                return gemm_ex_check_numerics_status;
        }
    }
    return status;
}

#endif // BUILD_WITH_TENSILE

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_template(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        rocblas_int       m,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const void*       alpha,
                                        const void*       a,
                                        rocblas_datatype  a_type,
                                        rocblas_stride    offsetAin,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_a,
                                        const void*       b,
                                        rocblas_datatype  b_type,
                                        rocblas_stride    offsetBin,
                                        rocblas_int       ldb,
                                        rocblas_stride    stride_b,
                                        const void*       beta,
                                        const void*       c,
                                        rocblas_datatype  c_type,
                                        rocblas_stride    offsetCin,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_c,
                                        void*             d,
                                        rocblas_datatype  d_type,
                                        rocblas_stride    offsetDin,
                                        rocblas_int       ldd,
                                        rocblas_stride    stride_d,
                                        rocblas_int       batch_count,
                                        rocblas_datatype  compute_type,
                                        rocblas_gemm_algo algo,
                                        int32_t           solution_index,
                                        uint32_t          flags)
{
    // Note: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(BATCHED)
    {
        stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
        stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
        stride_c = rocblas_stride(ldc) * n;
        stride_d = rocblas_stride(ldd) * n;
    }

    rocblas_status rb_status = rocblas_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                    \
    handle, trans_a, trans_b, m, n, k, alpha, a, offsetAin, lda, stride_a, b, offsetBin, ldb,  \
        stride_b, beta, c, offsetCin, ldc, stride_c, d, offsetDin, ldd, stride_d, batch_count, \
        algo, solution_index, rocblas_gemm_flags(flags)

#ifdef BUILD_WITH_TENSILE

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_typecasting<BATCHED, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<BATCHED, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
    {
        if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
        {
            if(compute_type == rocblas_datatype_f16_r)
            {
                rb_status = gemm_ex_typecasting<BATCHED, rocblas_half>(EX_TYPECASTING_PARM);
            }
            else if(compute_type == rocblas_datatype_f32_r)
            {
                rb_status = gemm_ex_typecasting<BATCHED, rocblas_half, rocblas_half, float>(
                    EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status
                = gemm_ex_typecasting<BATCHED, rocblas_half, float, float>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        if(c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        {
            rb_status = gemm_ex_typecasting<BATCHED, rocblas_bfloat16, rocblas_bfloat16, float>(
                EX_TYPECASTING_PARM);
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        {
            rb_status
                = gemm_ex_typecasting<BATCHED, rocblas_bfloat16, float, float>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        rb_status = gemm_ex_typecasting<BATCHED, int8_t, int32_t>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ex_typecasting<BATCHED,
                                        rocblas_float_complex,
                                        rocblas_float_complex,
                                        rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ex_typecasting<BATCHED,
                                        rocblas_double_complex,
                                        rocblas_double_complex,
                                        rocblas_double_complex>(EX_TYPECASTING_PARM);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;

#else // use source 64 kernels if no tensile

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
    {
        if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
        {
            if(compute_type == rocblas_datatype_f16_r)
            {
                rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED, rocblas_half>(
                    EX_TYPECASTING_PARM);
            }
            else if(compute_type == rocblas_datatype_f32_r)
            {
                rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                    rocblas_half,
                                                                    rocblas_half,
                                                                    float>(EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status
                = rocblas_internal_gemm_ex_typecasting_64<BATCHED, rocblas_half, float, float>(
                    EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        if(c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        {
            rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                rocblas_bfloat16,
                                                                rocblas_bfloat16,
                                                                float>(EX_TYPECASTING_PARM);
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        {
            rb_status
                = rocblas_internal_gemm_ex_typecasting_64<BATCHED, rocblas_bfloat16, float, float>(
                    EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED, int8_t, int32_t>(
            EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status
            = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                      rocblas_float_complex,
                                                      rocblas_float_complex,
                                                      rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status
            = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                      rocblas_double_complex,
                                                      rocblas_double_complex,
                                                      rocblas_double_complex>(EX_TYPECASTING_PARM);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;

#endif
}

#undef EX_TYPECASTING_PARM

#ifdef INSTANTIATE_GEMM_EX_TEMPLATE
#error INSTANTIATE_GEMM_EX_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMM_EX_TEMPLATE(BATCHED_)                                                   \
    template rocblas_status rocblas_gemm_ex_template<BATCHED_>(rocblas_handle    handle,         \
                                                               rocblas_operation trans_a,        \
                                                               rocblas_operation trans_b,        \
                                                               rocblas_int       m,              \
                                                               rocblas_int       n,              \
                                                               rocblas_int       k,              \
                                                               const void*       alpha,          \
                                                               const void*       a,              \
                                                               rocblas_datatype  a_type,         \
                                                               rocblas_stride    offsetAin,      \
                                                               rocblas_int       lda,            \
                                                               rocblas_stride    stride_a,       \
                                                               const void*       b,              \
                                                               rocblas_datatype  b_type,         \
                                                               rocblas_stride    offsetBin,      \
                                                               rocblas_int       ldb,            \
                                                               rocblas_stride    stride_b,       \
                                                               const void*       beta,           \
                                                               const void*       c,              \
                                                               rocblas_datatype  c_type,         \
                                                               rocblas_stride    offsetCin,      \
                                                               rocblas_int       ldc,            \
                                                               rocblas_stride    stride_c,       \
                                                               void*             d,              \
                                                               rocblas_datatype  d_type,         \
                                                               rocblas_stride    offsetDin,      \
                                                               rocblas_int       ldd,            \
                                                               rocblas_stride    stride_d,       \
                                                               rocblas_int       batch_count,    \
                                                               rocblas_datatype  compute_type,   \
                                                               rocblas_gemm_algo algo,           \
                                                               int32_t           solution_index, \
                                                               uint32_t          flags);

INSTANTIATE_GEMM_EX_TEMPLATE(true)
INSTANTIATE_GEMM_EX_TEMPLATE(false)
