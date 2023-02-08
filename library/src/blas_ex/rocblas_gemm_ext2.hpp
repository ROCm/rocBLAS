/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_gemm_ex.hpp"
#include "utility.hpp"

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ext2_batched_template(rocblas_handle handle,
                                          rocblas_int    m,
                                          rocblas_int    n,
                                          rocblas_int    k,
                                          const Tc*      alpha,
                                          const Ti*      a,
                                          rocblas_stride offset_a,
                                          rocblas_int    row_stride_a,
                                          rocblas_int    col_stride_a,
                                          rocblas_stride batch_stride_a,
                                          const Ti*      b,
                                          rocblas_stride offset_b,
                                          rocblas_int    row_stride_b,
                                          rocblas_int    col_stride_b,
                                          rocblas_stride batch_stride_b,
                                          const Tc*      beta,
                                          const To*      c,
                                          rocblas_stride offset_c,
                                          rocblas_int    row_stride_c,
                                          rocblas_int    col_stride_c,
                                          rocblas_stride batch_stride_c,
                                          To*            d,
                                          rocblas_stride offset_d,
                                          rocblas_int    row_stride_d,
                                          rocblas_int    col_stride_d,
                                          rocblas_stride batch_stride_d,
                                          rocblas_int    batch_count   = 1,
                                          bool           strided_batch = true)
{
    RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  a,
                                                  nullptr,
                                                  row_stride_a,
                                                  col_stride_a,
                                                  batch_stride_a,
                                                  offset_a,
                                                  b,
                                                  nullptr,
                                                  row_stride_b,
                                                  col_stride_b,
                                                  batch_stride_b,
                                                  offset_b,
                                                  beta,
                                                  c,
                                                  nullptr,
                                                  row_stride_c,
                                                  col_stride_c,
                                                  batch_stride_c,
                                                  offset_c,
                                                  d,
                                                  nullptr,
                                                  row_stride_d,
                                                  col_stride_d,
                                                  batch_stride_d,
                                                  offset_d,
                                                  batch_count,
                                                  strided_batch};

    return runContractionProblem(problem);
}

template <typename Ti, typename To = Ti, typename Tc = To>
rocblas_status rocblas_gemm_ext2_typecasting(rocblas_handle handle,
                                             rocblas_int    m,
                                             rocblas_int    n,
                                             rocblas_int    k,
                                             const void*    alpha,
                                             const void*    a,
                                             rocblas_stride offsetAin,
                                             rocblas_int    row_stride_a,
                                             rocblas_int    col_stride_a,
                                             rocblas_stride batch_stride_a,
                                             const void*    b,
                                             rocblas_stride offsetBin,
                                             rocblas_int    row_stride_b,
                                             rocblas_int    col_stride_b,
                                             rocblas_stride batch_stride_b,
                                             const void*    beta,
                                             const void*    c,
                                             rocblas_stride offsetCin,
                                             rocblas_int    row_stride_c,
                                             rocblas_int    col_stride_c,
                                             rocblas_stride batch_stride_c,
                                             void*          d,
                                             rocblas_stride offsetDin,
                                             rocblas_int    row_stride_d,
                                             rocblas_int    col_stride_d,
                                             rocblas_stride batch_stride_d,
                                             rocblas_int    batch_count)
{
    Tc alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
       || !isAligned(d, sizeof(To)))
        return rocblas_status_invalid_size;

    return gemm_ext2_batched_template(handle,
                                      m,
                                      n,
                                      k,
                                      (const Tc*)alpha,
                                      (const Ti*)a,
                                      offsetAin,
                                      row_stride_a,
                                      col_stride_a,
                                      batch_stride_a,
                                      (const Ti*)b,
                                      offsetBin,
                                      row_stride_b,
                                      col_stride_b,
                                      batch_stride_b,
                                      (const Tc*)beta,
                                      (const To*)c,
                                      offsetCin,
                                      row_stride_c,
                                      col_stride_c,
                                      batch_stride_c,
                                      (To*)d,
                                      offsetDin,
                                      row_stride_d,
                                      col_stride_d,
                                      batch_stride_d,
                                      batch_count);
}

inline rocblas_status rocblas_gemm_ext2_template(rocblas_handle   handle,
                                                 rocblas_int      m,
                                                 rocblas_int      n,
                                                 rocblas_int      k,
                                                 const void*      alpha,
                                                 const void*      a,
                                                 rocblas_datatype a_type,
                                                 rocblas_stride   offsetAin,
                                                 rocblas_int      row_stride_a,
                                                 rocblas_int      col_stride_a,
                                                 rocblas_stride   batch_stride_a,
                                                 const void*      b,
                                                 rocblas_datatype b_type,
                                                 rocblas_stride   offsetBin,
                                                 rocblas_int      row_stride_b,
                                                 rocblas_int      col_stride_b,
                                                 rocblas_stride   batch_stride_b,
                                                 const void*      beta,
                                                 const void*      c,
                                                 rocblas_datatype c_type,
                                                 rocblas_stride   offsetCin,
                                                 rocblas_int      row_stride_c,
                                                 rocblas_int      col_stride_c,
                                                 rocblas_stride   batch_stride_c,
                                                 void*            d,
                                                 rocblas_datatype d_type,
                                                 rocblas_stride   offsetDin,
                                                 rocblas_int      row_stride_d,
                                                 rocblas_int      col_stride_d,
                                                 rocblas_stride   batch_stride_d,
                                                 rocblas_int      batch_count,
                                                 rocblas_datatype compute_type,
                                                 uint32_t         flags)
{
    // Note: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    rocblas_status rb_status = rocblas_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                      \
    handle, m, n, k, alpha, a, offsetAin, row_stride_a, col_stride_a, batch_stride_a, b,         \
        offsetBin, row_stride_b, col_stride_b, batch_stride_b, beta, c, offsetCin, row_stride_c, \
        col_stride_c, batch_stride_c, d, offsetDin, row_stride_d, col_stride_d, batch_stride_d,  \
        batch_count

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = rocblas_gemm_ext2_typecasting<double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = rocblas_gemm_ext2_typecasting<float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
    {
        if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
        {
            if(compute_type == rocblas_datatype_f16_r)
            {
                rb_status = rocblas_gemm_ext2_typecasting<rocblas_half>(EX_TYPECASTING_PARM);
            }
            else if(compute_type == rocblas_datatype_f32_r)
            {
                rb_status = rocblas_gemm_ext2_typecasting<rocblas_half, rocblas_half, float>(
                    EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status
                = rocblas_gemm_ext2_typecasting<rocblas_half, float, float>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        if(c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        {
            rb_status = rocblas_gemm_ext2_typecasting<rocblas_bfloat16, rocblas_bfloat16, float>(
                EX_TYPECASTING_PARM);
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        {
            rb_status = rocblas_gemm_ext2_typecasting<rocblas_bfloat16, float, float>(
                EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        bool useInt8x4 = flags & rocblas_gemm_flags_pack_int8x4;
        // Here is point where we decide to branch to real int8 or rocblas_int8x4
        // MatrixInstruction kernel uses general int8 (unless rocblas_gemm_flags_pack_int8x4 is set)
        if(!useInt8x4)
        {
            rb_status = rocblas_gemm_ext2_typecasting<int8_t, int32_t>(EX_TYPECASTING_PARM);
        }
        // Else, we check if we can pack 4 int8:
        else
        {
            // For now, K must be a multiple of 4
            if(k % 4 || col_stride_b % 4 || batch_stride_a % 4 || batch_stride_b % 4)
            {
                rb_status = rocblas_status_invalid_size;
            }
            else
            {
                // adjust by 4 for Tensile
                col_stride_b /= 4;
                k /= 4;
                batch_stride_a /= 4;
                batch_stride_b /= 4;

                rb_status
                    = rocblas_gemm_ext2_typecasting<rocblas_int8x4, int32_t>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = rocblas_gemm_ext2_typecasting<rocblas_float_complex,
                                                  rocblas_float_complex,
                                                  rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = rocblas_gemm_ext2_typecasting<rocblas_double_complex,
                                                  rocblas_double_complex,
                                                  rocblas_double_complex>(EX_TYPECASTING_PARM);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}

#undef EX_TYPECASTING_PARM
