/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GEMM_EXT2_HPP
#define __ROCBLAS_GEMM_EXT2_HPP

// This functionality is only availble when using the new Tensile client
#ifdef USE_TENSILE_HOST

#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemm_ex.hpp"
#include "utility.h"

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ext2_batched_template(rocblas_handle handle,
                                          rocblas_int    m,
                                          rocblas_int    n,
                                          rocblas_int    k,
                                          const Tc*      alpha,
                                          const Ti*      a,
                                          size_t         offset_a,
                                          rocblas_int    row_stride_a,
                                          rocblas_int    col_stride_a,
                                          rocblas_stride batch_stride_a,
                                          const Ti*      b,
                                          size_t         offset_b,
                                          rocblas_int    row_stride_b,
                                          rocblas_int    col_stride_b,
                                          rocblas_stride batch_stride_b,
                                          const Tc*      beta,
                                          const To*      c,
                                          size_t         offset_c,
                                          rocblas_int    row_stride_c,
                                          rocblas_int    col_stride_c,
                                          rocblas_stride batch_stride_c,
                                          To*            d,
                                          size_t         offset_d,
                                          rocblas_int    row_stride_d,
                                          rocblas_int    col_stride_d,
                                          rocblas_stride batch_stride_d,
                                          rocblas_int    batch_count = 1)
{
    a += offset_a;
    b += offset_b;
    c += offset_c;
    d += offset_d;

    RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  a,
                                                  row_stride_a,
                                                  col_stride_a,
                                                  batch_stride_a,
                                                  b,
                                                  row_stride_b,
                                                  col_stride_b,
                                                  batch_stride_b,
                                                  beta,
                                                  c,
                                                  row_stride_c,
                                                  col_stride_c,
                                                  batch_stride_c,
                                                  d,
                                                  row_stride_d,
                                                  col_stride_d,
                                                  batch_stride_d,
                                                  batch_count};

    return runContractionProblem(problem);
}

template <typename Ti, typename To = Ti, typename Tc = To>
rocblas_status gemm_ext2_typecasting(rocblas_handle handle,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     rocblas_int    k,
                                     const void*    alpha,
                                     const void*    a,
                                     rocblas_int    offsetAin,
                                     rocblas_int    row_stride_a,
                                     rocblas_int    col_stride_a,
                                     rocblas_stride batch_stride_a,
                                     const void*    b,
                                     rocblas_int    offsetBin,
                                     rocblas_int    row_stride_b,
                                     rocblas_int    col_stride_b,
                                     rocblas_stride batch_stride_b,
                                     const void*    beta,
                                     const void*    c,
                                     rocblas_int    offsetCin,
                                     rocblas_int    row_stride_c,
                                     rocblas_int    col_stride_c,
                                     rocblas_stride batch_stride_c,
                                     void*          d,
                                     rocblas_int    offsetDin,
                                     rocblas_int    row_stride_d,
                                     rocblas_int    col_stride_d,
                                     rocblas_stride batch_stride_d,
                                     rocblas_int    batch_count)
{
    Tc alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

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
                                                 rocblas_int      offsetAin,
                                                 rocblas_int      row_stride_a,
                                                 rocblas_int      col_stride_a,
                                                 rocblas_stride   batch_stride_a,
                                                 const void*      b,
                                                 rocblas_datatype b_type,
                                                 rocblas_int      offsetBin,
                                                 rocblas_int      row_stride_b,
                                                 rocblas_int      col_stride_b,
                                                 rocblas_stride   batch_stride_b,
                                                 const void*      beta,
                                                 const void*      c,
                                                 rocblas_datatype c_type,
                                                 rocblas_int      offsetCin,
                                                 rocblas_int      row_stride_c,
                                                 rocblas_int      col_stride_c,
                                                 rocblas_stride   batch_stride_c,
                                                 void*            d,
                                                 rocblas_datatype d_type,
                                                 rocblas_int      offsetDin,
                                                 rocblas_int      row_stride_d,
                                                 rocblas_int      col_stride_d,
                                                 rocblas_stride   batch_stride_d,
                                                 rocblas_int      batch_count,
                                                 rocblas_datatype compute_type)
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
        rb_status = gemm_ext2_typecasting<double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ext2_typecasting<float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ext2_typecasting<rocblas_half>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ext2_typecasting<rocblas_half, rocblas_half, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status
            = gemm_ext2_typecasting<rocblas_bfloat16, rocblas_bfloat16, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
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

            rb_status = gemm_ext2_typecasting<int8_t, int32_t>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ext2_typecasting<rocblas_float_complex,
                                          rocblas_float_complex,
                                          rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ext2_typecasting<rocblas_double_complex,
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

#endif // USE_TENSILE_HOST

#endif
