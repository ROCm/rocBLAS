/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_rot.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <rocblas_int NB,
          bool        ISBATCHED,
          typename Tx,
          typename Ty  = Tx,
          typename Tcs = Ty,
          typename Tex = Tcs>
rocblas_status rot_ex_typecasting(rocblas_handle handle,
                                  rocblas_int    n,
                                  void*          x,
                                  rocblas_int    incx,
                                  rocblas_stride stride_x,
                                  void*          y,
                                  rocblas_int    incy,
                                  rocblas_stride stride_y,
                                  const void*    c,
                                  const void*    s,
                                  rocblas_int    batch_count)
{
    static constexpr rocblas_int    offset_0 = 0;
    static constexpr rocblas_stride stride_0 = 0;
    if(ISBATCHED)
    {
        return rocblas_rot_template<NB, Tex>(handle,
                                             n,
                                             (Tx* const*)x,
                                             offset_0,
                                             incx,
                                             stride_x,
                                             (Ty* const*)y,
                                             offset_0,
                                             incy,
                                             stride_y,
                                             (const Tcs*)c,
                                             stride_0,
                                             (const Tcs*)s,
                                             stride_0,
                                             batch_count);
    }
    else
    {
        return rocblas_rot_template<NB, Tex>(handle,
                                             n,
                                             (Tx*)x,
                                             offset_0,
                                             incx,
                                             stride_x,
                                             (Ty*)y,
                                             offset_0,
                                             incy,
                                             stride_y,
                                             (const Tcs*)c,
                                             stride_0,
                                             (const Tcs*)s,
                                             stride_0,
                                             batch_count);
    }
}

template <rocblas_int NB, bool ISBATCHED = false>
rocblas_status rocblas_rot_ex_template(rocblas_handle   handle,
                                       rocblas_int      n,
                                       void*            x,
                                       rocblas_datatype x_type,
                                       rocblas_int      incx,
                                       rocblas_stride   stride_x,
                                       void*            y,
                                       rocblas_datatype y_type,
                                       rocblas_int      incy,
                                       rocblas_stride   stride_y,
                                       const void*      c,
                                       const void*      s,
                                       rocblas_datatype cs_type,
                                       rocblas_int      batch_count,
                                       rocblas_datatype execution_type)
{
#define ROT_EX_TYPECASTING_PARAM handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count

    if(x_type == rocblas_datatype_bf16_r && y_type == rocblas_datatype_bf16_r
       && cs_type == rocblas_datatype_bf16_r && execution_type == rocblas_datatype_f32_r)
    {
        return rot_ex_typecasting<NB,
                                  ISBATCHED,
                                  rocblas_bfloat16,
                                  rocblas_bfloat16,
                                  rocblas_bfloat16,
                                  float>(ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f16_r && y_type == rocblas_datatype_f16_r
            && cs_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        return rot_ex_typecasting<NB, ISBATCHED, rocblas_half, rocblas_half, rocblas_half, float>(
            ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && y_type == rocblas_datatype_f32_r
            && cs_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_r)
    {
        return rot_ex_typecasting<NB, ISBATCHED, float>(ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_r && y_type == rocblas_datatype_f64_r
            && cs_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        return rot_ex_typecasting<NB, ISBATCHED, double>(ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && y_type == rocblas_datatype_f32_c
            && cs_type == rocblas_datatype_f32_c && execution_type == rocblas_datatype_f32_c)
    {
        return rot_ex_typecasting<NB, ISBATCHED, rocblas_float_complex>(ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && y_type == rocblas_datatype_f32_c
            && cs_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_c)
    {
        return rot_ex_typecasting<NB,
                                  ISBATCHED,
                                  rocblas_float_complex,
                                  rocblas_float_complex,
                                  float,
                                  rocblas_float_complex>(ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && y_type == rocblas_datatype_f64_c
            && cs_type == rocblas_datatype_f64_c && execution_type == rocblas_datatype_f64_c)
    {
        return rot_ex_typecasting<NB, ISBATCHED, rocblas_double_complex>(ROT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && y_type == rocblas_datatype_f64_c
            && cs_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_c)
    {
        return rot_ex_typecasting<NB,
                                  ISBATCHED,
                                  rocblas_double_complex,
                                  rocblas_double_complex,
                                  double,
                                  rocblas_double_complex>(ROT_EX_TYPECASTING_PARAM);
    }

    return rocblas_status_not_implemented;
}
