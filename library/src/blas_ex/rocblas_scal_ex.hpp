/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "../blas1/rocblas_scal.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <int NB, bool BATCHED, typename Ta, typename Tx = Ta, typename Tex = Tx>
rocblas_status scal_ex_typecasting(rocblas_handle handle,
                                   rocblas_int    n,
                                   const void*    alpha_void,
                                   void*          x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   rocblas_int    batch_count)
{
    const Ta*            alpha        = (const Ta*)alpha_void;
    const rocblas_stride stride_alpha = 0;
    const rocblas_int    offset_x     = 0;

    if(!alpha_void)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 1)
            return rocblas_status_success;
    }

    if(!x)
        return rocblas_status_invalid_pointer;

    if(BATCHED)
    {
        return rocblas_scal_template<NB, Tex>(
            handle, n, alpha, stride_alpha, (Tx* const*)x, offset_x, incx, stride_x, batch_count);
    }
    else
    {
        return rocblas_scal_template<NB, Tex>(
            handle, n, alpha, stride_alpha, (Tx*)x, offset_x, incx, stride_x, batch_count);
    }
}

template <int NB, bool BATCHED = false>
rocblas_status rocblas_scal_ex_template(rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        void*            x,
                                        rocblas_datatype x_type,
                                        rocblas_int      incx,
                                        rocblas_stride   stride_x,
                                        rocblas_int      batch_count,
                                        rocblas_datatype execution_type)
{
    // Error checking
    if(n <= 0 || incx <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
        return rocblas_status_success;

    // Quick return (alpha == 1) check and other nullptr checks will be done
    // once we know the type (in scal_ex_typecasting).

    rocblas_status status = rocblas_status_not_implemented;

#define SCAL_EX_TYPECASTING_PARAM handle, n, alpha, x, incx, stride_x, batch_count

    if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
       && execution_type == rocblas_datatype_f32_r)
    {
        // hscal with float computation
        status = scal_ex_typecasting<NB, BATCHED, rocblas_half, rocblas_half, float>(
            SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
            && execution_type == rocblas_datatype_f16_r)
    {
        // hscal
        status = scal_ex_typecasting<NB, BATCHED, rocblas_half>(SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f32_r
            && execution_type == rocblas_datatype_f32_r)
    {
        // sscal
        status = scal_ex_typecasting<NB, BATCHED, float>(SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_r && x_type == rocblas_datatype_f64_r
            && execution_type == rocblas_datatype_f64_r)
    {
        // dscal
        status = scal_ex_typecasting<NB, BATCHED, double>(SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_c && x_type == rocblas_datatype_f32_c
            && execution_type == rocblas_datatype_f32_c)
    {
        // cscal
        status = scal_ex_typecasting<NB, BATCHED, rocblas_float_complex>(SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_c && x_type == rocblas_datatype_f64_c
            && execution_type == rocblas_datatype_f64_c)
    {
        // zscal
        status
            = scal_ex_typecasting<NB, BATCHED, rocblas_double_complex>(SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f32_c
            && execution_type == rocblas_datatype_f32_c)
    {
        // csscal
        status
            = scal_ex_typecasting<NB, BATCHED, float, rocblas_float_complex, rocblas_float_complex>(
                SCAL_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_r && x_type == rocblas_datatype_f64_c
            && execution_type == rocblas_datatype_f64_c)
    {
        // zdscal
        status
            = scal_ex_typecasting<NB, BATCHED, float, rocblas_float_complex, rocblas_float_complex>(
                SCAL_EX_TYPECASTING_PARAM);
    }
    else
    {
        status = rocblas_status_not_implemented;
    }

    return status;

#undef SCAL_EX_TYPECASTING_PARAM
}
