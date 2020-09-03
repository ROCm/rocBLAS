/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "../blas1/rocblas_axpy.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <int NB, bool BATCHED, typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
rocblas_status axpy_ex_typecasting(rocblas_handle handle,
                                   rocblas_int    n,
                                   const void*    alpha,
                                   const void*    x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   void*          y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   rocblas_int    batch_count)
{
    const Ta* alphat = (const Ta*)alpha;
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alphat == 0)
            return rocblas_status_success;
    }

    if(!x || !y)
        return rocblas_status_invalid_pointer;

    if(BATCHED)
    {
        return rocblas_axpy_template<NB, Tex>(handle,
                                              n,
                                              alphat,
                                              (const Tx* const*)x,
                                              incx,
                                              stride_x,
                                              (Ty* const*)y,
                                              incy,
                                              stride_y,
                                              batch_count);
    }
    else
    {
        return rocblas_axpy_template<NB, Tex>(
            handle, n, alphat, (const Tx*)x, incx, stride_x, (Ty*)y, incy, stride_y, batch_count);
    }
}

template <int NB, bool BATCHED = false>
rocblas_status rocblas_axpy_ex_template(rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        rocblas_int      incx,
                                        rocblas_stride   stride_x,
                                        void*            y,
                                        rocblas_datatype y_type,
                                        rocblas_int      incy,
                                        rocblas_stride   stride_y,
                                        rocblas_int      batch_count,
                                        rocblas_datatype execution_type)
{
    // Error checking
    if(n <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    if(!alpha)
        return rocblas_status_invalid_pointer;

    // Quick return (alpha == 0) check and other nullptr checks will be done
    // once we know the type (in axpy_ex_typecasting).

    rocblas_status status = rocblas_status_not_implemented;

#define AXPY_EX_TYPECASTING_PARAM \
    handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count

    if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
       && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = axpy_ex_typecasting<NB, BATCHED, rocblas_half, rocblas_half, rocblas_half, float>(
            AXPY_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
            && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f16_r)
    {
        status = axpy_ex_typecasting<NB, BATCHED, rocblas_half>(AXPY_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f32_r
            && y_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_r)
    {
        status = axpy_ex_typecasting<NB, BATCHED, float>(AXPY_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_r && x_type == rocblas_datatype_f64_r
            && y_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        status = axpy_ex_typecasting<NB, BATCHED, double>(AXPY_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_c && x_type == rocblas_datatype_f32_c
            && y_type == rocblas_datatype_f32_c && execution_type == rocblas_datatype_f32_c)
    {
        status = axpy_ex_typecasting<NB, BATCHED, rocblas_float_complex>(AXPY_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_c && x_type == rocblas_datatype_f64_c
            && y_type == rocblas_datatype_f64_c && execution_type == rocblas_datatype_f64_c)
    {
        status
            = axpy_ex_typecasting<NB, BATCHED, rocblas_double_complex>(AXPY_EX_TYPECASTING_PARAM);
    }
    else
    {
        status = rocblas_status_not_implemented;
    }

    return status;

#undef AXPY_EX_TYPECASTING_PARAM
}
