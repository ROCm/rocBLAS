/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../blas1/rocblas_axpy.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_axpy_ex.hpp"

template <int NB, bool BATCHED, typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
rocblas_status axpy_ex_typecasting(const char*    name,
                                   rocblas_handle handle,
                                   rocblas_int    n,
                                   const void*    alpha,
                                   rocblas_stride stride_alpha,
                                   const void*    x,
                                   ptrdiff_t      offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   void*          y,
                                   ptrdiff_t      offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   rocblas_int    batch_count)
{
    auto check_numerics = handle->check_numerics;

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
        //Checking input batched vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status axpy_ex_check_numerics_status
                = rocblas_axpy_check_numerics(name,
                                              handle,
                                              n,
                                              (const Tx* const*)x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              (Ty* const*)y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_axpy_template<NB, Tex>(handle,
                                                                        n,
                                                                        alphat,
                                                                        stride_alpha,
                                                                        (const Tx* const*)x,
                                                                        offset_x,
                                                                        incx,
                                                                        stride_x,
                                                                        (Ty* const*)y,
                                                                        offset_y,
                                                                        incy,
                                                                        stride_y,
                                                                        batch_count);
        if(status != rocblas_status_success)
            return status;

        //Checking output batched vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status axpy_ex_check_numerics_status
                = rocblas_axpy_check_numerics(name,
                                              handle,
                                              n,
                                              (const Tx* const*)x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              (Ty* const*)y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }
        return status;
    }
    else
    {
        //Checking input vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status axpy_ex_check_numerics_status
                = rocblas_axpy_check_numerics(name,
                                              handle,
                                              n,
                                              (const Tx*)x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              (Ty*)y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_axpy_template<NB, Tex>(handle,
                                                                        n,
                                                                        alphat,
                                                                        stride_alpha,
                                                                        (const Tx*)x,
                                                                        offset_x,
                                                                        incx,
                                                                        stride_x,
                                                                        (Ty*)y,
                                                                        offset_y,
                                                                        incy,
                                                                        stride_y,
                                                                        batch_count);

        if(status != rocblas_status_success)
            return status;

        //Checking output vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status axpy_ex_check_numerics_status
                = rocblas_axpy_check_numerics(name,
                                              handle,
                                              n,
                                              (const Tx*)x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              (Ty*)y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }
        return status;
    }
}

template <int NB, bool BATCHED>
rocblas_status rocblas_axpy_ex_template(const char*      name,
                                        rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        rocblas_stride   stride_alpha,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        ptrdiff_t        offset_x,
                                        rocblas_int      incx,
                                        rocblas_stride   stride_x,
                                        void*            y,
                                        rocblas_datatype y_type,
                                        ptrdiff_t        offset_y,
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

#define AXPY_EX_TYPECASTING_PARAM                                                         \
    name, handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, y, offset_y, incy, \
        stride_y, batch_count

    if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
       && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = axpy_ex_typecasting<NB, BATCHED, rocblas_half, rocblas_half, rocblas_half, float>(
            AXPY_EX_TYPECASTING_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f16_r
            && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = axpy_ex_typecasting<NB, BATCHED, float, rocblas_half, rocblas_half, float>(
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

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *axpy_ex*.cpp

// clang-format off

#ifdef INSTANTIATE_AXPY_EX_TEMPLATE
#error INSTANTIATE_AXPY_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_AXPY_EX_TEMPLATE(NB, BATCHED)                         \
template rocblas_status rocblas_axpy_ex_template<NB, BATCHED>             \
                                       (const char*      name,            \
                                        rocblas_handle   handle,          \
                                        rocblas_int      n,               \
                                        const void*      alpha,           \
                                        rocblas_datatype alpha_type,      \
                                        rocblas_stride   stride_alpha,    \
                                        const void*      x,               \
                                        rocblas_datatype x_type,          \
                                        ptrdiff_t        offset_x,        \
                                        rocblas_int      incx,            \
                                        rocblas_stride   stride_x,        \
                                        void*            y,               \
                                        rocblas_datatype y_type,          \
                                        ptrdiff_t        offset_y,        \
                                        rocblas_int      incy,            \
                                        rocblas_stride   stride_y,        \
                                        rocblas_int      batch_count,     \
                                        rocblas_datatype execution_type);

INSTANTIATE_AXPY_EX_TEMPLATE(256, false)
INSTANTIATE_AXPY_EX_TEMPLATE(256, true)

#undef INSTANTIATE_AXPY_EX_TEMPLATE

// clang-format on
