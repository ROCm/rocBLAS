/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocblas_arguments.hpp"

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto rocblas_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case rocblas_datatype_f16_r:
        return TEST<rocblas_half>{}(arg);
    case rocblas_datatype_bf16_r:
        return TEST<rocblas_bfloat16>{}(arg);
    case rocblas_datatype_f32_r:
        return TEST<float>{}(arg);
    case rocblas_datatype_f64_r:
        return TEST<double>{}(arg);
    //  case rocblas_datatype_f16_c:
    //      return TEST<rocblas_half_complex>{}(arg);
    case rocblas_datatype_f32_c:
        return TEST<rocblas_float_complex>{}(arg);
    case rocblas_datatype_f64_c:
        return TEST<rocblas_double_complex>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// BLAS1 functions
template <template <typename...> class TEST>
auto rocblas_blas1_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, Tb = arg.b_type, To = arg.d_type;
    const auto Tc = arg.c_type;
    if(Ti == To)
    {
        if(Tb == Ti)
            return rocblas_simple_dispatch<TEST>(arg);
        else
        { // for csscal and zdscal and complex rot/rotg only
            if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r
               && Tc == rocblas_datatype_f32_r)
                return TEST<rocblas_float_complex, float>{}(arg);
            else if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r
                    && Tc == rocblas_datatype_f32_c)
                return TEST<rocblas_float_complex, float, rocblas_float_complex>{}(arg);
            else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                    && Tc == rocblas_datatype_f64_r)
                return TEST<rocblas_double_complex, double>{}(arg);
            else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                    && Tc == rocblas_datatype_f64_c)
                return TEST<rocblas_double_complex, double, rocblas_double_complex>{}(arg);
        }
    }
    else if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r)
        return TEST<rocblas_float_complex, float>{}(arg);
    else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r)
        return TEST<rocblas_double_complex, double>{}(arg);
    else if(Ti == rocblas_datatype_f32_r && Tb == rocblas_datatype_f32_r)
        return TEST<float, float>{}(arg);
    else if(Ti == rocblas_datatype_f64_r && Tb == rocblas_datatype_f64_r)
        return TEST<double, double>{}(arg);

    //  else if(Ti == rocblas_datatype_f16_c && To == rocblas_datatype_f16_r)
    //      return TEST<rocblas_half_complex, rocblas_half>{}(arg);

    return TEST<void>{}(arg);
}

// BLAS1_ex functions
template <template <typename...> class TEST>
auto rocblas_blas1_ex_dispatch(const Arguments& arg)
{
    const auto Ta = arg.a_type, Tx = arg.b_type, Ty = arg.c_type, Tex = arg.compute_type;

    if(Ta == Tx && Tx == Ty && Ty == Tex)
    {
        return rocblas_simple_dispatch<TEST>(arg); // Ta == Tx == Ty == Tex
    }
    else if(Ta == Tx && Tx == Ty && Ta == rocblas_datatype_f16_r && Tex == rocblas_datatype_f32_r)
    {
        return TEST<rocblas_half, rocblas_half, rocblas_half, float>{}(arg);
    }
    else if(Ta == Tx && Tx == Ty && Ta == rocblas_datatype_bf16_r && Tex == rocblas_datatype_f32_r)
    {
        return TEST<rocblas_bfloat16, rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
    }
    else if(Ta == Tx && Ta == rocblas_datatype_f16_r && Tex == rocblas_datatype_f32_r)
    {
        // scal half
        return TEST<rocblas_half, rocblas_half, float>{}(arg);
    }
    else if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_r
            && Tex == rocblas_datatype_f32_c)
    {
        // csscal
        return TEST<float, rocblas_float_complex, rocblas_float_complex>{}(arg);
    }
    else if(Ta == rocblas_datatype_f64_c && Tx == rocblas_datatype_f64_r
            && Tex == rocblas_datatype_f64_c)
    {
        // zdscal
        return TEST<double, rocblas_double_complex, rocblas_double_complex>{}(arg);
    }
    else if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_r
            && Tex == rocblas_datatype_f32_r)
    {
        // scnrm2
        return TEST<rocblas_float_complex, float, float>{}(arg);
    }
    else if(Ta == rocblas_datatype_f64_c && Tx == rocblas_datatype_f64_r
            && Tex == rocblas_datatype_f64_r)
    {
        // dznrm2
        return TEST<rocblas_double_complex, double, double>{}(arg);
    }
    else if(Ta == rocblas_datatype_f16_r && Tx == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        // nrm2 half with float execution
        return TEST<rocblas_half, rocblas_half, float>{}(arg);
    }
    else if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_c
            && Ty == rocblas_datatype_f32_r && Tex == rocblas_datatype_f32_c)
    {
        // rot with complex x/y/compute and real cs
        return TEST<rocblas_float_complex, rocblas_float_complex, float, rocblas_float_complex>{}(
            arg);
    }
    else if(Ta == rocblas_datatype_f64_c && Tx == rocblas_datatype_f64_c
            && Ty == rocblas_datatype_f64_r && Tex == rocblas_datatype_f64_c)
    {
        // rot with complex x/y/compute and real cs
        return TEST<rocblas_double_complex,
                    rocblas_double_complex,
                    double,
                    rocblas_double_complex>{}(arg);
    }

    return TEST<void>{}(arg);
}

// gemm functions
template <template <typename...> class TEST>
auto rocblas_gemm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tc = arg.compute_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti != To)
        {
            if(Ti == rocblas_datatype_i8_r && To == rocblas_datatype_i32_r && Tc == To)
            {
                return TEST<int8_t, int32_t, int32_t>{}(arg);
            }
            else if(To == rocblas_datatype_f32_r && Tc == rocblas_datatype_f32_r)
            {
                if(Ti == rocblas_datatype_f16_r)
                {
                    return TEST<rocblas_half, float, float>{}(arg);
                }
                else if(Ti == rocblas_datatype_bf16_r)
                {
                    return TEST<rocblas_bfloat16, float, float>{}(arg);
                }
            }
        }
        else if(Tc != To)
        {
            if(To == rocblas_datatype_f16_r && Tc == rocblas_datatype_f32_r)
            {
                return TEST<rocblas_half, rocblas_half, float>{}(arg);
            }
            else if(To == rocblas_datatype_bf16_r && Tc == rocblas_datatype_f32_r)
            {
                return TEST<rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
            }
        }
        else
        {
            return rocblas_simple_dispatch<TEST>(arg); // Ti = To = Tc
        }
    }
    return TEST<void>{}(arg);
}
