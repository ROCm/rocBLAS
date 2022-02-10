/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocblas_arguments.hpp"

template <typename T>
constexpr auto rocblas_type2datatype()
{
    if(std::is_same<T, rocblas_half>{})
        return rocblas_datatype_f16_r;
    if(std::is_same<T, rocblas_bfloat16>{})
        return rocblas_datatype_bf16_r;
    if(std::is_same<T, rocblas_float>{})
        return rocblas_datatype_f32_r;
    if(std::is_same<T, rocblas_double>{})
        return rocblas_datatype_f64_r;
    // if(std::is_same<T, rocblas_half_complex>{})
    //     return rocblas_datatype_f16_c;
    if(std::is_same<T, rocblas_float_complex>{})
        return rocblas_datatype_f32_c;
    if(std::is_same<T, rocblas_double_complex>{})
        return rocblas_datatype_f64_c;
    if(std::is_same<T, char>{})
        return rocblas_datatype_i8_r;
    if(std::is_same<T, unsigned char>{})
        return rocblas_datatype_u8_r;

    return rocblas_datatype_f32_r; // testing purposes we default to f32 ex
}

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
    // case rocblas_datatype_f16_c:
    //     return TEST<rocblas_half_complex>{}(arg);
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
            else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                    && Tc == rocblas_datatype_f64_r)
                return TEST<rocblas_double_complex, double>{}(arg);
            else if(strstr(arg.function, "scal"))
            {
                // for csscal and zdscal
                if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r
                   && Tc == rocblas_datatype_f32_c)
                    return TEST<rocblas_float_complex, float>{}(arg);
                else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                        && Tc == rocblas_datatype_f64_c)
                    return TEST<rocblas_double_complex, double>{}(arg);
            }
            else if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r
                    && Tc == rocblas_datatype_f32_c)
                return TEST<rocblas_float_complex, float, rocblas_float_complex>{}(arg);
            else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                    && Tc == rocblas_datatype_f64_c)
                return TEST<rocblas_double_complex, double, rocblas_double_complex>{}(arg);
        }
    }
    //
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
    const auto        Ta = arg.a_type, Tx = arg.b_type, Ty = arg.c_type, Tex = arg.compute_type;
    const std::string function = arg.function;
    const bool        is_axpy  = function == "axpy_ex" || function == "axpy_batched_ex"
                         || function == "axpy_strided_batched_ex";
    const bool is_dot = function == "dot_ex" || function == "dot_batched_ex"
                        || function == "dot_strided_batched_ex" || function == "dotc_ex"
                        || function == "dotc_batched_ex" || function == "dotc_strided_batched_ex";
    const bool is_nrm2 = function == "nrm2_ex" || function == "nrm2_batched_ex"
                         || function == "nrm2_strided_batched_ex";
    const bool is_rot = function == "rot_ex" || function == "rot_batched_ex"
                        || function == "rot_strided_batched_ex";
    const bool is_scal = function == "scal_ex" || function == "scal_batched_ex"
                         || function == "scal_strided_batched_ex";

    if(Ta == Tx && Tx == Ty && Ty == Tex)
    {
        return rocblas_simple_dispatch<TEST>(arg); // Ta == Tx == Ty == Tex
    }
    else if(is_scal && Ta == Tx && Tx == Tex)
    {
        // hscal with f16_r compute (scal doesn't care about Ty)
        return rocblas_simple_dispatch<TEST>(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        return TEST<rocblas_half, rocblas_half, rocblas_half, float>{}(arg);
    }
    else if((is_rot || is_dot) && Ta == Tx && Tx == Ty && Ta == rocblas_datatype_bf16_r
            && Tex == rocblas_datatype_f32_r)
    {
        return TEST<rocblas_bfloat16, rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && Tx == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        return TEST<float, rocblas_half, rocblas_half, float>{}(arg);
    }
    else if((is_scal || is_nrm2 || is_axpy) && Ta == Tx && Ta == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        // half scal, nrm2, axpy
        return TEST<rocblas_half, rocblas_half, float>{}(arg);
    }
    // exclusive functions cases
    else if(is_scal)
    {
        // scal_ex ordering: <alphaType, dataType, exType> opposite order of scal test

        if(Ta == rocblas_datatype_f32_r && Tx == rocblas_datatype_f16_r
           && Tex == rocblas_datatype_f32_r)
        {
            // scal half with float alpha
            return TEST<float, rocblas_half, float>{}(arg);
        }
        else if(Ta == rocblas_datatype_f32_r && Tx == rocblas_datatype_f32_c
                && Tex == rocblas_datatype_f32_c)
        {
            // csscal-like
            return TEST<float, rocblas_float_complex, rocblas_float_complex>{}(arg);
        }
        else if(Ta == rocblas_datatype_f64_r && Tx == rocblas_datatype_f64_c
                && Tex == rocblas_datatype_f64_c)
        {
            // zdscal-like
            return TEST<double, rocblas_double_complex, rocblas_double_complex>{}(arg);
        }
    }
    else if(is_nrm2)
    {
        if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_r
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
    }
    else if(is_rot)
    {
        if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_c
           && Ty == rocblas_datatype_f32_r && Tex == rocblas_datatype_f32_c)
        {
            // rot with complex x/y/compute and real cs
            return TEST<rocblas_float_complex,
                        rocblas_float_complex,
                        float,
                        rocblas_float_complex>{}(arg);
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
    }

    return TEST<void>{}(arg);
}

// gemm functions
template <template <typename...> class TEST>
auto rocblas_gemm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tc = arg.compute_type;

    if((rocblas_gemm_flags_fp16_alt_impl & arg.flags)
       && rocblas_internal_get_arch_name() != "gfx90a")
        return TEST<void>{}(arg);

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti != To)
        {
            // TODO- Maybe we chould add a new datatype_enum such as rocblas_datatype_i8x4_r
            // So that we could go to the correct branch here.
            // So far, using whether int8_t or int8x4 is determined in TEST function (gemm_ex)
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
