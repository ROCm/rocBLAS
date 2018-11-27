/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <string>
#include <iostream>
#include "rocblas.h"

// return letter N,T,C in place of rocblas_operation enum
std::string rocblas_transpose_letter(rocblas_operation trans)
{
    if(trans == rocblas_operation_none)
    {
        return "N";
    }
    else if(trans == rocblas_operation_transpose)
    {
        return "T";
    }
    else if(trans == rocblas_operation_conjugate_transpose)
    {
        return "C";
    }
    else
    {
        std::cerr << "rocblas ERROR: trans != N, T, C" << std::endl;
        return " ";
    }
}
// return letter in place of rocblas_side enum
std::string rocblas_side_letter(rocblas_side side)
{
    if(side == rocblas_side_left)
    {
        return "L";
    }
    else if(side == rocblas_side_right)
    {
        return "R";
    }
    else if(side == rocblas_side_both)
    {
        return "B";
    }
    else
    {
        std::cerr << "rocblas ERROR: side != L, R, B" << std::endl;
        return " ";
    }
}
// return letter U, L, B in place of rocblas_fill enum
std::string rocblas_fill_letter(rocblas_fill fill)
{
    if(fill == rocblas_fill_upper)
    {
        return "U";
    }
    else if(fill == rocblas_fill_lower)
    {
        return "L";
    }
    else if(fill == rocblas_fill_full)
    {
        return "F";
    }
    else
    {
        std::cerr << "rocblas ERROR: fill != U, L, B" << std::endl;
        return " ";
    }
}
// return letter N,T,C in place of rocblas_operation enum
std::string rocblas_diag_letter(rocblas_diagonal diag)
{
    if(diag == rocblas_diagonal_non_unit)
    {
        return "N";
    }
    else if(diag == rocblas_diagonal_unit)
    {
        return "U";
    }
    else
    {
        std::cerr << "rocblas ERROR: diag != N, U" << std::endl;
        return " ";
    }
}
// return letter h, s, d, k, c, z in place of rocblas_datatype
std::string rocblas_datatype_letter(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r: return "h";
    case rocblas_datatype_f32_r: return "s";
    case rocblas_datatype_f64_r: return "d";
    case rocblas_datatype_f16_c: return "k";
    case rocblas_datatype_f32_c: return "c";
    case rocblas_datatype_f64_c: return "z";
    case rocblas_datatype_i8_r: return "i8r";
    case rocblas_datatype_u8_r: return "u8r";
    case rocblas_datatype_i32_r: return "i32r";
    case rocblas_datatype_u32_r: return "u32r";
    case rocblas_datatype_i8_c: return "i8c";
    case rocblas_datatype_u8_c: return "u8c";
    case rocblas_datatype_i32_c: return "i32c";
    case rocblas_datatype_u32_c: return "u32c";
    default:
        std::cerr << "rocblas ERROR: unsupported datatype (" << type << ")" << std::endl;
        return " ";
    }
}
