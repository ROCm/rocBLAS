/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_DATATYPE2CHAR_H_
#define ROCBLAS_DATATYPE2CHAR_H_

#include "rocblas.h"
#include <string>

enum class rocblas_initialization
{
    rand_int   = 111,
    trig_float = 222,
    hpl        = 333,
};

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

constexpr auto rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none:
        return 'N';
    case rocblas_operation_transpose:
        return 'T';
    case rocblas_operation_conjugate_transpose:
        return 'C';
    }
    return '\0';
}

constexpr auto rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper:
        return 'U';
    case rocblas_fill_lower:
        return 'L';
    case rocblas_fill_full:
        return 'F';
    }
    return '\0';
}

constexpr auto rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit:
        return 'U';
    case rocblas_diagonal_non_unit:
        return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left:
        return 'L';
    case rocblas_side_right:
        return 'R';
    case rocblas_side_both:
        return 'B';
    }
    return '\0';
}

// return precision string for rocblas_datatype
constexpr auto rocblas_datatype2string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return "f16_r";
    case rocblas_datatype_f32_r:
        return "f32_r";
    case rocblas_datatype_f64_r:
        return "f64_r";
    case rocblas_datatype_f16_c:
        return "f16_k";
    case rocblas_datatype_f32_c:
        return "f32_c";
    case rocblas_datatype_f64_c:
        return "f64_c";
    case rocblas_datatype_i8_r:
        return "i8_r";
    case rocblas_datatype_u8_r:
        return "u8_r";
    case rocblas_datatype_i32_r:
        return "i32_r";
    case rocblas_datatype_u32_r:
        return "u32_r";
    case rocblas_datatype_i8_c:
        return "i8_c";
    case rocblas_datatype_u8_c:
        return "u8_c";
    case rocblas_datatype_i32_c:
        return "i32_c";
    case rocblas_datatype_u32_c:
        return "u32_c";
    case rocblas_datatype_bf16_r:
        return "bf16_r";
    case rocblas_datatype_bf16_c:
        return "bf16_c";
    }
    return "invalid";
}

constexpr auto rocblas_initialization2string(rocblas_initialization init)
{
    switch(init)
    {
    case rocblas_initialization::rand_int:
        return "rand_int";
    case rocblas_initialization::trig_float:
        return "trig_float";
    case rocblas_initialization::hpl:
        return "hpl";
    }
    return "invalid";
}

inline rocblas_ostream& operator<<(rocblas_ostream& os, rocblas_initialization init)
{
    return os << rocblas_initialization2string(init);
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return rocblas_operation_none;
    case 'T':
    case 't':
        return rocblas_operation_transpose;
    case 'C':
    case 'c':
        return rocblas_operation_conjugate_transpose;
    default:
        return static_cast<rocblas_operation>(-1);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_fill_upper;
    case 'L':
    case 'l':
        return rocblas_fill_lower;
    default:
        return static_cast<rocblas_fill>(-1);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_diagonal_unit;
    case 'N':
    case 'n':
        return rocblas_diagonal_non_unit;
    default:
        return static_cast<rocblas_diagonal>(-1);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l':
        return rocblas_side_left;
    case 'R':
    case 'r':
        return rocblas_side_right;
    default:
        return static_cast<rocblas_side>(-1);
    }
}

// clang-format off
inline rocblas_initialization string2rocblas_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? rocblas_initialization::rand_int   :
        value == "trig_float" ? rocblas_initialization::trig_float :
        value == "hpl"        ? rocblas_initialization::hpl        :
        static_cast<rocblas_initialization>(-1);
}

inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r  :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r  :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r  :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c  :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c  :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c  :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r   :
        value == "i32_r"                 ? rocblas_datatype_i32_r  :
        value == "i8_c"                  ? rocblas_datatype_i8_c   :
        value == "i32_c"                 ? rocblas_datatype_i32_c  :
        value == "u8_r"                  ? rocblas_datatype_u8_r   :
        value == "u32_r"                 ? rocblas_datatype_u32_r  :
        value == "u8_c"                  ? rocblas_datatype_u8_c   :
        value == "u32_c"                 ? rocblas_datatype_u32_c  :
        static_cast<rocblas_datatype>(-1);
}
// clang-format on

#endif
