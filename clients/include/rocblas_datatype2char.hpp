/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_DATATYPE2CHAR_H_
#define ROCBLAS_DATATYPE2CHAR_H_

#include "rocblas.h"

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

constexpr char rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none: return 'N';
    case rocblas_operation_transpose: return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return '\0';
}

constexpr char rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full: return 'F';
    }
    return '\0';
}

constexpr char rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit: return 'U';
    case rocblas_diagonal_non_unit: return 'N';
    }
    return '\0';
}

constexpr char rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left: return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both: return 'B';
    }
    return '\0';
}

constexpr char rocblas_datatype2char(rocblas_datatype value)
{
    switch(value)
    {
    case rocblas_datatype_f16_r: return 'h';
    case rocblas_datatype_f32_r: return 's';
    case rocblas_datatype_f64_r: return 'd';
    case rocblas_datatype_f16_c: return 'k';
    case rocblas_datatype_f32_c: return 'c';
    case rocblas_datatype_f64_c: return 'z';
    default:
        return 'e'; // todo, handle integer types
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n': return rocblas_operation_none;
    case 'T':
    case 't': return rocblas_operation_transpose;
    case 'C':
    case 'c': return rocblas_operation_conjugate_transpose;
    default: return static_cast<rocblas_operation>(-1);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_fill_upper;
    case 'L':
    case 'l': return rocblas_fill_lower;
    default: return static_cast<rocblas_fill>(-1);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_diagonal_unit;
    case 'N':
    case 'n': return rocblas_diagonal_non_unit;
    default: return static_cast<rocblas_diagonal>(-1);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l': return rocblas_side_left;
    case 'R':
    case 'r': return rocblas_side_right;
    default: return static_cast<rocblas_side>(-1);
    }
}

constexpr rocblas_datatype char2rocblas_datatype(char value)
{
    switch(value)
    {
    case 'H':
    case 'h': return rocblas_datatype_f16_r;
    case 'S':
    case 's': return rocblas_datatype_f32_r;
    case 'D':
    case 'd': return rocblas_datatype_f64_r;
    case 'C':
    case 'c': return rocblas_datatype_f32_c;
    case 'Z':
    case 'z': return rocblas_datatype_f64_c;
    default: return static_cast<rocblas_datatype>(-1);
    }
}

rocblas_datatype string2rocblas_datatype(std::string value);

#endif
