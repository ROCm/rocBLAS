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
