/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
