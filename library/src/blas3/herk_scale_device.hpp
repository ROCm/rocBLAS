/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "macros.hpp"
#include "rocblas/rocblas.h"

template <typename T, typename U>
ROCBLAS_KERNEL_ILF void herk_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n)
    {
        auto& e = C[ty * ldc + tx];
        if(from < to)
        {
            e = beta ? beta * e : 0;
        }
        else if(from == to)
        {
            // multiply only real component and zero imaginary on diagonal
            e = {beta ? e.real() * beta : 0, 0};
        }
    }
}
