/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas.h"

template <typename T, typename U>
__device__ void herk_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
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
            e *= beta;
        }
        else if(from == to)
        {
            // multiply only real component and zero imaginary on diagonal
            e = {e.real() * beta, 0};
        }
    }
}
