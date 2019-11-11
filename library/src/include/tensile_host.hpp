/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************/

// Declaration of the rocBLAS<->Tensile interface layer.

#pragma once
#ifndef __TENSILE_HOST_HPP__
#define __TENSILE_HOST_HPP__

#ifndef USE_TENSILE_HOST
#error "tensile_host.hpp #include'd when USE_TENSILE_HOST is undefined."
#endif

#include "handle.h"

enum struct ContractionProblemType
{
    gemm,
    gemm_strided_batched,
    gemm_ex,
    gemm_strided_batched_ex,
};

// RocblasContractionProblem captures the arguments for a GEMM-like contraction problem, to be
// passed to runContractionProblem.
template <typename T, typename U = T, typename V = T>
struct RocblasContractionProblem
{
    ContractionProblemType problem_type;
    rocblas_operation      trans_a;
    rocblas_operation      trans_b;
    rocblas_int            m;
    rocblas_int            n;
    rocblas_int            k;
    const T                alpha;
    const T*               A;
    rocblas_int            ld_a;
    rocblas_stride         stride_a{0};
    const T*               B;
    rocblas_int            ld_b;
    rocblas_stride         stride_b{0};
    const T                beta;
    T*                     C;
    rocblas_int            ld_c;
    rocblas_stride         stride_c{0};
    rocblas_int            batch_count{1};

    RocblasContractionProblem(rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const T           alpha,
                              const T*          A,
                              rocblas_int       ld_a,
                              const T*          B,
                              rocblas_int       ld_b,
                              const T           beta,
                              T*                C,
                              rocblas_int       ld_c)
        : problem_type{ContractionProblemType::gemm}
        , trans_a{trans_a}
        , trans_b{trans_b}
        , m{m}
        , n{n}
        , k{k}
        , alpha{alpha}
        , A{A}
        , ld_a{ld_a}
        , B{B}
        , ld_b{ld_b}
        , beta{beta}
        , C{C}
        , ld_c{ld_c}
    {
    }

    RocblasContractionProblem(rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const T           alpha,
                              const T*          A,
                              rocblas_int       ld_a,
                              rocblas_stride    stride_a,
                              const T*          B,
                              rocblas_int       ld_b,
                              rocblas_stride    stride_b,
                              const T           beta,
                              T*                C,
                              rocblas_int       ld_c,
                              rocblas_stride    stride_c,
                              rocblas_int       batch_count)
        : problem_type{ContractionProblemType::gemm_strided_batched}
        , trans_a{trans_a}
        , trans_b{trans_b}
        , m{m}
        , n{n}
        , k{k}
        , alpha{alpha}
        , A{A}
        , ld_a{ld_a}
        , B{B}
        , stride_a{stride_a}
        , ld_b{ld_b}
        , stride_b{stride_b}
        , beta{beta}
        , C{C}
        , ld_c{ld_c}
        , stride_c{stride_c}
        , batch_count{batch_count}
    {
    }
};

// TensileHost is the base class used to represent the interface with Tensile.
// The actual implementation is in TensileHostImpl defined in tensile_host.cpp.
struct TensileHost
{
    template <typename T, typename U, typename V>
    rocblas_status runContractionProblem(const RocblasContractionProblem<T, U, V>& problem);

    virtual ~TensileHost() = default; // Allow the polymorphic deletion of TensileHost

protected:
    TensileHost() = default; // Prevent instantiating this class except as base class
};

// createTensileHost() returns an instance of TensileHostImpl as a TensileHost
TensileHost* createTensileHost();

#endif // __TENSILE_HOST_HPP__
