#pragma once
#ifndef __TENSILE_HOST_HPP__
#define __TENSILE_HOST_HPP__

#ifdef USE_TENSILE_HOST
#include "rocblas.h"

enum ContractionProblemType
{
    GEMM,
    GEMMStridedBatch,
};

template <typename T>
struct RocblasContractionProblem
{
    ContractionProblemType problem_type;
    rocblas_operation      trans_a;
    rocblas_operation      trans_b;
    size_t                 m;
    size_t                 n;
    size_t                 k;
    const T                alpha;
    const T*               A;
    size_t                 ld_a;
    size_t                 stride_a;
    const T*               B;
    size_t                 ld_b;
    size_t                 stride_b;
    const T                beta;
    T*                     C;
    size_t                 ld_c;
    size_t                 stride_c;
    size_t                 batch_size;

    RocblasContractionProblem(ContractionProblemType problem_type,
                              rocblas_operation      trans_a,
                              rocblas_operation      trans_b,
                              size_t                 m,
                              size_t                 n,
                              size_t                 k,
                              const T                alpha,
                              const T*               A,
                              size_t                 ld_a,
                              const T*               B,
                              size_t                 ld_b,
                              const T                beta,
                              T*                     C,
                              size_t                 ld_c)
        : problem_type(problem_type)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , ld_a(ld_a)
        , B(B)
        , stride_a(1)
        , ld_b(ld_b)
        , stride_b(1)
        , beta(beta)
        , C(C)
        , ld_c(ld_c)
        , stride_c(1)
        , batch_size(1)
    {
    }

    RocblasContractionProblem(ContractionProblemType problem_type,
                              rocblas_operation      trans_a,
                              rocblas_operation      trans_b,
                              size_t                 m,
                              size_t                 n,
                              size_t                 k,
                              const T                alpha,
                              const T*               A,
                              size_t                 ld_a,
                              size_t                 stride_a,
                              const T*               B,
                              size_t                 ld_b,
                              size_t                 stride_b,
                              const T                beta,
                              T*                     C,
                              size_t                 ld_c,
                              size_t                 stride_c,
                              size_t                 batch_size)
        : problem_type(problem_type)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , ld_a(ld_a)
        , B(B)
        , stride_a(stride_a)
        , ld_b(ld_b)
        , stride_b(stride_b)
        , beta(beta)
        , C(C)
        , ld_c(ld_c)
        , stride_c(stride_c)
        , batch_size(batch_size)
    {
    }
};

struct TensileHost
{
    template <typename T>
    rocblas_status runContractionProblem(const RocblasContractionProblem<T>& problem);

protected:
    TensileHost() = default; // Prevent instantiating this class except as base class
};

TensileHost* createTensileHost();

#endif

#endif // __TENSILE_HOST_HPP__
