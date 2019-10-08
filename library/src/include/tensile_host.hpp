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
class RocblasContractionProblem
{
public:
    ContractionProblemType problem_type;
    rocblas_operation      trans_a;
    rocblas_operation      trans_b;
    unsigned long          m;
    unsigned long          n;
    unsigned long          k;
    const T*               alpha;
    const T*               A;
    const unsigned long    ld_a;
    unsigned long          stride_a;
    const T*               B;
    unsigned long          ld_b;
    unsigned long          stride_b;
    const T*               beta;
    T*                     C;
    unsigned long          ld_c;
    unsigned long          stride_c;
    unsigned long          batch_size;

    RocblasContractionProblem(ContractionProblemType problem_type,
                              rocblas_operation      trans_a,
                              rocblas_operation      trans_b,
                              unsigned long          m,
                              unsigned long          n,
                              unsigned long          k,
                              const T*               alpha,
                              const T*               A,
                              unsigned long          ld_a,
                              const T*               B,
                              unsigned long          ld_b,
                              const T*               beta,
                              T*                     C,
                              unsigned long          ld_c)
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
                              unsigned long          m,
                              unsigned long          n,
                              unsigned long          k,
                              const T*               alpha,
                              const T*               A,
                              unsigned long          ld_a,
                              unsigned long          stride_a,
                              const T*               B,
                              unsigned long          ld_b,
                              unsigned long          stride_b,
                              const T*               beta,
                              T*                     C,
                              unsigned long          ld_c,
                              unsigned long          stride_c,
                              unsigned long          batch_size)
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

class TensileHost
{
public:
    virtual void initializeHost(const char*) {}
};

template <typename T>
class TensileHostCall
{
public:
    rocblas_status runContractionProblem(RocblasContractionProblem<T>* problem, TensileHost* host);
};

TensileHost* createTensileHost();

rocblas_status callTensileContraction_half(RocblasContractionProblem<rocblas_half>* problem,
                                           TensileHost*                             host);
rocblas_status callTensileContraction_float(RocblasContractionProblem<float>* problem,
                                            TensileHost*                      host);
rocblas_status callTensileContraction_double(RocblasContractionProblem<double>* problem,
                                             TensileHost*                       host);
rocblas_status
               callTensileContraction_float_complex(RocblasContractionProblem<rocblas_float_complex>* problem,
                                                    TensileHost*                                      host);
rocblas_status callTensileContraction_double_complex(
    RocblasContractionProblem<rocblas_double_complex>* problem, TensileHost* host);

#endif

#endif // __TENSILE_HOST_HPP__
