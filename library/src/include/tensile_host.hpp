/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************/

/*********************************************************
 * Declaration of the rocBLAS<->Tensile interface layer. *
 *********************************************************/
#pragma once
#ifndef __TENSILE_HOST_HPP__
#define __TENSILE_HOST_HPP__

#ifndef USE_TENSILE_HOST
#error "tensile_host.hpp #include'd when USE_TENSILE_HOST is undefined."
#endif

#include "handle.h"

/********************************************************************
 * RocblasContractionProblem captures the arguments for a GEMM-like *
 * contraction problem, to be passed to runContractionProblem.      *
 ********************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
struct RocblasContractionProblem
{
    rocblas_handle    handle;
    rocblas_operation trans_a;
    rocblas_operation trans_b;

    // The size data members should exactly match Tensile's size parameters
    // even if rocBLAS uses smaller or differently-signed types
    size_t    m;
    size_t    n;
    size_t    k;
    const Tc* alpha;
    const Ti* A;
    size_t    ld_a;
    size_t    stride_a;
    const Ti* B;
    size_t    ld_b;
    size_t    stride_b;
    const Tc* beta;
    const To* C;
    size_t    ld_c;
    size_t    stride_c;
    To*       D        = const_cast<To*>(C); // Default D = C
    size_t    ld_d     = ld_c;
    size_t    stride_d = stride_c;
    size_t    batch_count;

    // gemm
    // gemm_strided_batched
    RocblasContractionProblem(rocblas_handle    handle,
                              rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const Tc*         alpha,
                              const Ti*         A,
                              rocblas_int       ld_a,
                              rocblas_stride    stride_a,
                              const Ti*         B,
                              rocblas_int       ld_b,
                              rocblas_stride    stride_b,
                              const Tc*         beta,
                              const To*         C,
                              rocblas_int       ld_c,
                              rocblas_stride    stride_c,
                              rocblas_int       batch_count = 1)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , ld_a(ld_a)
        , stride_a(stride_a)
        , B(B)
        , ld_b(ld_b)
        , stride_b(stride_b)
        , beta(beta)
        , C(C)
        , ld_c(ld_c)
        , stride_c(stride_c)
        , batch_count(batch_count)
    {
    }

    // gemm_ex
    // gemm_strided_batched_ex
    RocblasContractionProblem(rocblas_handle    handle,
                              rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const Tc*         alpha,
                              const Ti*         A,
                              rocblas_int       ld_a,
                              rocblas_stride    stride_a,
                              const Ti*         B,
                              rocblas_int       ld_b,
                              rocblas_stride    stride_b,
                              const Tc*         beta,
                              const To*         C,
                              rocblas_int       ld_c,
                              rocblas_stride    stride_c,
                              To*               D,
                              rocblas_int       ld_d,
                              rocblas_stride    stride_d,
                              rocblas_int       batch_count = 1)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , ld_a(ld_a)
        , stride_a(stride_a)
        , B(B)
        , ld_b(ld_b)
        , stride_b(stride_b)
        , beta(beta)
        , C(C)
        , ld_c(ld_c)
        , stride_c(stride_c)
        , D(D)
        , ld_d(ld_d)
        , stride_d(stride_d)
        , batch_count(batch_count)
    {
    }
};

/********************************************************************************
 * TensileHost is the base class used to represent the interface with Tensile.  *
 * The actual implementation is in TensileHostImpl defined in tensile_host.cpp. *
 ********************************************************************************/
struct TensileHost
{
    // runContractionProblem() is the how a RocblasContractionProblem is run
    template <typename Ti, typename To, typename Tc>
    rocblas_status runContractionProblem(RocblasContractionProblem<Ti, To, Tc> const& problem);

    // Allow the polymorphic deletion of TensileHost
    virtual ~TensileHost() = default;

    // Prevent instantiating this class except as base class
protected:
    TensileHost() = default;
};

/*******************************************************************************
 * createTensileHost() returns an instance of TensileHostImpl as a TensileHost *
 *******************************************************************************/
TensileHost* createTensileHost();

#endif // __TENSILE_HOST_HPP__
