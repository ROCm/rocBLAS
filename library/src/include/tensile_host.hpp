/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************/

/*********************************************************
 * Declaration of the rocBLAS<->Tensile interface layer. *
 *********************************************************/
#ifndef __TENSILE_HOST_HPP__
#define __TENSILE_HOST_HPP__

#ifndef USE_TENSILE_HOST
#error "tensile_host.hpp #include'd when USE_TENSILE_HOST is undefined."
#endif

/*****************************************************************************
 * WARNING: Tensile-specific data types, functions and macros should only be *
 * referenced from tensile_host.cpp. This header file defines the interface  *
 * that the rest of rocBLAS uses to access Tensile. If another Tensile       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any Tensile-specific identifiers here.  *
 *****************************************************************************/

#include "handle.h"
#include "tuple_helper.hpp"

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

    // The RocblasContractionProblem data members should exactly match
    // Tensile's parameter types, even if rocBLAS uses differently
    // sized or signed types. The constructors should convert rocBLAS
    // types into the corresponding Tensile types stored in this class.
    size_t m;
    size_t n;
    size_t k;

    const Tc* alpha;

    const Ti* A;
    size_t    row_stride_a;
    size_t    col_stride_a;
    size_t    batch_stride_a;

    const Ti* B;
    size_t    row_stride_b;
    size_t    col_stride_b;
    size_t    batch_stride_b;

    const Tc* beta;

    const To* C;
    size_t    row_stride_c;
    size_t    col_stride_c;
    size_t    batch_stride_c;

    To*    D;
    size_t row_stride_d;
    size_t col_stride_d;
    size_t batch_stride_d;

    size_t batch_count;

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
                              rocblas_stride    batch_stride_a,
                              const Ti*         B,
                              rocblas_int       ld_b,
                              rocblas_stride    batch_stride_b,
                              const Tc*         beta,
                              To*               C,
                              rocblas_int       ld_c,
                              rocblas_stride    batch_stride_c,
                              rocblas_int       batch_count)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , B(B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , beta(beta)
        , C(C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , D(C)
        , row_stride_d(1)
        , col_stride_d(ld_c)
        , batch_stride_d(batch_stride_c)
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
                              rocblas_stride    batch_stride_a,
                              const Ti*         B,
                              rocblas_int       ld_b,
                              rocblas_stride    batch_stride_b,
                              const Tc*         beta,
                              const To*         C,
                              rocblas_int       ld_c,
                              rocblas_stride    batch_stride_c,
                              To*               D,
                              rocblas_int       ld_d,
                              rocblas_stride    batch_stride_d,
                              rocblas_int       batch_count)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , B(B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , beta(beta)
        , C(C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , D(D)
        , row_stride_d(1)
        , col_stride_d(ld_d)
        , batch_stride_d(batch_stride_d)
        , batch_count(batch_count)
    {
    }

    // gemm_ext2
    // gemm_strided_batched_ext2
    RocblasContractionProblem(rocblas_handle handle,
                              rocblas_int    m,
                              rocblas_int    n,
                              rocblas_int    k,
                              const Tc*      alpha,
                              const Ti*      A,
                              rocblas_stride row_stride_a,
                              rocblas_stride col_stride_a,
                              rocblas_stride batch_stride_a,
                              const Ti*      B,
                              rocblas_stride row_stride_b,
                              rocblas_stride col_stride_b,
                              rocblas_stride batch_stride_b,
                              const Tc*      beta,
                              const To*      C,
                              rocblas_stride row_stride_c,
                              rocblas_stride col_stride_c,
                              rocblas_stride batch_stride_c,
                              To*            D,
                              rocblas_stride row_stride_d,
                              rocblas_stride col_stride_d,
                              rocblas_stride batch_stride_d,
                              rocblas_int    batch_count)
        : handle(handle)
        , trans_a(rocblas_operation_none)
        , trans_b(rocblas_operation_none)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , row_stride_a(row_stride_a)
        , col_stride_a(col_stride_a)
        , batch_stride_a(batch_stride_a)
        , B(B)
        , row_stride_b(row_stride_b)
        , col_stride_b(col_stride_b)
        , batch_stride_b(batch_stride_b)
        , beta(beta)
        , C(C)
        , row_stride_c(row_stride_c)
        , col_stride_c(col_stride_c)
        , batch_stride_c(batch_stride_c)
        , D(D)
        , row_stride_d(row_stride_d)
        , col_stride_d(col_stride_d)
        , batch_stride_d(batch_stride_d)
        , batch_count(batch_count)
    {
    }

    /***************************************************
     * Print a RocblasContractionProblem for debugging *
     ***************************************************/
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const RocblasContractionProblem& prob)
    {
        return tuple_helper::print_tuple_pairs(
            os,
            std::make_tuple("a_type",
                            rocblas_precision_string<Ti>,
                            "b_type",
                            rocblas_precision_string<Ti>,
                            "c_type",
                            rocblas_precision_string<To>,
                            "d_type",
                            rocblas_precision_string<To>,
                            "compute_type",
                            rocblas_precision_string<Tc>,
                            "transA",
                            rocblas_transpose_letter(prob.trans_a),
                            "transB",
                            rocblas_transpose_letter(prob.trans_b),
                            "M",
                            prob.m,
                            "N",
                            prob.n,
                            "K",
                            prob.k,
                            "alpha",
                            *prob.alpha,
                            "row_stride_a",
                            prob.row_stride_a,
                            "col_stride_a",
                            prob.col_stride_a,
                            "row_stride_b",
                            prob.row_stride_b,
                            "col_stride_b",
                            prob.col_stride_b,
                            "row_stride_c",
                            prob.row_stride_c,
                            "col_stride_c",
                            prob.col_stride_c,
                            "row_stride_d",
                            prob.row_stride_d,
                            "col_stride_d",
                            prob.col_stride_d,
                            "beta",
                            *prob.beta,
                            "batch_count",
                            prob.batch_count,
                            "stride_a",
                            prob.batch_stride_a,
                            "stride_b",
                            prob.batch_stride_b,
                            "stride_c",
                            prob.batch_stride_c,
                            "stride_d",
                            prob.batch_stride_d,
                            "atomics mode",
                            prob.handle->atomics_mode));
    };
};

/*******************************************************************************
 * runContractionProblem() solves a RocblasContractionProblem                  *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status runContractionProblem(RocblasContractionProblem<Ti, To, Tc> const& problem);

#endif // __TENSILE_HOST_HPP__
