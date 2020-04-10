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

/**************************************************************************
 * Return the value category for a value, as a double precision value,    *
 * such as whether it's 0, 1, or some other value. Tensile uses a double  *
 * precision value to express the category of beta. This function is to   *
 * convert complex or other types to a double representing the category.  *
 **************************************************************************/
template <typename T>
constexpr double value_category(const T& beta)
{
    return beta == T(0) ? 0.0 : beta == T(1) ? 1.0 : -12345.0;
}

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
    size_t m;
    size_t n;
    size_t k;

    const Tc* alpha;

    const Ti* A;
    size_t    a_row_stride;
    size_t    a_col_stride;
    size_t    batch_stride_a;

    const Ti* B;
    size_t    b_row_stride;
    size_t    b_col_stride;
    size_t    batch_stride_b;

    const Tc* beta;

    const To* C;
    size_t    c_row_stride;
    size_t    c_col_stride;
    size_t    batch_stride_c;

    To*    D;
    size_t d_row_stride;
    size_t d_col_stride;
    size_t batch_stride_d;

    size_t batch_count;

    // Functions to print RocblasContractionProblem out to stream in YAML format
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const RocblasContractionProblem& prob);

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
        , a_row_stride(1)
        , a_col_stride(ld_a)
        , batch_stride_a(batch_stride_a)
        , B(B)
        , b_row_stride(1)
        , b_col_stride(ld_b)
        , batch_stride_b(batch_stride_b)
        , beta(beta)
        , C(C)
        , c_row_stride(1)
        , c_col_stride(ld_c)
        , batch_stride_c(batch_stride_c)
        , D(C)
        , d_row_stride(1)
        , d_col_stride(ld_c)
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
        , a_row_stride(1)
        , a_col_stride(ld_a)
        , batch_stride_a(batch_stride_a)
        , B(B)
        , b_row_stride(1)
        , b_col_stride(ld_b)
        , batch_stride_b(batch_stride_b)
        , beta(beta)
        , C(C)
        , c_row_stride(1)
        , c_col_stride(ld_c)
        , batch_stride_c(batch_stride_c)
        , D(D)
        , d_row_stride(1)
        , d_col_stride(ld_d)
        , batch_stride_d(batch_stride_d)
        , batch_count(batch_count)
    {
    }

    // gemm_ext2
    // gemm_strided_batched_ext2
    RocblasContractionProblem(rocblas_handle    handle,
                              rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const Tc*         alpha,
                              const Ti*         A,
                              rocblas_int       a_row_stride,
                              rocblas_int       a_col_stride,
                              rocblas_stride    batch_stride_a,
                              const Ti*         B,
                              rocblas_int       b_row_stride,
                              rocblas_int       b_col_stride,
                              rocblas_stride    batch_stride_b,
                              const Tc*         beta,
                              const To*         C,
                              rocblas_int       c_row_stride,
                              rocblas_int       c_col_stride,
                              rocblas_stride    batch_stride_c,
                              To*               D,
                              rocblas_int       d_row_stride,
                              rocblas_int       d_col_stride,
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
        , a_row_stride(a_row_stride)
        , a_col_stride(a_col_stride)
        , batch_stride_a(batch_stride_a)
        , B(B)
        , b_row_stride(b_row_stride)
        , b_col_stride(b_col_stride)
        , batch_stride_b(batch_stride_b)
        , beta(beta)
        , C(C)
        , c_row_stride(c_row_stride)
        , c_col_stride(c_col_stride)
        , batch_stride_c(batch_stride_c)
        , D(D)
        , d_row_stride(d_row_stride)
        , d_col_stride(d_col_stride)
        , batch_stride_d(batch_stride_d)
        , batch_count(batch_count)
    {
    }

    /***************************************************
     * Print a RocblasContractionProblem for debugging *
     ***************************************************/
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const RocblasContractionProblem& prob)
    {
        rocblas_handle    handle;
        rocblas_operation trans_a;
        rocblas_operation trans_b;

        // The size data members should exactly match Tensile's size parameters
        // even if rocBLAS uses smaller or differently-signed types
        size_t m;
        size_t n;
        size_t k;

        const Tc* alpha;

        const Ti* A;
        size_t    a_row_stride;
        size_t    a_col_stride;
        size_t    batch_stride_a;

        const Ti* B;
        size_t    b_row_stride;
        size_t    b_col_stride;
        size_t    batch_stride_b;

        const Tc* beta;

        const To* C;
        size_t    c_row_stride;
        size_t    c_col_stride;
        size_t    batch_stride_c;

        To*    D;
        size_t d_row_stride;
        size_t d_col_stride;
        size_t batch_stride_d;

        size_t batch_count;

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
                            "a_row_stride",
                            prob.a_row_stride,
                            "a_col_stride",
                            prob.a_col_stride,
                            "b_row_stride",
                            prob.b_row_stride,
                            "b_col_stride",
                            prob.b_col_stride,
                            "c_row_stride",
                            prob.c_row_stride,
                            "c_col_stride",
                            prob.c_col_stride,
                            "d_row_stride",
                            prob.d_row_stride,
                            "d_col_stride",
                            prob.d_col_stride,
                            "beta",
                            value_category(prob.beta),
                            "batch_count",
                            prob.batch_count,
                            "stride_a",
                            prob.batch_stride_a,
                            "stride_b",
                            prob.batch_stride_b,
                            "stride_c",
                            prob.batch_stride_c,
                            "stride_d",
                            prob.batch_stride_d));
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
