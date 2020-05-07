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
                            "lda",
                            prob.ld_a,
                            "ldb",
                            prob.ld_b,
                            "ldc",
                            prob.ld_c,
                            "ldd",
                            prob.ld_d,
                            "beta",
                            value_category(prob.beta),
                            "batch_count",
                            prob.batch_count,
                            "stride_a",
                            prob.stride_a,
                            "stride_b",
                            prob.stride_b,
                            "stride_c",
                            prob.stride_c,
                            "stride_d",
                            prob.stride_d));
    }
};

/*******************************************************************************
 * runContractionProblem() solves a RocblasContractionProblem                  *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status runContractionProblem(RocblasContractionProblem<Ti, To, Tc> const& problem);

#endif // __TENSILE_HOST_HPP__
