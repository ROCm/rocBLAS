/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <ostream>

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
    RocblasContractionProblem(rocblas_handle    handle,
                              rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              const Tc*         alpha,
                              const Ti*         A,
                              rocblas_int       row_stride_a,
                              rocblas_int       col_stride_a,
                              rocblas_stride    batch_stride_a,
                              const Ti*         B,
                              rocblas_int       row_stride_b,
                              rocblas_int       col_stride_b,
                              rocblas_stride    batch_stride_b,
                              const Tc*         beta,
                              const To*         C,
                              rocblas_int       row_stride_c,
                              rocblas_int       col_stride_c,
                              rocblas_stride    batch_stride_c,
                              To*               D,
                              rocblas_int       row_stride_d,
                              rocblas_int       col_stride_d,
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

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    friend std::ostream& operator<<(std::ostream& str, const RocblasContractionProblem& prob)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

#define PRINT(name, value) print(#name, value)

        PRINT(a_type, rocblas_precision_string<Ti>);
        PRINT(b_type, rocblas_precision_string<Ti>);
        PRINT(c_type, rocblas_precision_string<To>);
        PRINT(d_type, rocblas_precision_string<To>);
        PRINT(compute_type, rocblas_precision_string<Tc>);
        PRINT(transA, rocblas_transpose_letter(prob.trans_a));
        PRINT(transB, rocblas_transpose_letter(prob.trans_b));
        PRINT(M, prob.m);
        PRINT(N, prob.n);
        PRINT(K, prob.k);
        PRINT(row_stride_a, prob.col_stride_a);
        PRINT(col_stride_a, prob.col_stride_a);
        PRINT(row_stride_b, prob.row_stride_b);
        PRINT(col_stride_b, prob.col_stride_b);
        PRINT(row_stride_c, prob.row_stride_c);
        PRINT(col_stride_c, prob.col_stride_c);
        PRINT(row_stride_d, prob.row_stride_d);
        PRINT(col_stride_d, prob.col_stride_d);
        PRINT(beta, value_category(prob.beta));
        PRINT(batch_count, prob.batch_count);
        PRINT(batch_stride_a, prob.batch_stride_a);
        PRINT(batch_stride_b, prob.batch_stride_b);
        PRINT(batch_stride_c, prob.batch_stride_c);
        PRINT(batch_stride_d, prob.batch_stride_d);

#undef PRINT
        str << " }\n";
        return str;
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
