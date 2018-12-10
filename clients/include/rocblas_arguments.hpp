/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_ARGUMENTS_H_
#define ROCBLAS_ARGUMENTS_H_

#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <type_traits>
#include "rocblas.h"
#include "rocblas_datatype2char.hpp"

/* ============================================================================================ */
/*! \brief Class used to parse command arguments in both client & gtest   */
struct Arguments
{
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_int lda;
    rocblas_int ldb;
    rocblas_int ldc;
    rocblas_int ldd;

    rocblas_datatype a_type;
    rocblas_datatype b_type;
    rocblas_datatype c_type;
    rocblas_datatype d_type;
    rocblas_datatype compute_type;

    rocblas_int incx;
    rocblas_int incy;
    rocblas_int incd;
    rocblas_int incb;

    double alpha;
    double beta;

    char transA;
    char transB;
    char side;
    char uplo;
    char diag;

    rocblas_int batch_count;

    rocblas_int stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    rocblas_int stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    rocblas_int stride_c; //  stride_c > ldc * N
    rocblas_int stride_d; //  stride_d > ldd * N

    rocblas_int norm_check;
    rocblas_int unit_check;
    rocblas_int timing;
    rocblas_int iters;

    uint32_t algo;
    int32_t solution_index;
    uint32_t flags;
    size_t workspace_size;

    char function[64];
    char name[64];
    char category[32];

    private:
    /* =============================================================================================
     */
    /* All Arguments data members are above. Below are support functions for reading and formatting
     */
    /* =============================================================================================
     */

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
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
        char s[32];
        snprintf(s, sizeof(s) - 2, "%.17g", x);
        if(!strpbrk(s, ".eE"))
            strcat(s, ".0"); // If no decimal point or exponent, append one
        str << s;
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b) { str << (b ? "true" : "false"); }

    // string output
    static void print_value(std::ostream& str, const char* s) { str << std::quoted(s); }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{' ](const char* name, auto x) mutable
        {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("lda", arg.lda);
        print("ldb", arg.ldb);
        print("ldc", arg.ldc);
        print("ldd", arg.ldd);
        print("a_type", rocblas_datatype2char(arg.a_type));
        print("b_type", rocblas_datatype2char(arg.b_type));
        print("c_type", rocblas_datatype2char(arg.c_type));
        print("d_type", rocblas_datatype2char(arg.d_type));
        print("compute_type", rocblas_datatype2char(arg.compute_type));
        print("incx", arg.incx);
        print("incy", arg.incy);
        print("incd", arg.incd);
        print("incb", arg.incb);
        print("alpha", arg.alpha);
        print("beta", arg.beta);
        print("transA", arg.transA);
        print("transB", arg.transB);
        print("side", arg.side);
        print("uplo", arg.uplo);
        print("diag", arg.diag);
        print("batch_count", arg.batch_count);
        print("stride_a", arg.stride_a);
        print("stride_b", arg.stride_b);
        print("stride_c", arg.stride_c);
        print("stride_d", arg.stride_d);
        print("norm_check", arg.norm_check);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        print("algo", arg.algo);
        print("solution_index", arg.solution_index);
        print("flags", arg.flags);
        print("function", arg.function);
        print("name", arg.name);
        print("category", arg.category);

        return str << " }\n";
    }
};

static_assert(std::is_pod<Arguments>(),
              "Arguments is not a POD type, and thus is incompatible with C.");

#endif
