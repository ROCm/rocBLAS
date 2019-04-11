/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_ARGUMENTS_H_
#define ROCBLAS_ARGUMENTS_H_

#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <type_traits>
#include "rocblas.h"
#include "rocblas_datatype2string.hpp"

/* ============================================================================================ */
/*! \brief Class used to parse command arguments in both client & gtest   */
/* WARNING: If this data is changed, then rocblas_common.yaml must also be changed. */

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

    rocblas_initialization initialization;

    // Validate input format.
    // rocblas_gentest.py is expected to conform to this format.
    // rocblas_gentest.py uses rocblas_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [] {
            std::cerr << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                         "define exactly the same Arguments, that rocblas_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            exit(EXIT_FAILURE);
        };

        char header[8]{}, trailer[8]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "rocBLAS") || strcmp(trailer, "ROCblas"))
            error();

        auto check = [&, sig = (unsigned char)0 ](const auto& elem) mutable
        {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(unsigned char i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const unsigned char*>(&elem)[i] ^ sig ^ i)
                    error();
            sig += 89;
        };

        // Order is important
        check(arg.M);
        check(arg.N);
        check(arg.K);
        check(arg.lda);
        check(arg.ldb);
        check(arg.ldc);
        check(arg.ldd);
        check(arg.a_type);
        check(arg.b_type);
        check(arg.c_type);
        check(arg.d_type);
        check(arg.compute_type);
        check(arg.incx);
        check(arg.incy);
        check(arg.incd);
        check(arg.incb);
        check(arg.alpha);
        check(arg.beta);
        check(arg.transA);
        check(arg.transB);
        check(arg.side);
        check(arg.uplo);
        check(arg.diag);
        check(arg.batch_count);
        check(arg.stride_a);
        check(arg.stride_b);
        check(arg.stride_c);
        check(arg.stride_d);
        check(arg.norm_check);
        check(arg.unit_check);
        check(arg.timing);
        check(arg.iters);
        check(arg.algo);
        check(arg.solution_index);
        check(arg.flags);
        check(arg.workspace_size);
        check(arg.function);
        check(arg.name);
        check(arg.category);
        check(arg.initialization);
    }

    private:
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

        print("function", arg.function);
        print("a_type", rocblas_datatype2string(arg.a_type));
        print("b_type", rocblas_datatype2string(arg.b_type));
        print("c_type", rocblas_datatype2string(arg.c_type));
        print("d_type", rocblas_datatype2string(arg.d_type));
        print("compute_type", rocblas_datatype2string(arg.compute_type));
        print("transA", arg.transA);
        print("transB", arg.transB);
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("lda", arg.lda);
        print("ldb", arg.ldb);
        print("ldc", arg.ldc);
        print("ldd", arg.ldd);
        print("incx", arg.incx);
        print("incy", arg.incy);
        print("incd", arg.incd);
        print("incb", arg.incb);
        print("alpha", arg.alpha);
        print("beta", arg.beta);
        print("side", arg.side);
        print("uplo", arg.uplo);
        print("diag", arg.diag);
        print("batch_count", arg.batch_count);
        print("stride_a", arg.stride_a);
        print("stride_b", arg.stride_b);
        print("stride_c", arg.stride_c);
        print("stride_d", arg.stride_d);
        print("algo", arg.algo);
        print("solution_index", arg.solution_index);
        print("flags", arg.flags);
        print("name", arg.name);
        print("category", arg.category);
        print("norm_check", arg.norm_check);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        print("initialization", arg.initialization);

        return str << " }\n";
    }
};

static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is incompatible with C.");

static_assert(std::is_trivially_copyable<Arguments>{},
              "Arguments is not a trivially copyable type, and thus is incompatible with C.");

#endif
