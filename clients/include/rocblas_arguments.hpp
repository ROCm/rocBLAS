/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_ARGUMENTS_H_
#define ROCBLAS_ARGUMENTS_H_

#include "rocblas.h"
#include "rocblas_datatype2string.hpp"
#include "rocblas_math.hpp"
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

/* ============================================================================================
 */
/*! \brief Class used to parse command arguments in both client & gtest   */
/* WARNING: If this data is changed, then rocblas_common.yaml must also be
 * changed. */

class Arguments
{
public:
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_int KL;
    rocblas_int KU;

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
    double alphai;
    double beta;
    double betai;

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
    rocblas_int stride_x;
    rocblas_int stride_y;

    rocblas_int norm_check;
    rocblas_int unit_check;
    rocblas_int timing;
    rocblas_int iters;

    uint32_t algo;
    int32_t  solution_index;
    uint32_t flags;

    char function[64];
    char name[64];
    char category[64];

    rocblas_initialization initialization;
    char                   known_bug_platforms[64];

    // Validate input format.
    // rocblas_gentest.py is expected to conform to this format.
    // rocblas_gentest.py uses rocblas_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                         "define exactly the same Arguments, that rocblas_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[8]{}, trailer[8]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "rocBLAS"))
            error("header");
        else if(strcmp(trailer, "ROCblas"))
            error("trailer");

        auto check_func = [&, sig = (unsigned char)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(unsigned char i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const unsigned char*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define ROCBLAS_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        ROCBLAS_FORMAT_CHECK(M);
        ROCBLAS_FORMAT_CHECK(N);
        ROCBLAS_FORMAT_CHECK(K);
        ROCBLAS_FORMAT_CHECK(KL);
        ROCBLAS_FORMAT_CHECK(KU);
        ROCBLAS_FORMAT_CHECK(lda);
        ROCBLAS_FORMAT_CHECK(ldb);
        ROCBLAS_FORMAT_CHECK(ldc);
        ROCBLAS_FORMAT_CHECK(ldd);
        ROCBLAS_FORMAT_CHECK(a_type);
        ROCBLAS_FORMAT_CHECK(b_type);
        ROCBLAS_FORMAT_CHECK(c_type);
        ROCBLAS_FORMAT_CHECK(d_type);
        ROCBLAS_FORMAT_CHECK(compute_type);
        ROCBLAS_FORMAT_CHECK(incx);
        ROCBLAS_FORMAT_CHECK(incy);
        ROCBLAS_FORMAT_CHECK(incd);
        ROCBLAS_FORMAT_CHECK(incb);
        ROCBLAS_FORMAT_CHECK(alpha);
        ROCBLAS_FORMAT_CHECK(alphai);
        ROCBLAS_FORMAT_CHECK(beta);
        ROCBLAS_FORMAT_CHECK(betai);
        ROCBLAS_FORMAT_CHECK(transA);
        ROCBLAS_FORMAT_CHECK(transB);
        ROCBLAS_FORMAT_CHECK(side);
        ROCBLAS_FORMAT_CHECK(uplo);
        ROCBLAS_FORMAT_CHECK(diag);
        ROCBLAS_FORMAT_CHECK(batch_count);
        ROCBLAS_FORMAT_CHECK(stride_a);
        ROCBLAS_FORMAT_CHECK(stride_b);
        ROCBLAS_FORMAT_CHECK(stride_c);
        ROCBLAS_FORMAT_CHECK(stride_d);
        ROCBLAS_FORMAT_CHECK(stride_x);
        ROCBLAS_FORMAT_CHECK(stride_y);
        ROCBLAS_FORMAT_CHECK(norm_check);
        ROCBLAS_FORMAT_CHECK(unit_check);
        ROCBLAS_FORMAT_CHECK(timing);
        ROCBLAS_FORMAT_CHECK(iters);
        ROCBLAS_FORMAT_CHECK(algo);
        ROCBLAS_FORMAT_CHECK(solution_index);
        ROCBLAS_FORMAT_CHECK(flags);
        ROCBLAS_FORMAT_CHECK(function);
        ROCBLAS_FORMAT_CHECK(name);
        ROCBLAS_FORMAT_CHECK(category);
        ROCBLAS_FORMAT_CHECK(initialization);
        ROCBLAS_FORMAT_CHECK(known_bug_platforms);
    }

    template <typename T>
    T get_alpha() const
    {
        return rocblas_isnan(alpha) || rocblas_isnan(alphai) ? T(0)
                                                             : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return rocblas_isnan(beta) || rocblas_isnan(betai) ? T(0)
                                                           : convert_alpha_beta<T>(beta, betai);
    }

private:
    template <typename T, typename U, typename std::enable_if<!is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r);
    }

    template <typename T, typename U, typename std::enable_if<+is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r, i);
    }

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

    // Float16 output
    static void print_value(std::ostream& str, rocblas_half x)
    {
        print_value(str, double(x));
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
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

#define PRINT(n) print(#n, arg.n)

        PRINT(function);
        PRINT(a_type);
        PRINT(b_type);
        PRINT(c_type);
        PRINT(d_type);
        PRINT(compute_type);
        PRINT(transA);
        PRINT(transB);
        PRINT(M);
        PRINT(N);
        PRINT(K);
        PRINT(KL);
        PRINT(KU);
        PRINT(lda);
        PRINT(ldb);
        PRINT(ldc);
        PRINT(ldd);
        PRINT(incx);
        PRINT(incy);
        PRINT(incd);
        PRINT(incb);
        PRINT(alpha);
        PRINT(alphai);
        PRINT(beta);
        PRINT(betai);
        PRINT(side);
        PRINT(uplo);
        PRINT(diag);
        PRINT(batch_count);
        PRINT(stride_a);
        PRINT(stride_b);
        PRINT(stride_c);
        PRINT(stride_d);
        PRINT(stride_x);
        PRINT(stride_y);
        PRINT(algo);
        PRINT(solution_index);
        PRINT(flags);
        PRINT(name);
        PRINT(category);
        PRINT(norm_check);
        PRINT(unit_check);
        PRINT(timing);
        PRINT(iters);
        PRINT(initialization);
        PRINT(known_bug_platforms);

#undef PRINT
        return str << " }\n";
    }

    friend class ArgumentModel;
};

static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

enum arg_type
{
    e_M,
    e_N,
    e_K,
    e_lda,
    e_ldb,
    e_ldc,
    e_ldd,
    e_a_type,
    e_b_type,
    e_c_type,
    e_d_type,
    e_compute_type,
    e_incx,
    e_incy,
    e_incb,
    e_incd,
    e_alpha,
    e_beta,
    e_transA,
    e_transB,
    e_side,
    e_uplo,
    e_diag,
    e_batch_count,
    e_stride_a,
    e_stride_b,
    e_stride_c,
    e_stride_d,
    e_stride_x,
    e_stride_y,
    e_algo,
    e_solution_index,

};

#define CTOKEN(n) const char* const c_tok_##n = #n
CTOKEN(M);
CTOKEN(N);
CTOKEN(K);
CTOKEN(lda);
CTOKEN(ldb);
CTOKEN(ldc);
CTOKEN(ldd);
CTOKEN(a_type);
CTOKEN(b_type);
CTOKEN(c_type);
CTOKEN(d_type);
CTOKEN(compute_type);
CTOKEN(incx);
CTOKEN(incy);
CTOKEN(incb);
CTOKEN(incd);
CTOKEN(alpha);
CTOKEN(beta);
CTOKEN(transA);
CTOKEN(transB);
CTOKEN(side);
CTOKEN(uplo);
CTOKEN(diag);
CTOKEN(batch_count);
CTOKEN(stride_a);
CTOKEN(stride_b);
CTOKEN(stride_c);
CTOKEN(stride_d);
CTOKEN(stride_x);
CTOKEN(stride_y);
CTOKEN(algo);
CTOKEN(solution_index);
#undef CTOKEN

class ArgumentModel
{
public:
    ArgumentModel(const std::vector<arg_type>& params);
    virtual ~ArgumentModel() {}

    bool hasParam(arg_type a);

    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        Arguments::print_value(str, x);
    }

    // Float16 output
    static void print_value(std::ostream& str, rocblas_half x)
    {
        Arguments::print_value(str, double(x));
    }

    template <typename T>
    void log_args(std::ostream&    str,
                  const Arguments& args,
                  double           gpu_us,
                  double           gflops,
                  double           gpu_bytes = 0,
                  double           cpu_us    = 0,
                  double           norm1     = 0,
                  double           norm2     = 0);

    virtual void log_perf(std::stringstream& name_str,
                          std::stringstream& val_str,
                          const Arguments&   arg,
                          double             gpu_us,
                          double             gflops,
                          double             gpu_bytes,
                          double             cpu_us,
                          double             norm1,
                          double             norm2);

protected:
    std::vector<arg_type> m_args;
};

template <typename T>
void ArgumentModel::log_args(std::ostream&    str,
                             const Arguments& arg,
                             double           gpu_us,
                             double           gflops,
                             double           gpu_bytes,
                             double           cpu_us,
                             double           norm1,
                             double           norm2)
{
    std::stringstream name_list;
    std::stringstream value_list;
    const char        delim = ',';

    auto print = [&](const char* const name, auto x) mutable {
        name_list << name << delim;
        print_value(value_list, x);
        value_list << delim;
    };

#define CASE(x)                  \
    case(e_##x):                 \
    {                            \
        print(c_tok_##x, arg.x); \
    }                            \
    break

    for(auto&& i : m_args)
    {
        switch(i)
        {
        case e_alpha:
        {
            T a = arg.get_alpha<T>();
            print(c_tok_alpha, a);
        }
        break;
        case e_beta:
        {
            T b = arg.get_beta<T>();
            print(c_tok_beta, b);
        }
        break;

            CASE(M);
            CASE(N);
            CASE(K);
            CASE(lda);
            CASE(ldb);
            CASE(ldc);
            CASE(ldd);
            CASE(a_type);
            CASE(b_type);
            CASE(c_type);
            CASE(d_type);
            CASE(compute_type);
            CASE(incx);
            CASE(incy);
            CASE(incd);
            CASE(incb);
            // alpha beta special cased above
            CASE(transA);
            CASE(transB);
            CASE(side);
            CASE(uplo);
            CASE(diag);
            CASE(batch_count);
            CASE(stride_a);
            CASE(stride_b);
            CASE(stride_c);
            CASE(stride_d);
            CASE(stride_x);
            CASE(stride_y);
            CASE(algo);
            CASE(solution_index);

            // default:
            // {
            //     name_list << "unknown,";
            //     value_list << "unknown,";
            // }
            // break;
        }
    }

#undef CASE

    if(arg.timing)
    {
        log_perf(name_list, value_list, arg, gpu_us, gflops, gpu_bytes, cpu_us, norm1, norm2);
    }

    str << name_list.str() << std::endl;
    str << value_list.str() << std::endl;
}

#endif
