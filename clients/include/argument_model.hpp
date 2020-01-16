/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#if 0
#ifndef _ARGUMENT_MODEL_HPP_
#define _ARGUMENT_MODEL_HPP_

#include "rocblas_arguments.hpp"

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

#endif
