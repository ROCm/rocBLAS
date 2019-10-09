/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas.h"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_parse_data.hpp"
// blas1
#include "testing_asum.hpp"
#include "testing_asum_batched.hpp"
#include "testing_asum_strided_batched.hpp"
#include "testing_axpy.hpp"
#include "testing_copy.hpp"
#include "testing_copy_batched.hpp"
#include "testing_copy_strided_batched.hpp"
#include "testing_dot.hpp"
#include "testing_iamax_iamin.hpp"
#include "testing_nrm2.hpp"
#include "testing_nrm2_batched.hpp"
#include "testing_nrm2_strided_batched.hpp"
#include "testing_rot.hpp"
#include "testing_rot_batched.hpp"
#include "testing_rot_strided_batched.hpp"
#include "testing_rotg.hpp"
#include "testing_rotg_batched.hpp"
#include "testing_rotg_strided_batched.hpp"
#include "testing_rotm.hpp"
#include "testing_rotm_batched.hpp"
#include "testing_rotm_strided_batched.hpp"
#include "testing_rotmg.hpp"
#include "testing_rotmg_batched.hpp"
#include "testing_rotmg_strided_batched.hpp"
#include "testing_scal.hpp"
#include "testing_scal_batched.hpp"
#include "testing_scal_strided_batched.hpp"
#include "testing_set_get_matrix.hpp"
#include "testing_set_get_vector.hpp"
#include "testing_swap.hpp"
#include "testing_swap_batched.hpp"
#include "testing_swap_strided_batched.hpp"
// blas2
#include "testing_gemv.hpp"
#include "testing_gemv_batched.hpp"
#include "testing_gemv_strided_batched.hpp"
#include "testing_ger.hpp"
#include "testing_syr.hpp"
#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#undef I
#include <boost/program_options.hpp>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace std::literals;

#if BUILD_WITH_TENSILE

#include "testing_geam.hpp"
#include "testing_gemm.hpp"
#include "testing_gemm_batched.hpp"
#include "testing_gemm_batched_ex.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched.hpp"
#include "testing_gemm_strided_batched_ex.hpp"
#include "testing_trsm.hpp"
#include "testing_trsm_ex.hpp"
#include "testing_trsv.hpp"
#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#include "testing_trtri_strided_batched.hpp"

// Template to dispatch testing_gemm_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_ex : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_ex<Ti,
                    To,
                    Tc,
                    typename std::enable_if<!std::is_same<Ti, void>{}
                                            && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                                                 && std::is_same<Ti, rocblas_bfloat16>{})>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "gemm_ex"))
            testing_gemm_ex<Ti, To, Tc>(arg);
        else if(!strcmp(arg.function, "gemm_batched_ex"))
            testing_gemm_batched_ex<Ti, To, Tc>(arg);
    }
};

// Template to dispatch testing_gemm_strided_batched_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_strided_batched_ex : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_strided_batched_ex<
    Ti,
    To,
    Tc,
    typename std::enable_if<!std::is_same<Ti, void>{}
                            && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                                 && std::is_same<Ti, rocblas_bfloat16>{})>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        testing_gemm_strided_batched_ex<Ti, To, Tc>(arg);
    }
};

#endif

template <typename T, typename U = T, typename = void>
struct perf_blas : rocblas_test_invalid
{
};

template <typename T, typename U>
struct perf_blas<
    T,
    U,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "asum"))
            testing_asum<T>(arg);
        else if(!strcmp(arg.function, "axpy"))
            testing_axpy<T>(arg);
        else if(!strcmp(arg.function, "copy"))
            testing_copy<T>(arg);
        else if(!strcmp(arg.function, "copy_batched"))
            testing_copy_batched<T>(arg);
        else if(!strcmp(arg.function, "copy_strided_batched"))
            testing_copy_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "dot"))
            testing_dot<T>(arg);
        else if(!strcmp(arg.function, "swap"))
            testing_swap<T>(arg);
        else if(!strcmp(arg.function, "swap_batched"))
            testing_swap_batched<T>(arg);
        else if(!strcmp(arg.function, "swap_strided_batched"))
            testing_swap_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "iamax"))
            testing_iamax<T>(arg);
        else if(!strcmp(arg.function, "iamin"))
            testing_iamin<T>(arg);
        else if(!strcmp(arg.function, "nrm2"))
            testing_nrm2<T>(arg);
        else if(!strcmp(arg.function, "nrm2_batched"))
            testing_nrm2_batched<T>(arg);
        else if(!strcmp(arg.function, "nrm2_strided_batched"))
            testing_nrm2_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "set_get_vector"))
            testing_set_get_vector<T>(arg);
        else if(!strcmp(arg.function, "set_get_matrix"))
            testing_set_get_matrix<T>(arg);
        else if(!strcmp(arg.function, "rotm"))
            testing_rotm<T>(arg);
        else if(!strcmp(arg.function, "rotm_batched"))
            testing_rotm_batched<T>(arg);
        else if(!strcmp(arg.function, "rotm_strided_batched"))
            testing_rotm_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "rotmg"))
            testing_rotmg<T>(arg);
        else if(!strcmp(arg.function, "rotmg_batched"))
            testing_rotmg_batched<T>(arg);
        else if(!strcmp(arg.function, "rotmg_strided_batched"))
            testing_rotmg_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "gemv"))
            testing_gemv<T>(arg);
        else if(!strcmp(arg.function, "gemv_batched"))
            testing_gemv_batched<T>(arg);
        else if(!strcmp(arg.function, "gemv_strided_batched"))
            testing_gemv_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "ger"))
            testing_ger<T>(arg);
        else if(!strcmp(arg.function, "syr"))
            testing_syr<T>(arg);
#if BUILD_WITH_TENSILE
        else if(!strcmp(arg.function, "geam"))
            testing_geam<T>(arg);
        else if(!strcmp(arg.function, "trtri"))
            testing_trtri<T>(arg);
        else if(!strcmp(arg.function, "trtri_batched"))
            testing_trtri_batched<T>(arg);
        else if(!strcmp(arg.function, "trtri_strided_batched"))
            testing_trtri_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "gemm"))
            testing_gemm<T>(arg);
        else if(!strcmp(arg.function, "gemm_batched"))
            testing_gemm_batched<T>(arg);
        else if(!strcmp(arg.function, "gemm_strided_batched"))
            testing_gemm_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "trsm"))
            testing_trsm<T>(arg);
        else if(!strcmp(arg.function, "trsm_ex"))
            testing_trsm_ex<T>(arg);
        else if(!strcmp(arg.function, "trsv"))
            testing_trsv<T>(arg);
#endif
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + rocblas_datatype2string(arg.a_type));
    }
};

template <typename T, typename U>
struct perf_blas<T, U, typename std::enable_if<std::is_same<T, rocblas_bfloat16>{}>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "dot"))
            testing_dot<T>(arg);
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + rocblas_datatype2string(arg.a_type));
    }
};

template <typename T, typename U>
struct perf_blas<T, U, typename std::enable_if<std::is_same<T, rocblas_half>{}>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "axpy"))
            testing_axpy<T>(arg);
        else if(!strcmp(arg.function, "dot"))
            testing_dot<T>(arg);
#if BUILD_WITH_TENSILE
        else if(!strcmp(arg.function, "gemm"))
            testing_gemm<T>(arg);
        else if(!strcmp(arg.function, "gemm_batched"))
            testing_gemm_batched<T>(arg);
        else if(!strcmp(arg.function, "gemm_strided_batched"))
            testing_gemm_strided_batched<T>(arg);
#endif
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + rocblas_datatype2string(arg.a_type));
    }
};

template <typename T, typename U>
struct perf_blas<T,
                 U,
                 typename std::enable_if<std::is_same<T, rocblas_double_complex>{}
                                         || std::is_same<T, rocblas_float_complex>{}>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "asum"))
            testing_asum<T>(arg);
        else if(!strcmp(arg.function, "asum_batched"))
            testing_asum_batched<T>(arg);
        else if(!strcmp(arg.function, "asum_strided_batched"))
            testing_asum_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "axpy"))
            testing_axpy<T>(arg);
        else if(!strcmp(arg.function, "copy"))
            testing_copy<T>(arg);
        else if(!strcmp(arg.function, "copy_batched"))
            testing_copy_batched<T>(arg);
        else if(!strcmp(arg.function, "copy_strided_batched"))
            testing_copy_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "dot"))
            testing_dot<T>(arg);
        else if(!strcmp(arg.function, "dotc"))
            testing_dotc<T>(arg);
        else if(!strcmp(arg.function, "nrm2"))
            testing_nrm2<T>(arg);
        else if(!strcmp(arg.function, "nrm2_batched"))
            testing_nrm2_batched<T>(arg);
        else if(!strcmp(arg.function, "nrm2_strided_batched"))
            testing_nrm2_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "swap"))
            testing_swap<T>(arg);
        else if(!strcmp(arg.function, "swap_batched"))
            testing_swap_batched<T>(arg);
        else if(!strcmp(arg.function, "swap_strided_batched"))
            testing_swap_strided_batched<T>(arg);
        else if(!strcmp(arg.function, "iamax"))
            testing_iamax<T>(arg);
        else if(!strcmp(arg.function, "iamin"))
            testing_iamin<T>(arg);
        else if(!strcmp(arg.function, "gemv"))
            testing_gemv<T>(arg);
#if BUILD_WITH_TENSILE
        else if(!strcmp(arg.function, "gemm"))
            testing_gemm<T>(arg);
        else if(!strcmp(arg.function, "gemm_batched"))
            testing_gemm_batched<T>(arg);
        else if(!strcmp(arg.function, "gemm_strided_batched"))
            testing_gemm_strided_batched<T>(arg);
#endif
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + rocblas_datatype2string(arg.a_type));
    }
};

template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_blas_rot : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_blas_rot<
    Ti,
    To,
    Tc,
    typename std::enable_if<(
        (std::is_same<Ti, float>{} && std::is_same<Ti, To>{} && std::is_same<To, Tc>{})
        || (std::is_same<Ti, double>{} && std::is_same<Ti, To>{} && std::is_same<To, Tc>{})
        || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{}
            && std::is_same<Tc, rocblas_float_complex>{})
        || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{}
            && std::is_same<Tc, float>{})
        || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{}
            && std::is_same<Tc, rocblas_double_complex>{})
        || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{}
            && std::is_same<Tc, double>{}))>::type>
{
    explicit operator bool()
    {
        return true;
    }

    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "rot"))
            testing_rot<Ti, To, Tc>(arg);
        else if(!strcmp(arg.function, "rot_batched"))
            testing_rot_batched<Ti, To, Tc>(arg);
        else if(!strcmp(arg.function, "rot_strided_batched"))
            testing_rot_strided_batched<Ti, To, Tc>(arg);
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + rocblas_datatype2string(arg.a_type));
    }
};

template <typename Ta, typename Tb = Ta, typename = void>
struct perf_blas_scal : rocblas_test_invalid
{
};

template <typename Ta, typename Tb>
struct perf_blas_scal<
    Ta,
    Tb,
    typename std::enable_if<
        (std::is_same<Ta, double>{} && std::is_same<Tb, rocblas_double_complex>{})
        || (std::is_same<Ta, float>{} && std::is_same<Tb, rocblas_float_complex>{})
        || (std::is_same<Ta, Tb>{} && std::is_same<Ta, float>{})
        || (std::is_same<Ta, Tb>{} && std::is_same<Ta, double>{})
        || (std::is_same<Ta, Tb>{} && std::is_same<Ta, rocblas_float_complex>{})
        || (std::is_same<Ta, Tb>{} && std::is_same<Ta, rocblas_double_complex>{})>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "scal"))
            testing_scal<Ta, Tb>(arg);
        else if(!strcmp(arg.function, "scal_batched"))
            testing_scal_batched<Ta, Tb>(arg);
        else if(!strcmp(arg.function, "scal_strided_batched"))
            testing_scal_strided_batched<Ta, Tb>(arg);
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + rocblas_datatype2string(arg.a_type));
    }
};

template <typename Ta, typename Tb = Ta, typename = void>
struct perf_blas_rotg : rocblas_test_invalid
{
};

template <typename Ta, typename Tb>
struct perf_blas_rotg<
    Ta,
    Tb,
    typename std::enable_if<
        (std::is_same<Ta, rocblas_double_complex>{} && std::is_same<Tb, double>{})
        || (std::is_same<Ta, rocblas_float_complex>{} && std::is_same<Tb, float>{})
        || (std::is_same<Ta, Tb>{} && std::is_same<Ta, float>{})
        || (std::is_same<Ta, Tb>{} && std::is_same<Ta, double>{})>::type>
{
    explicit operator bool()
    {
        return true;
    }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "rotg"))
            testing_rotg<Ta, Tb>(arg);
        else if(!strcmp(arg.function, "rotg_batched"))
            testing_rotg_batched<Ta, Tb>(arg);
        else if(!strcmp(arg.function, "rotg_strided_batched"))
            testing_rotg_strided_batched<Ta, Tb>(arg);
        else
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type " + rocblas_datatype2string(arg.a_type)
                                        + " --b_type " + rocblas_datatype2string(arg.b_type));
    }
};

int run_bench_test(Arguments& arg)
{
    // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    arg.timing = 1;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

#if BUILD_WITH_TENSILE
    if(!strcmp(function, "gemm") || !strcmp(function, "gemm_batched"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;

        if(arg.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
    }
    else if(!strcmp(function, "gemm_strided_batched"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;
        if(arg.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }

        //      rocblas_int min_stride_a =
        //          arg.transA == 'N' ? arg.K * arg.lda : arg.M * arg.lda;
        //      rocblas_int min_stride_b =
        //          arg.transB == 'N' ? arg.N * arg.ldb : arg.K * arg.ldb;
        //      rocblas_int min_stride_a =
        //          arg.transA == 'N' ? arg.K * arg.lda : arg.M * arg.lda;
        //      rocblas_int min_stride_b =
        //          arg.transB == 'N' ? arg.N * arg.ldb : arg.K * arg.ldb;
        rocblas_int min_stride_c = arg.ldc * arg.N;
        //      if (arg.stride_a < min_stride_a)
        //      {
        //          std::cout << "rocblas-bench INFO: stride_a < min_stride_a, set stride_a = " <<
        //          min_stride_a << std::endl;
        //          arg.stride_a = min_stride_a;
        //      }
        //      if (arg.stride_b < min_stride_b)
        //      {
        //          std::cout << "rocblas-bench INFO: stride_b < min_stride_b, set stride_b = " <<
        //          min_stride_b << std::endl;
        //          arg.stride_b = min_stride_b;
        //      }
        if(arg.stride_c < min_stride_c)
        {
            std::cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }
    }

    if(!strcmp(function, "gemm_ex") || !strcmp(function, "gemm_batched_ex"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;
        rocblas_int min_ldd = arg.M;

        if(arg.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
        if(arg.ldd < min_ldd)
        {
            std::cout << "rocblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd = min_ldd;
        }
        rocblas_gemm_dispatch<perf_gemm_ex>(arg);
    }
    else if(!strcmp(function, "gemm_strided_batched_ex"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;
        rocblas_int min_ldd = arg.M;
        if(arg.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
        if(arg.ldd < min_ldd)
        {
            std::cout << "rocblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd = min_ldd;
        }
        rocblas_int min_stride_c = arg.ldc * arg.N;
        if(arg.stride_c < min_stride_c)
        {
            std::cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }

        rocblas_gemm_dispatch<perf_gemm_strided_batched_ex>(arg);
    }
    else
#endif
    {
        if(!strcmp(function, "scal") || !strcmp(function, "scal_batched")
           || !strcmp(function, "scal_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_scal>(arg);
        else if(!strcmp(function, "rotg") || !strcmp(function, "rotg_batched")
                || !strcmp(function, "rotg_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_rotg>(arg);
        else if(!strcmp(function, "rot") || !strcmp(function, "rot_batched")
                || !strcmp(function, "rot_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_rot>(arg);
        else
            rocblas_simple_dispatch<perf_blas>(arg);
    }
    return 0;
}

int rocblas_bench_datafile()
{
    int ret = 0;
    for(Arguments arg : RocBLAS_TestData())
        ret |= run_bench_test(arg);
    test_cleanup::cleanup();
    return ret;
}

using namespace boost::program_options;

int main(int argc, char* argv[])
try
{
    Arguments arg;

    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string initialization;

    rocblas_int device_id;
    bool        datafile = rocblas_parse_data(argc, argv);

    options_description desc("rocblas-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<rocblas_int>(&arg.M)->default_value(128),
         "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows or columns in matrix.")

        ("sizen,n",
         value<rocblas_int>(&arg.N)->default_value(128),
         "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of rows or columns in matrix")

        ("sizek,k",
         value<rocblas_int>(&arg.K)->default_value(128),
         "Specific matrix size:sizek is only applicable to BLAS-3: the number of columns in "
         "A and rows in B.")

        ("lda",
         value<rocblas_int>(&arg.lda)->default_value(128),
         "Leading dimension of matrix A, is only applicable to BLAS-2 & BLAS-3.")

        ("ldb",
         value<rocblas_int>(&arg.ldb)->default_value(128),
         "Leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3.")

        ("ldc",
         value<rocblas_int>(&arg.ldc)->default_value(128),
         "Leading dimension of matrix C, is only applicable to BLAS-2 & BLAS-3.")

        ("ldd",
         value<rocblas_int>(&arg.ldd)->default_value(128),
         "Leading dimension of matrix D, is only applicable to BLAS-EX ")

        ("stride_a",
         value<rocblas_int>(&arg.stride_a)->default_value(128*128),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_b",
         value<rocblas_int>(&arg.stride_b)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_c",
         value<rocblas_int>(&arg.stride_c)->default_value(128*128),
         "Specific stride of strided_batched matrix C, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_d",
         value<rocblas_int>(&arg.stride_d)->default_value(128*128),
         "Specific stride of strided_batched matrix D, is only applicable to strided batched"
         "BLAS_EX: second dimension * leading dimension.")

        ("stride_x",
         value<rocblas_int>(&arg.stride_x)->default_value(128*128),
         "Specific stride of strided_batched vector x, is only applicable to strided batched"
         "BLAS_2: second dimension.")

        ("stride_y",
         value<rocblas_int>(&arg.stride_y)->default_value(128*128),
         "Specific stride of strided_batched vector y, is only applicable to strided batched"
         "BLAS_2: leading dimension.")

        ("incx",
         value<rocblas_int>(&arg.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<rocblas_int>(&arg.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("alphai",
         value<double>(&arg.alphai)->default_value(0.0), "specifies the imaginary part of the scalar alpha")

        ("beta",
         value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("betai",
         value<double>(&arg.betai)->default_value(0.0), "specifies the imaginary part of the scalar beta")

        ("function,f",
         value<std::string>(&function),
         "BLAS function to test.")

        ("precision,r",
         value<std::string>(&precision)->default_value("f32_r"), "Precision. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("compute_type",
         value<std::string>(&compute_type), "Precision of computation. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("initialization",
         value<std::string>(&initialization)->default_value("rand_int"),
         "Intialize with random integers, trig functions sin and cos, or hpl-like input. "
         "Options: rand_int, trig_float, hpl")

        ("transposeA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         value<char>(&arg.side)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         value<char>(&arg.uplo)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm xtrsm_ex
                                                                     // xtrmm xtrsv
        ("diag",
         value<char>(&arg.diag)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm xtrsm_ex xtrsv

        ("batch",
         value<rocblas_int>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines") // xtrsm xtrsm_ex xtrmm xgemm

        ("verify,v",
         value<rocblas_int>(&arg.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<rocblas_int>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("algo",
         value<uint32_t>(&arg.algo)->default_value(0),
         "extended precision gemm algorithm")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(0),
         "extended precision gemm solution index")

        ("flags",
         value<uint32_t>(&arg.flags)->default_value(10),
         "extended precision gemm flags")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("help,h", "produces this help message")

        ("version", "Prints the version number");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(vm.find("version") != vm.end())
    {
        char blas_version[100];
        rocblas_get_version_string(blas_version, sizeof(blas_version));
        std::cout << "rocBLAS version: " << blas_version << std::endl;
        return 0;
    }
    // Device Query
    rocblas_int device_count = query_device_property();

    std::cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);

    if(datafile)
        return rocblas_bench_datafile();

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2rocblas_datatype(precision);
    if(prec == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string2rocblas_datatype(a_type);
    if(arg.a_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string2rocblas_datatype(b_type);
    if(arg.b_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string2rocblas_datatype(c_type);
    if(arg.c_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string2rocblas_datatype(d_type);
    if(arg.d_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    arg.compute_type = compute_type == "" ? prec : string2rocblas_datatype(compute_type);
    if(arg.compute_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    arg.initialization = string2rocblas_initialization(initialization);
    if(arg.initialization == static_cast<rocblas_initialization>(-1))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    return run_bench_test(arg);
}
catch(const std::invalid_argument& exp)
{
    std::cerr << exp.what() << std::endl;
    return -1;
}
