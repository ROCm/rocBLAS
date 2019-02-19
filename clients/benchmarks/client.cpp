/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <boost/program_options.hpp>

#include "rocblas.h"
#include "utility.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_iamax_iamin.hpp"
#include "testing_asum.hpp"
#include "testing_axpy.hpp"
#include "testing_copy.hpp"
#include "testing_dot.hpp"
#include "testing_swap.hpp"
#include "testing_gemv.hpp"
#include "testing_ger.hpp"
#include "testing_syr.hpp"
#include "testing_nrm2.hpp"
#include "testing_scal.hpp"
#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#include "testing_geam.hpp"
#include "testing_set_get_vector.hpp"
#include "testing_set_get_matrix.hpp"
#include "type_dispatch.hpp"
#include "rocblas_parse_data.hpp"

#if BUILD_WITH_TENSILE
#include "testing_gemm.hpp"
#include "testing_gemm_strided_batched.hpp"
#include "testing_trsm.hpp"
#include "testing_trsv.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched_ex.hpp"

// Template to dispatch testing_gemm_ex for performance tests
// When Ti == void or complex, the test is marked invalid
template <typename Ti,
          typename To = Ti,
          typename Tc = To,
          typename    = typename std::conditional<!std::is_same<Ti, void>{} && !is_complex<Ti>,
                                               std::true_type,
                                               std::false_type>::type>
struct perf_gemm_ex
{
    explicit operator bool() const { return true; }
    void operator()(const Arguments& arg) { testing_gemm_ex<Ti, To, Tc>(arg); }
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_ex<Ti, To, Tc, std::false_type> : rocblas_test_invalid
{
};

// Template to dispatch testing_gemm_strided_batched_ex for performance tests
// When Ti == void or complex, the test is marked invalid
template <typename Ti,
          typename To = Ti,
          typename Tc = To,
          typename    = typename std::conditional<!std::is_same<Ti, void>{} && !is_complex<Ti>,
                                               std::true_type,
                                               std::false_type>::type>
struct perf_gemm_strided_batched_ex
{
    explicit operator bool() const { return true; }
    void operator()(const Arguments& arg) { testing_gemm_strided_batched_ex<Ti, To, Tc>(arg); }
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_strided_batched_ex<Ti, To, Tc, std::false_type> : rocblas_test_invalid
{
};

#endif

int run_bench_test(const char* function, char precision, Arguments arg)
{
    static constexpr char prefix[] = "testing_";
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
    {
        function += sizeof(prefix) - 1;
    }

    if(!strcmp(function, "asum"))
    {
        if(precision == 's')
            testing_asum<float, float>(arg);
        else if(precision == 'd')
            testing_asum<double, double>(arg);
    }
    else if(!strcmp(function, "axpy"))
    {
        if(precision == 'h')
            testing_axpy<rocblas_half>(arg);
        else if(precision == 's')
            testing_axpy<float>(arg);
        else if(precision == 'd')
            testing_axpy<double>(arg);
    }
    else if(!strcmp(function, "copy"))
    {
        if(precision == 's')
            testing_copy<float>(arg);
        else if(precision == 'd')
            testing_copy<double>(arg);
    }
    else if(!strcmp(function, "dot"))
    {
        if(precision == 's')
            testing_dot<float>(arg);
        else if(precision == 'd')
            testing_dot<double>(arg);
    }
    else if(!strcmp(function, "swap"))
    {
        if(precision == 's')
            testing_swap<float>(arg);
        else if(precision == 'd')
            testing_swap<double>(arg);
    }
    else if(!strcmp(function, "iamax"))
    {
        if(precision == 's')
            testing_iamax<float>(arg);
        else if(precision == 'd')
            testing_iamax<double>(arg);
    }
    else if(!strcmp(function, "iamin"))
    {
        if(precision == 's')
            testing_iamin<float>(arg);
        else if(precision == 'd')
            testing_iamin<double>(arg);
    }
    else if(!strcmp(function, "nrm2"))
    {
        if(precision == 's')
            testing_nrm2<float, float>(arg);
        else if(precision == 'd')
            testing_nrm2<double, double>(arg);
    }
    else if(!strcmp(function, "scal"))
    {
        if(precision == 's')
            testing_scal<float>(arg);
        else if(precision == 'd')
            testing_scal<double>(arg);
    }
    else if(!strcmp(function, "gemv"))
    {
        if(precision == 's')
            testing_gemv<float>(arg);
        else if(precision == 'd')
            testing_gemv<double>(arg);
    }
    else if(!strcmp(function, "ger"))
    {
        if(precision == 's')
            testing_ger<float>(arg);
        else if(precision == 'd')
            testing_ger<double>(arg);
    }
    else if(!strcmp(function, "syr"))
    {
        if(precision == 's')
            testing_syr<float>(arg);
        else if(precision == 'd')
            testing_syr<double>(arg);
    }
    else if(!strcmp(function, "trtri"))
    {
        if(precision == 's')
            testing_trtri<float>(arg);
        else if(precision == 'd')
            testing_trtri<double>(arg);
    }
    else if(!strcmp(function, "trtri_batched"))
    {
        if(precision == 's')
            testing_trtri_batched<float>(arg);
        else if(precision == 'd')
            testing_trtri_batched<double>(arg);
    }
    else if(!strcmp(function, "geam"))
    {
        if(precision == 's')
            testing_geam<float>(arg);
        else if(precision == 'd')
            testing_geam<double>(arg);
    }
    else if(!strcmp(function, "set_get_vector"))
    {
        if(precision == 's')
            testing_set_get_vector<float>(arg);
        else if(precision == 'd')
            testing_set_get_vector<double>(arg);
    }
    else if(!strcmp(function, "set_get_matrix"))
    {
        if(precision == 's')
            testing_set_get_matrix<float>(arg);
        else if(precision == 'd')
            testing_set_get_matrix<double>(arg);
    }
#if BUILD_WITH_TENSILE
    else if(!strcmp(function, "gemm"))
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

        if(precision == 'h')
            testing_gemm<rocblas_half>(arg);
        else if(precision == 's')
            testing_gemm<float>(arg);
        else if(precision == 'd')
            testing_gemm<double>(arg);
    }
    else if(!strcmp(function, "gemm_ex"))
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

        if(precision == 'h')
            testing_gemm_strided_batched<rocblas_half>(arg);
        else if(precision == 's')
            testing_gemm_strided_batched<float>(arg);
        else if(precision == 'd')
            testing_gemm_strided_batched<double>(arg);
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
    else if(!strcmp(function, "trsm"))
    {
        if(precision == 's')
            testing_trsm<float>(arg);
        else if(precision == 'd')
            testing_trsm<double>(arg);
    }
    else if(!strcmp(function, "trsv"))
    {
        if(precision == 's')
            testing_trsv<float>(arg);
        else if(precision == 'd')
            testing_trsv<double>(arg);
    }
#endif
    else
    {
        printf("Invalid value for --function \n");
        return -1;
    }

    return 0;
}

int rocblas_bench_datafile()
{
    for(auto i = RocBLAS_TestData::begin(); i != RocBLAS_TestData::end(); ++i)
    {
        Arguments arg = *i;
        char precision;

        // disable unit_check in client benchmark, it is only used in gtest unit test
        arg.unit_check = 0;

        // enable timing check,otherwise no performance data collected
        arg.timing = 1;

        switch(arg.a_type)
        {
        case rocblas_datatype_f64_r: precision = 'd'; break;
        case rocblas_datatype_f32_r: precision = 's'; break;
        case rocblas_datatype_f16_r: precision = 'h'; break;
        case rocblas_datatype_f64_c: precision = 'z'; break;
        case rocblas_datatype_f32_c: precision = 'c'; break;
        case rocblas_datatype_f16_c: precision = 'k'; break;
        default: precision                     = 's'; break;
        }
        run_bench_test(arg.function, precision, arg);
    }

    test_cleanup::cleanup();
    return 0;
}

using namespace boost::program_options;

int main(int argc, char* argv[])
{
    Arguments arg;
    arg.unit_check =
        0;          // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.timing = 1; // enable timing check,otherwise no performance data collected

    std::string function;
    char precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string initialization;

    rocblas_int device_id;
    bool datafile = rocblas_parse_data(argc, argv);

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

        ("incx",
         value<rocblas_int>(&arg.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<rocblas_int>(&arg.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function)->default_value("gemv"),
         "BLAS function to test. Options: gemv, ger, syr, trsm, trsv, trmm, symv, syrk, syr2k")

        ("precision,r",
         value<char>(&precision)->default_value('s'), "Options: h,s,d,c,z")

        ("a_type",
         value<std::string>(&a_type)->default_value("f32_r"), "Precision of matrix A, only applicable to BLAS_EX. "
         "Options: f16_r,f32_r,f64_r,i8_r,i32_r")

        ("b_type",
         value<std::string>(&b_type)->default_value("f32_r"), "Precision of matrix B, only applicable to BLAS_EX. "
         "Options: f16_r,f32_r,f64_r,i8_r,i32_r")

        ("c_type",
         value<std::string>(&c_type)->default_value("f32_r"), "Precision of matrix C, only applicable to BLAS_EX. "
         "Options: f16_r,f32_r,f64_r,i8_r,i32_r")

        ("d_type",
         value<std::string>(&d_type)->default_value("f32_r"), "Precision of matrix D, only applicable to BLAS_EX. "
         "Options: f16_r,f32_r,f64_r,i8_r,i32_r")

        ("compute_type",
         value<std::string>(&compute_type)->default_value("f32_r"), "Precision of computation, only applicable to BLAS_EX. "
         "Options: f16_r,f32_r,f64_r,i8_r,i32_r")

        ("initialization",
         value<std::string>(&initialization)->default_value("rand_int"), "Intialize with random integers or trig functions sin and cos. "
         "Options: rand_int, trig_float")

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
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm
                                                                     // xtrmm xtrsv
        ("diag",
         value<char>(&arg.diag)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm xtrsv

        ("batch",
         value<rocblas_int>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines") // xtrsm xtrmm xgemm

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

        ("workspace_size",
         value<size_t>(&arg.workspace_size)->default_value(10),
         "extended precision gemm workspace size")

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
    {
        printf("Error: Invalid device ID. There may not be such device ID. Will exit \n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    if(datafile)
    {
        return rocblas_bench_datafile();
    }

    if(!strchr("hsdcz", tolower(precision)))
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    arg.a_type = string2rocblas_datatype(a_type);
    if(arg.a_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --a_type" << std::endl;
        return -1;
    }

    arg.b_type = string2rocblas_datatype(b_type);
    if(arg.b_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --b_type" << std::endl;
        return -1;
    }

    arg.c_type = string2rocblas_datatype(c_type);
    if(arg.c_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --c_type" << std::endl;
        return -1;
    }

    arg.d_type = string2rocblas_datatype(d_type);
    if(arg.d_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --d_type" << std::endl;
        return -1;
    }

    arg.compute_type = string2rocblas_datatype(compute_type);
    if(arg.compute_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --compute_type" << std::endl;
        return -1;
    }

    if(initialization == "rand_int")
    {
        arg.initialization = rocblas_initialization_random_int;
    }
    else if(initialization == "trig_float")
    {
        arg.initialization = rocblas_initialization_trig_float;
    }
    else
    {
        std::cerr << "Invalid value for --initialization" << std::endl;
        return -1;
    }

    if(arg.M < 0 || arg.N < 0 || arg.K < 0)
    {
        printf("Invalid matrix dimension\n");
    }

    return run_bench_test(function.c_str(), precision, arg);
}
