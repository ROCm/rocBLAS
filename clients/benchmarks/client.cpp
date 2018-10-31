/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <iostream>
#include <cstring>
#include <cctype>
#include <boost/program_options.hpp>

#include "rocblas.h"
#include "utility.h"
#include "rocblas.hpp"
#include "testing_iamax.hpp"
#include "testing_iamin.hpp"
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
#if BUILD_WITH_TENSILE
#include "testing_gemm.hpp"
#include "testing_gemm_strided_batched.hpp"
#include "testing_gemm_kernel_name.hpp"
#include "testing_gemm_strided_batched_kernel_name.hpp"
#include "testing_trsm.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched_ex.hpp"
#endif

static int run_bench_test(const char *function, char precision, Arguments argus)
{
    static const char prefix[] = "testing_";
    if (!strncmp(function, prefix, sizeof(prefix)-1)) {
        function += sizeof(prefix)-1;
    }

    if (!strcmp(function,"asum"))
    {
        if (precision == 's')
            testing_asum<float, float>(argus);
        else if (precision == 'd')
            testing_asum<double, double>(argus);
    }
    else if (!strcmp(function,"axpy"))
    {
        if (precision == 'h')
            testing_axpy<rocblas_half>(argus);
        else if (precision == 's')
            testing_axpy<float>(argus);
        else if (precision == 'd')
            testing_axpy<double>(argus);
    }
    else if (!strcmp(function,"copy"))
    {
        if (precision == 's')
            testing_copy<float>(argus);
        else if (precision == 'd')
            testing_copy<double>(argus);
    }
    else if (!strcmp(function,"dot"))
    {
        if (precision == 's')
            testing_dot<float>(argus);
        else if (precision == 'd')
            testing_dot<double>(argus);
    }
    else if (!strcmp(function,"swap"))
    {
        if (precision == 's')
            testing_swap<float>(argus);
        else if (precision == 'd')
            testing_swap<double>(argus);
    }
    else if (!strcmp(function,"iamax"))
    {
        if (precision == 's')
            testing_iamax<float>(argus);
        else if (precision == 'd')
            testing_iamax<double>(argus);
    }
    else if (!strcmp(function,"iamin"))
    {
        if (precision == 's')
            testing_iamin<float>(argus);
        else if (precision == 'd')
            testing_iamin<double>(argus);
    }
    else if (!strcmp(function,"nrm2"))
    {
        if (precision == 's')
            testing_nrm2<float, float>(argus);
        else if (precision == 'd')
            testing_nrm2<double, double>(argus);
    }
    else if (!strcmp(function,"scal"))
    {
        if (precision == 's')
            testing_scal<float>(argus);
        else if (precision == 'd')
            testing_scal<double>(argus);
    }
    else if (!strcmp(function,"gemv"))
    {
        if (precision == 's')
            testing_gemv<float>(argus);
        else if (precision == 'd')
            testing_gemv<double>(argus);
    }
    else if (!strcmp(function,"ger"))
    {
        if (precision == 's')
            testing_ger<float>(argus);
        else if (precision == 'd')
            testing_ger<double>(argus);
    }
    else if (!strcmp(function,"syr"))
    {
        if (precision == 's')
            testing_syr<float>(argus);
        else if (precision == 'd')
            testing_syr<double>(argus);
    }
    else if (!strcmp(function,"trtri"))
    {
        if (precision == 's')
            testing_trtri<float>(argus);
        else if (precision == 'd')
            testing_trtri<double>(argus);
    }
    else if (!strcmp(function,"trtri_batched"))
    {
        if (precision == 's')
            testing_trtri_batched<float>(argus);
        else if (precision == 'd')
            testing_trtri_batched<double>(argus);
    }
    else if (!strcmp(function,"geam"))
    {
        if (precision == 's')
            testing_geam<float>(argus);
        else if (precision == 'd')
            testing_geam<double>(argus);
    }
    else if (!strcmp(function,"set_get_vector"))
    {
        if (precision == 's')
            testing_set_get_vector<float>(argus);
        else if (precision == 'd')
            testing_set_get_vector<double>(argus);
    }
    else if (!strcmp(function,"set_get_matrix"))
    {
        if (precision == 's')
            testing_set_get_matrix<float>(argus);
        else if (precision == 'd')
            testing_set_get_matrix<double>(argus);
    }
#if BUILD_WITH_TENSILE
    else if (!strcmp(function,"gemm"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;

        if (argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if (argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if (argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        if (precision == 'h')
            testing_gemm<rocblas_half>(argus);
        else if (precision == 's')
            testing_gemm<float>(argus);
        else if (precision == 'd')
            testing_gemm<double>(argus);
    }
    else if (!strcmp(function,"gemm_ex"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        rocblas_int min_ldd = argus.M;

        if (argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if (argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if (argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }
        if (argus.ldd < min_ldd)
        {
            std::cout << "rocblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            argus.ldd = min_ldd;
        }
        testing_gemm_ex(argus);
    }
    else if (!strcmp(function,"gemm_strided_batched"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        if (argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if (argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if (argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        //      rocblas_int min_stride_a =
        //          argus.transA_option == 'N' ? argus.K * argus.lda : argus.M * argus.lda;
        //      rocblas_int min_stride_b =
        //          argus.transB_option == 'N' ? argus.N * argus.ldb : argus.K * argus.ldb;
        //      rocblas_int min_stride_a =
        //          argus.transA_option == 'N' ? argus.K * argus.lda : argus.M * argus.lda;
        //      rocblas_int min_stride_b =
        //          argus.transB_option == 'N' ? argus.N * argus.ldb : argus.K * argus.ldb;
        rocblas_int min_stride_c = argus.ldc * argus.N;
        //      if (argus.stride_a < min_stride_a)
        //      {
        //          std::cout << "rocblas-bench INFO: stride_a < min_stride_a, set stride_a = " <<
        //          min_stride_a << std::endl;
        //          argus.stride_a = min_stride_a;
        //      }
        //      if (argus.stride_b < min_stride_b)
        //      {
        //          std::cout << "rocblas-bench INFO: stride_b < min_stride_b, set stride_b = " <<
        //          min_stride_b << std::endl;
        //          argus.stride_b = min_stride_b;
        //      }
        if (argus.stride_c < min_stride_c)
        {
            std::cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            argus.stride_c = min_stride_c;
        }

        if (precision == 'h')
            testing_gemm_strided_batched<rocblas_half>(argus);
        else if (precision == 's')
            testing_gemm_strided_batched<float>(argus);
        else if (precision == 'd')
            testing_gemm_strided_batched<double>(argus);
    }
    else if (!strcmp(function,"gemm_strided_batched_ex"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        rocblas_int min_ldd = argus.M;
        if (argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if (argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if (argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }
        if(argus.ldd < min_ldd)
        {
            std::cout << "rocblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            argus.ldd = min_ldd;
        }
        rocblas_int min_stride_c = argus.ldc * argus.N;
        if (argus.stride_c < min_stride_c)
        {
            std::cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            argus.stride_c = min_stride_c;
        }

        testing_gemm_strided_batched_ex(argus);
    }
    else if (!strcmp(function,"gemm_kernel_name"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        if (argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if (argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if (argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        if (precision == 'h')
            testing_gemm_strided_batched_kernel_name<rocblas_half>(argus);
        else if (precision == 's')
            testing_gemm_strided_batched_kernel_name<float>(argus);
        else if (precision == 'd')
            testing_gemm_strided_batched_kernel_name<double>(argus);
    }
    else if (!strcmp(function,"gemm_strided_batched_kernel_name"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        if (argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if (argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if (argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        //      rocblas_int min_stride_a =
        //          argus.transA_option == 'N' ? argus.K * argus.lda : argus.M * argus.lda;
        //      rocblas_int min_stride_b =
        //          argus.transB_option == 'N' ? argus.N * argus.ldb : argus.K * argus.ldb;
        rocblas_int min_stride_c = argus.ldc * argus.N;
        //      if (argus.stride_a < min_stride_a)
        //      {
        //          std::cout << "rocblas-bench INFO: stride_a < min_stride_a, set stride_a = " <<
        //          min_stride_a << std::endl;
        //          argus.stride_a = min_stride_a;
        //      }
        //      if (argus.stride_b < min_stride_b)
        //      {
        //          std::cout << "rocblas-bench INFO: stride_b < min_stride_b, set stride_b = " <<
        //          min_stride_b << std::endl;
        //          argus.stride_b = min_stride_b;
        //      }
        if (argus.stride_c < min_stride_c)
        {
            std::cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            argus.stride_c = min_stride_c;
        }

        if (precision == 'h')
            testing_gemm_strided_batched_kernel_name<rocblas_half>(argus);
        else if (precision == 's')
            testing_gemm_strided_batched_kernel_name<float>(argus);
        else if (precision == 'd')
            testing_gemm_strided_batched_kernel_name<double>(argus);
    }
    else if (!strcmp(function,"trsm"))
    {
        if (precision == 's')
            testing_trsm<float>(argus);
        else if (precision == 'd')
            testing_trsm<double>(argus);
    }
#endif
    else
    {
        printf("Invalid value for --function \n");
        return -1;
    }

   return 0;
}

static int rocblas_bench_datafile(const string& datafile)
{
    RocBLAS_PerfData::init(datafile);

    for (auto i = RocBLAS_PerfData::begin(); i != RocBLAS_PerfData::end(); ++i)
    {
        Arguments argus = *i;
        char precision;

        // disable unit_check in client benchmark, it is only used in gtest unit test
        argus.unit_check = 0;

        // enable timing check,otherwise no performance data collected
        argus.timing = 1;

        switch (argus.a_type) {
        case rocblas_datatype_f64_r: precision = 'd'; break;
        case rocblas_datatype_f32_r: precision = 's'; break;
        case rocblas_datatype_f16_r: precision = 'h'; break;
        case rocblas_datatype_f64_c: precision = 'z'; break;
        case rocblas_datatype_f32_c: precision = 'c'; break;
        case rocblas_datatype_f16_c: precision = 'h'; break;
        default:                     precision = 's'; break;
        }
        run_bench_test(argus.function, precision, argus);
    }

    return 0;
}

using namespace boost::program_options;

int main(int argc, char* argv[])
{
    Arguments argus;
    argus.unit_check =
        0;            // disable unit_check in client benchmark, it is only used in gtest unit test
    argus.timing = 1; // enable timing check,otherwise no performance data collected

    std::string function;
    char precision;
    char a_type;
    char b_type;
    char c_type;
    char d_type;
    char compute_type;

    rocblas_int device_id;
    std::string datafile;

    options_description desc("rocblas client command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<rocblas_int>(&argus.M)->default_value(128),
         "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows or columns in matrix.")

        ("sizen,n",
         value<rocblas_int>(&argus.N)->default_value(128),
         "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of rows or columns in matrix")

        ("sizek,k",
         value<rocblas_int>(&argus.K)->default_value(128),
         "Specific matrix size:sizek is only applicable to BLAS-3: the number of columns in "
         "A and rows in B.")

        ("lda",
         value<rocblas_int>(&argus.lda)->default_value(128),
         "Leading dimension of matrix A, is only applicable to BLAS-2 & BLAS-3.")

        ("ldb",
         value<rocblas_int>(&argus.ldb)->default_value(128),
         "Leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3.")

        ("ldc",
         value<rocblas_int>(&argus.ldc)->default_value(128),
         "Leading dimension of matrix C, is only applicable to BLAS-2 & BLAS-3.")

        ("ldd",
         value<rocblas_int>(&argus.ldd)->default_value(128),
         "Leading dimension of matrix D, is only applicable to BLAS-EX ")

        ("stride_a",
         value<rocblas_int>(&argus.stride_a)->default_value(128*128),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_b",
         value<rocblas_int>(&argus.stride_b)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_c",
         value<rocblas_int>(&argus.stride_c)->default_value(128*128),
         "Specific stride of strided_batched matrix C, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_d",
         value<rocblas_int>(&argus.stride_d)->default_value(128*128),
         "Specific stride of strided_batched matrix D, is only applicable to strided batched"
         "BLAS_EX: second dimension * leading dimension.")

        ("incx",
         value<rocblas_int>(&argus.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<rocblas_int>(&argus.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          value<double>(&argus.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<double>(&argus.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function)->default_value("gemv"),
         "BLAS function to test. Options: gemv, ger, syr, trsm, trmm, symv, syrk, syr2k")

        ("precision,r",
         value<char>(&precision)->default_value('s'), "Options: h,s,d,c,z")

        ("a_type",
         value<char>(&a_type)->default_value('s'), "Options: h,s,d,c,z"
         "Precision of matrix A, only applicable to BLAS_EX")

        ("b_type",
         value<char>(&b_type)->default_value('s'), "Options: h,s,d,c,z"
         "Precision of matrix B, only applicable to BLAS_EX")

        ("c_type",
         value<char>(&c_type)->default_value('s'), "Options: h,s,d,c,z"
         "Precision of matrix C, only applicable to BLAS_EX")

        ("d_type",
         value<char>(&d_type)->default_value('s'), "Options: h,s,d,c,z"
         "Precision of matrix D, only applicable to BLAS_EX")

        ("compute_type",
         value<char>(&compute_type)->default_value('s'), "Options: h,s,d,c,z"
         "Precision of computation, only applicable to BLAS_EX")

        ("transposeA",
         value<char>(&argus.transA_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&argus.transB_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         value<char>(&argus.side_option)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         value<char>(&argus.uplo_option)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm
                                                                     // xtrmm
        ("diag",
         value<char>(&argus.diag_option)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm
                                                                                          // xtrmm
        ("batch",
         value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines") // xtrsm xtrmm xgemm

        ("verify,v",
         value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<rocblas_int>(&argus.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("algo",
         value<uint32_t>(&argus.algo)->default_value(0),
         "extended precision gemm algorithm")

        ("solution_index",
         value<uint32_t>(&argus.solution_index)->default_value(0),
         "extended precision gemm solution index")

        ("flags",
         value<uint32_t>(&argus.flags)->default_value(10),
         "extended precision gemm flags")

        ("workspace_size",
         value<size_t>(&argus.workspace_size)->default_value(10),
         "extended precision gemm workspace size")

        ("data",
         value<string>(&datafile),
         "Data file to use for test arguments (overrides all of the above)")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("help,h", "produces this help message");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    // Device Query
    rocblas_int device_count = query_device_property();

    std::cout << std::endl;
    if (device_count <= device_id)
    {
        printf("Error: Invalid device ID. There may not be such device ID. Will exit \n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    if (datafile != "") {
        return rocblas_bench_datafile(datafile);
    }

    if (!strchr("hsdcz", tolower(precision)))
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    argus.a_type = char2rocblas_datatype(a_type);
    if (argus.a_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --a_type" << std::endl;
        return -1;
    }

    argus.b_type = char2rocblas_datatype(b_type);
    if (argus.b_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --b_type" << std::endl;
        return -1;
    }

    argus.c_type = char2rocblas_datatype(c_type);
    if (argus.c_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --c_type" << std::endl;
        return -1;
    }

    argus.d_type = char2rocblas_datatype(d_type);
    if (argus.d_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --d_type" << std::endl;
        return -1;
    }

    argus.compute_type = char2rocblas_datatype(compute_type);
    if (argus.compute_type == static_cast<rocblas_datatype>(-1))
    {
        std::cerr << "Invalid value for --compute_type" << std::endl;
        return -1;
    }

    if (argus.M < 0 || argus.N < 0 || argus.K < 0)
    {
        printf("Invalid matrix dimension\n");
    }

    return run_bench_test(function.c_str(), precision, argus);
}
