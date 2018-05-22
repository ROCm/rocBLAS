/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <iostream>
#include <stdio.h>
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
#endif

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    Arguments argus;
    argus.unit_check =
        0;            // disable unit_check in client benchmark, it is only used in gtest unit test
    argus.timing = 1; // enable timing check,otherwise no performance data collected

    std::string function;
    char precision;

    rocblas_int device_id;
    vector<rocblas_int> range = {-1, -1, -1};

    po::options_description desc("rocblas client command line options");
    desc.add_options()("help,h", "produces this help message")
        // clang-format off
        ("range",
         po::value<vector<rocblas_int>>(&range)->multitoken(),
         "Range matrix size testing: BLAS-3 benchmarking only. Accept three positive integers. "
         "Usage: "
         "--range start end step"
         ". e.g "
         "--range 100 1000 200"
         ". Diabled if not specified. If enabled, user specified m,n,k will be nullified")
        
        ("sizem,m",
         po::value<rocblas_int>(&argus.M)->default_value(128),
         "Specific matrix size testing: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows.")
        
        ("sizen,n",
         po::value<rocblas_int>(&argus.N)->default_value(128),
         "Specific matrix/vector size testing: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of columns")

        ("sizek,k",
         po::value<rocblas_int>(&argus.K)->default_value(128),
         "Specific matrix size testing:sizek is only applicable to BLAS-3: the number of columns in "
         "A & C  and rows in B.")

        ("lda",
         po::value<rocblas_int>(&argus.lda)->default_value(128),
         "Specific leading dimension of matrix A, is only applicable to "
         "BLAS-2 & BLAS-3: the number of rows.")

        ("ldb",
         po::value<rocblas_int>(&argus.ldb)->default_value(128),
         "Specific leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3: the number "
         "of rows.")

        ("ldc",
         po::value<rocblas_int>(&argus.ldc)->default_value(128),
         "Specific leading dimension of matrix C, is only applicable to BLAS-2 & "
         "BLAS-3: the number of rows.")

        ("bsa",
         po::value<rocblas_int>(&argus.bsa)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsb",
         po::value<rocblas_int>(&argus.bsb)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsc",
         po::value<rocblas_int>(&argus.bsc)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("incx",
         po::value<rocblas_int>(&argus.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         po::value<rocblas_int>(&argus.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha", 
          po::value<double>(&argus.alpha)->default_value(1.0), "specifies the scalar alpha")
        
        ("beta",
         po::value<double>(&argus.beta)->default_value(0.0), "specifies the scalar beta")
              
        ("function,f",
         po::value<std::string>(&function)->default_value("gemv"),
         "BLAS function to test. Options: gemv, ger, syr, trsm, trmm, symv, syrk, syr2k")
        
        ("precision,r", 
         po::value<char>(&precision)->default_value('s'), "Options: h,s,d,c,z")
        
        ("transposeA",
         po::value<char>(&argus.transA_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")
        
        ("transposeB",
         po::value<char>(&argus.transB_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")
        
        ("side",
         po::value<char>(&argus.side_option)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")
        
        ("uplo",
         po::value<char>(&argus.uplo_option)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm
                                                                     // xtrmm
        ("diag",
         po::value<char>(&argus.diag_option)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm
                                                                                          // xtrmm
        ("batch",
         po::value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines") // xtrsm xtrmm xgemm

        ("verify,v",
         po::value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         po::value<rocblas_int>(&argus.iters)->default_value(10),
         "Iterations to run inside timing loop")
        
        ("device",
         po::value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(precision != 'h' && precision != 's' && precision != 'd' && precision != 'c' &&
       precision != 'z')
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    // Device Query
    rocblas_int device_count = query_device_property();

    if(device_count <= device_id)
    {
        printf("Error: invalid device ID. There may not be such device ID. Will exit \n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }
    /* ============================================================================================
     */
    if(argus.M < 0 || argus.N < 0 || argus.K < 0)
    {
        printf("Invalide matrix dimension\n");
    }

    argus.start = range[0];
    argus.step  = range[1];
    argus.end   = range[2];

    if(function == "asum")
    {
        if(precision == 's')
            testing_asum<float, float>(argus);
        else if(precision == 'd')
            testing_asum<double, double>(argus);
    }
    else if(function == "axpy")
    {
        if(precision == 'h')
            testing_axpy<rocblas_half>(argus);
        else if(precision == 's')
            testing_axpy<float>(argus);
        else if(precision == 'd')
            testing_axpy<double>(argus);
    }
    else if(function == "copy")
    {
        if(precision == 's')
            testing_copy<float>(argus);
        else if(precision == 'd')
            testing_copy<double>(argus);
    }
    else if(function == "dot")
    {
        if(precision == 's')
            testing_dot<float>(argus);
        else if(precision == 'd')
            testing_dot<double>(argus);
    }
    else if(function == "swap")
    {
        if(precision == 's')
            testing_swap<float>(argus);
        else if(precision == 'd')
            testing_swap<double>(argus);
    }
    else if(function == "iamax")
    {
        if(precision == 's')
            testing_iamax<float>(argus);
        else if(precision == 'd')
            testing_iamax<double>(argus);
    }
    else if(function == "iamin")
    {
        if(precision == 's')
            testing_iamin<float>(argus);
        else if(precision == 'd')
            testing_iamin<double>(argus);
    }
    else if(function == "nrm2")
    {
        if(precision == 's')
            testing_nrm2<float, float>(argus);
        else if(precision == 'd')
            testing_nrm2<double, double>(argus);
    }
    else if(function == "scal")
    {
        if(precision == 's')
            testing_scal<float>(argus);
        else if(precision == 'd')
            testing_scal<double>(argus);
    }
    else if(function == "gemv")
    {
        if(precision == 's')
            testing_gemv<float>(argus);
        else if(precision == 'd')
            testing_gemv<double>(argus);
    }
    else if(function == "ger")
    {
        if(precision == 's')
            testing_ger<float>(argus);
        else if(precision == 'd')
            testing_ger<double>(argus);
    }
    else if(function == "syr")
    {
        if(precision == 's')
            testing_syr<float>(argus);
        else if(precision == 'd')
            testing_syr<double>(argus);
    }
    else if(function == "trtri")
    {
        if(precision == 's')
            testing_trtri<float>(argus);
        else if(precision == 'd')
            testing_trtri<double>(argus);
    }
    else if(function == "trtri_batched")
    {
        if(precision == 's')
            testing_trtri_batched<float>(argus);
        else if(precision == 'd')
            testing_trtri_batched<double>(argus);
    }
    else if(function == "geam")
    {
        if(precision == 's')
            testing_geam<float>(argus);
        else if(precision == 'd')
            testing_geam<double>(argus);
    }
    else if(function == "set_get_vector")
    {
        if(precision == 's')
            testing_set_get_vector<float>(argus);
        else if(precision == 'd')
            testing_set_get_vector<double>(argus);
    }
    else if(function == "set_get_matrix")
    {
        if(precision == 's')
            testing_set_get_matrix<float>(argus);
        else if(precision == 'd')
            testing_set_get_matrix<double>(argus);
    }
#if BUILD_WITH_TENSILE
    else if(function == "gemm")
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;

        if(argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if(argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if(argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        if(precision == 'h')
            testing_gemm<rocblas_half>(argus);
        else if(precision == 's')
            testing_gemm<float>(argus);
        else if(precision == 'd')
            testing_gemm<double>(argus);
    }
    else if(function == "gemm_strided_batched")
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        if(argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if(argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if(argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        rocblas_int min_bsa =
            argus.transA_option == 'N' ? argus.K * argus.lda : argus.M * argus.lda;
        rocblas_int min_bsb =
            argus.transB_option == 'N' ? argus.N * argus.ldb : argus.K * argus.ldb;
        rocblas_int min_bsc = argus.ldc * argus.N;
        if(argus.bsa < min_bsa)
        {
            std::cout << "rocblas-bench INFO: bsa < min_bsa, set bsa = " << min_bsa << std::endl;
            argus.bsa = min_bsa;
        }
        if(argus.bsb < min_bsb)
        {
            std::cout << "rocblas-bench INFO: bsb < min_bsb, set bsb = " << min_bsb << std::endl;
            argus.bsb = min_bsb;
        }
        if(argus.bsc < min_bsc)
        {
            std::cout << "rocblas-bench INFO: bsc < min_bsc, set bsc = " << min_bsc << std::endl;
            argus.bsc = min_bsc;
        }

        if(precision == 'h')
            testing_gemm_strided_batched<rocblas_half>(argus);
        else if(precision == 's')
            testing_gemm_strided_batched<float>(argus);
        else if(precision == 'd')
            testing_gemm_strided_batched<double>(argus);
    }
    else if(function == "gemm_kernel_name")
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        if(argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if(argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if(argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        if(precision == 'h')
            testing_gemm_strided_batched_kernel_name<rocblas_half>(argus);
        else if(precision == 's')
            testing_gemm_strided_batched_kernel_name<float>(argus);
        else if(precision == 'd')
            testing_gemm_strided_batched_kernel_name<double>(argus);
    }
    else if(function == "gemm_strided_batched_kernel_name")
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = argus.transA_option == 'N' ? argus.M : argus.K;
        rocblas_int min_ldb = argus.transB_option == 'N' ? argus.K : argus.N;
        rocblas_int min_ldc = argus.M;
        if(argus.lda < min_lda)
        {
            std::cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            argus.lda = min_lda;
        }
        if(argus.ldb < min_ldb)
        {
            std::cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            argus.ldb = min_ldb;
        }
        if(argus.ldc < min_ldc)
        {
            std::cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            argus.ldc = min_ldc;
        }

        rocblas_int min_bsa =
            argus.transA_option == 'N' ? argus.K * argus.lda : argus.M * argus.lda;
        rocblas_int min_bsb =
            argus.transB_option == 'N' ? argus.N * argus.ldb : argus.K * argus.ldb;
        rocblas_int min_bsc = argus.ldc * argus.N;
        if(argus.bsa < min_bsa)
        {
            std::cout << "rocblas-bench INFO: bsa < min_bsa, set bsa = " << min_bsa << std::endl;
            argus.bsa = min_bsa;
        }
        if(argus.bsb < min_bsb)
        {
            std::cout << "rocblas-bench INFO: bsb < min_bsb, set bsb = " << min_bsb << std::endl;
            argus.bsb = min_bsb;
        }
        if(argus.bsc < min_bsc)
        {
            std::cout << "rocblas-bench INFO: bsc < min_bsc, set bsc = " << min_bsc << std::endl;
            argus.bsc = min_bsc;
        }

        if(precision == 'h')
            testing_gemm_strided_batched_kernel_name<rocblas_half>(argus);
        else if(precision == 's')
            testing_gemm_strided_batched_kernel_name<float>(argus);
        else if(precision == 'd')
            testing_gemm_strided_batched_kernel_name<double>(argus);
    }
    else if(function == "trsm")
    {
        if(precision == 's')
            testing_trsm<float>(argus);
        else if(precision == 'd')
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
