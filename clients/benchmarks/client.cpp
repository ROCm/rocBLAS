/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <iostream>
#include <stdio.h>
#include <boost/program_options.hpp>

#include "rocblas.h"
#include "utility.h"
#include "rocblas.hpp"
#include "testing_scal.hpp"
#include "testing_dot.hpp"
#include "testing_nrm2.hpp"
#include "testing_asum.hpp"
#include "testing_amax.hpp"
#include "testing_gemv.hpp"
#include "testing_ger.hpp"
#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#if BUILD_WITH_TENSILE
    #include "testing_gemm.hpp"
    #include "testing_gemm_strided_batched.hpp"
    #include "testing_trsm.hpp"
#endif

namespace po = boost::program_options;


int main(int argc, char *argv[])
{
    Arguments argus;
    argus.unit_check = 0;//disable unit_check in client benchmark, it is only used in gtest unit test
    argus.timing = 1; //enable timing check,otherwise no performance data collected

    std::string function;
    char precision;

    rocblas_int device_id;
    vector<rocblas_int> range = {-1, -1, -1};

    po::options_description desc( "rocblas client command line options" );
    desc.add_options()
        ( "help,h", "produces this help message" )
        (   "range", po::value< vector<rocblas_int> >( &range )->multitoken(), "Range matrix size testing: BLAS-3 benchmarking only. Accept three positive integers. Usage: ""--range start end step"". e.g ""--range 100 1000 200"". Diabled if not specified. If enabled, user specified m,n,k will be nullified")
        ( "sizem,m", po::value<rocblas_int>( &argus.M )->default_value(128), "Specific matrix size testing: sizem is only applicable to BLAS-2 & BLAS-3: the number of rows." )
        ( "sizen,n", po::value<rocblas_int>( &argus.N )->default_value(128), "Specific matrix/vector size testing: BLAS-1: the length of the vector. BLAS-2 & BLAS-3: the number of columns" )
        ( "sizek,k", po::value<rocblas_int>( &argus.K )->default_value(128), "Specific matrix size testing:sizek is only applicable to BLAS-3: the number of columns in A & C  and rows in B." )
        ( "lda", po::value<rocblas_int>( &argus.lda )->default_value(128), "Specific leading dimension of matrix A, is only applicable to BLAS-2 & BLAS-3: the number of rows." )
        ( "ldb", po::value<rocblas_int>( &argus.ldb )->default_value(128), "Specific leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3: the number of rows." )
        ( "ldc", po::value<rocblas_int>( &argus.ldc )->default_value(128), "Specific leading dimension of matrix C, is only applicable to BLAS-2 & BLAS-3: the number of rows." )
        ( "alpha",   po::value<double>( &argus.alpha)->default_value(1.0), "specifies the scalar alpha" )
        ( "beta",    po::value<double>( &argus.beta )->default_value(0.0), "specifies the scalar beta" )
        ( "function,f", po::value<std::string>( &function )->default_value("gemv"), "BLAS function to test. Options: gemv, ger, trsm, trmm, symv, syrk, syr2k" )
        ( "precision,r", po::value<char>( &precision )->default_value('s'), "Options: s,d,c,z" )
        ( "transposeA", po::value<char>( &argus.transA_option )->default_value('N'), "N = no transpose, T = transpose, C = conjugate transpose" )
        ( "transposeB", po::value<char>( &argus.transB_option )->default_value('N'), "N = no transpose, T = transpose, C = conjugate transpose" )
        ( "side", po::value<char>( &argus.side_option )->default_value('L'), "L = left, R = right. Only applicable to certain routines" )
        ( "uplo", po::value<char>( &argus.uplo_option )->default_value('U'), "U = upper, L = lower. Only applicable to certain routines" )    // xsymv xsyrk xsyr2k xtrsm xtrmm
        ( "diag", po::value<char>( &argus.diag_option )->default_value('N'), "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines" ) // xtrsm xtrmm
        ( "batch", po::value<rocblas_int>( &argus.batch_count )->default_value(10), "Number of matrices. Only applicable to batched routines" ) // xtrsm xtrmm
        ( "verify,v", po::value<rocblas_int>(&argus.norm_check)->default_value(0), "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")
        ( "device", po::value<rocblas_int>(&device_id)->default_value(0), "Set default device to be used for subsequent program runs")
        ;


    po::variables_map vm;
    po::store( po::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );


    if( vm.count( "help" ) ){
        std::cout << desc << std::endl;
        return 0;
    }

    if( precision != 's' && precision != 'd' && precision != 'c' && precision != 'z' ){
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }


    //Device Query
    rocblas_int device_count = query_device_property();

    if(device_count <= device_id){
        printf("Error: invalid device ID. There may not be such device ID. Will exit \n");
        return -1;
    }
    else{
        set_device(device_id);
    }
    /* ============================================================================================ */
    if( argus.M < 0 || argus.N < 0 || argus.K < 0 ){
        printf("Invalide matrix dimension\n");
    }


    argus.start = range[0]; argus.step = range[1]; argus.end = range[2];


    if (function == "scal"){
        if (precision == 's')
            testing_scal<float>( argus );
        else if (precision == 'd')
            testing_scal<double>( argus );
    }
    else if (function == "dot"){
        if (precision == 's')
            testing_dot<float>( argus );
        else if (precision == 'd')
            testing_dot<double>( argus );
    }
    else if (function == "asum"){
        if (precision == 's')
            testing_asum<float, float>( argus );
        else if (precision == 'd')
            testing_asum<double, double>( argus );
    }
    else if (function == "nrm2"){
        if (precision == 's')
            testing_nrm2<float, float>( argus );
        else if (precision == 'd')
            testing_nrm2<double, double>( argus );
    }
    else if (function == "amax"){
        if (precision == 's')
            testing_amax<float>( argus );
        else if (precision == 'd')
            testing_amax<double>( argus );
    }
    else if (function == "gemv"){
        if (precision == 's')
            testing_gemv<float>( argus );
        else if (precision == 'd')
            testing_gemv<double>( argus );
    }
    else if (function == "ger"){
        if (precision == 's')
            testing_ger<float>( argus );
        else if (precision == 'd')
            testing_ger<double>( argus );
    }
    else if (function == "trtri"){
        if (precision == 's')
            testing_trtri<float>( argus );
        else if (precision == 'd')
            testing_trtri<double>( argus );
    }
    else if (function == "trtri_batched"){
        if (precision == 's')
            testing_trtri_batched<float>( argus );
        else if (precision == 'd')
            testing_trtri_batched<double>( argus );
    }
#if BUILD_WITH_TENSILE
    else if (function == "gemm"){

        //adjust dimension for GEMM routines
        argus.transA_option == 'N' ? argus.lda = argus.M : argus.lda = argus.K;
        argus.transB_option == 'N' ? argus.ldb = argus.K : argus.ldb = argus.N;
        argus.ldc = argus.M;

        if (precision == 's')
            testing_gemm<float>( argus );
        else if (precision == 'd')
            testing_gemm<double>( argus );
    }
    else if (function == "gemm_strided_batched"){
        //adjust dimension for GEMM routines
        argus.transA_option == 'N' ? argus.lda = argus.M : argus.lda = argus.K;
        argus.transB_option == 'N' ? argus.ldb = argus.K : argus.ldb = argus.N;
        argus.ldc = argus.M;
        if (precision == 's')
            testing_gemm_strided_batched<float>( argus );
        else if (precision == 'd')
            testing_gemm_strided_batched<double>( argus );
    }
    else if (function == "trsm"){
        if (precision == 's')
            testing_trsm<float>( argus );
        else if (precision == 'd')
            testing_trsm<double>( argus );
    }
#endif
    else{
        printf("Invalid value for --function \n");
        return -1;
    }


    return 0;
}
