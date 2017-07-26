/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_trsm.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

//only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, double, vector<char>> trsm_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */


/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */


//vector of vector, each vector is a {M, N, lda, ldb};
//add/delete as a group
const
vector<vector<int>> matrix_size_range = {
                                        {-1, -1, 1, 1},
                                        {10, 10, 20, 100},
                                        {600, 500, 600, 600},
                                        {1024, 1024, 1024, 1024}
                                       };

const
vector<vector<int>> full_matrix_size_range = {
                                        {192, 192, 192, 192},
                                        {640, 640, 960, 960},
                                        {1000, 1000, 1000, 1000},
                                        {2000, 2000, 2000, 2000},
                                       };

const
vector<double> alpha_range = {1.0, -5.0};


//vector of vector, each pair is a {side, uplo, transA, diag};
// side has two option "Lefe (L), Right (R)"
// uplo has two "Lower (L), Upper (U)"
// transA has three ("Nontranspose (N), conjTranspose(C), transpose (T)")
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) automatically in strsm/dtrsm,
// so we use 'C'
// Diag has two options ("Non-unit (N), Unit (U)")

//Each letter is capitalizied, e.g. do not use 'l', but use 'L' instead.

const
vector<vector<char>> side_uplo_transA_diag_range = {
                                        {'L', 'L', 'N', 'N'},
                                        {'R', 'L', 'N', 'N'},
                                        {'L', 'U', 'C', 'N'},
                                       };

//has all the 16 options
const
vector<vector<char>> full_side_uplo_transA_diag_range = {
                                        {'L', 'L', 'N', 'N'},
                                        {'R', 'L', 'N', 'N'},
                                        {'L', 'U', 'N', 'N'},
                                        {'R', 'U', 'N', 'N'},
                                        {'L', 'L', 'C', 'N'},
                                        {'R', 'L', 'C', 'N'},
                                        {'L', 'U', 'C', 'N'},
                                        {'R', 'U', 'C', 'N'},
                                        {'L', 'L', 'N', 'U'},
                                        {'R', 'L', 'N', 'U'},
                                        {'L', 'U', 'N', 'U'},
                                        {'R', 'U', 'N', 'U'},
                                        {'L', 'L', 'C', 'U'},
                                        {'R', 'L', 'C', 'U'},
                                        {'L', 'U', 'C', 'U'},
                                        {'R', 'U', 'C', 'U'},
                                       };

/* ===============Google Unit Test==================================================== */


/* =====================================================================
     BLAS-3 trsm:
=================================================================== */

/* ============================Setup Arguments======================================= */

//Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
//Some routines may not touch/use certain "members" of objects "argus".
//like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
//That is fine. These testers & routines will leave untouched members alone.
//Do not use std::tuple to directly pass parameters to testers
//by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is not intuitive and error-prone

Arguments setup_trsm_arguments(trsm_tuple tup)
{

    vector<int> matrix_size = std::get<0>(tup);
    double alpha = std::get<1>(tup);
    vector<char> side_uplo_transA_diag = std::get<2>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M = matrix_size[0];
    arg.N = matrix_size[1];
    arg.lda = matrix_size[2];
    arg.ldb = matrix_size[3];

    arg.alpha = alpha;

    arg.side_option   = side_uplo_transA_diag[0];
    arg.uplo_option   = side_uplo_transA_diag[1];
    arg.transA_option = side_uplo_transA_diag[2];
    arg.diag_option   = side_uplo_transA_diag[3];

    arg.timing = 0;

    return arg;
}


class trsm_gtest: public :: TestWithParam <trsm_tuple>
{
    protected:
        trsm_gtest(){}
        virtual ~trsm_gtest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};


TEST_P(trsm_gtest, trsm_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.


    Arguments arg = setup_trsm_arguments( GetParam() );

    rocblas_status status = testing_trsm<float>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){

        if( arg.M < 0 || arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.side_option == 'L' ? arg.lda < arg.M : arg.lda < arg.N){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb < arg.M){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }

}

TEST_P(trsm_gtest, trsm_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.


    Arguments arg = setup_trsm_arguments( GetParam() );

    rocblas_status status = testing_trsm<double>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){

        if( arg.M < 0 || arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.side_option == 'L' ? arg.lda < arg.M : arg.lda < arg.N){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb < arg.M){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }

}

//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda, ldb}, alpha, {side, uplo, transA, diag} }

//THis function mainly test the scope of matrix_size. the scope of side_uplo_transA_diag_range is small
//Testing order: side_uplo_transA_xx first, alpha_range second, full_matrix_size last
//i.e fix the matrix size and alpha, test all the side_uplo_transA_xx first.
INSTANTIATE_TEST_CASE_P(rocblas_trsm_matrix_size,
                        trsm_gtest,
                        Combine(
                                  ValuesIn(full_matrix_size_range), ValuesIn(alpha_range), ValuesIn(side_uplo_transA_diag_range)
                               )
                        );


//THis function mainly test the scope of  full_side_uplo_transA_diag_range,.the scope of matrix_size_range is small

INSTANTIATE_TEST_CASE_P(rocblas_trsm_scalar_transpose,
                        trsm_gtest,
                        Combine(
                                  ValuesIn(matrix_size_range), ValuesIn(alpha_range), ValuesIn(full_side_uplo_transA_diag_range)
                               )
                        );

