/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_ger.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

//only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, vector<int>, double> ger_tuple;

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


//vector of vector, each vector is a {M, N, lda};
//add/delete as a group
const
vector<vector<int>> matrix_size_range = {
                                        {-1,  1,  1},
                                        { 1, -1,  1},
                                        { 1,  1, -1},
                                        {10,  1,  9},
                                        { 0,  1,  1},
                                        { 1,  0,  1},
                                        { 1,  1,  0},
                                        {11, 11, 11},
                                        {16, 16, 16},
                                        {32, 32, 32},
                                        {65, 65, 65},
                                   /*   {10, 10, 2},       */
                                   /*   {600,500, 500},    */
                                        {1000, 1000, 1000},
                                        {2000, 2000, 2000},
                                        {4011, 4011, 4011},
                                        {8000, 8000, 8000}
                                       };

//vector of vector, each pair is a {incx, incy};
//add/delete this list in pairs, like {1, 1}
const
vector<vector<int>> incx_incy_range = {
                                            { 1,   1},
                                            { 0,  -1},
                                            { 0,   1},
                                            { 1,   0},
                                            { 2,   1},
                                            {10, 100}
                                          };

//vector, each entry is  {alpha};
//add/delete single values, like {2.0}
const
vector<double> alpha_range = {
                                            -0.5,
                                             2.0,
                                             0.0
                                          };


/* ===============Google Unit Test==================================================== */


/* =====================================================================
     BLAS-2 ger:
=================================================================== */

/* ============================Setup Arguments======================================= */

//Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
//Some routines may not touch/use certain "members" of objects "argus".
//like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
//That is fine. These testers & routines will leave untouched members alone.
//Do not use std::tuple to directly pass parameters to testers
//by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is not intuitive and error-prone

Arguments setup_ger_arguments(ger_tuple tup)
{

    vector<int> matrix_size = std::get<0>(tup);
    vector<int> incx_incy = std::get<1>(tup);
    double alpha = std::get<2>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M = matrix_size[0];
    arg.N = matrix_size[1];
    arg.lda = matrix_size[2];

    // see the comments about matrix_size_range above
    arg.incx = incx_incy[0];
    arg.incy = incx_incy[1];

    arg.alpha = alpha;

    arg.timing = 0;

    return arg;
}


class ger_gtest: public :: TestWithParam <ger_tuple>
{
    protected:
        ger_gtest(){}
        virtual ~ger_gtest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};


TEST_P(ger_gtest, ger_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.


    Arguments arg = setup_ger_arguments( GetParam() );

    rocblas_status status = testing_ger<float>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){

        if( arg.M < 0 || arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.M){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incx <= 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incy <= 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }

}

//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_CASE_P(rocblas_ger,
                        ger_gtest,
                        Combine(
                                  ValuesIn(matrix_size_range), ValuesIn(incx_incy_range), ValuesIn(alpha_range)
                               )
                        );
