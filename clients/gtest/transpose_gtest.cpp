/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <functional>
#include "testing_transpose.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

//only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>> transpose_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */


/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the matrix.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */


// small sizes

//vector of vector, each triple is a {rows, cols, lda, ldb};
//add/delete this list in pairs, like {3, 4, 4, 5}

const
vector<vector<int>> size_range = {
                                            { 3, 3, 3, 3},
                                            { 3, 3, 4, 4},
                                            {10, 10, 10, 10},
                                            {100, 100, 98, 101},
                                            {1024, 1024, 1024, 1024},
                                            {1119, 111, 2000, 111},
                                            {5000, 5000, 6000, 7000}
                                                                                   
};




/* ===============Google Unit Test==================================================== */


/* =====================================================================
     BLAS transpose:
=================================================================== */

/* ============================Setup Arguments======================================= */

//Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
//Some routines may not touch/use certain "members" of objects "argus".
//like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
//That is fine. These testers & routines will leave untouched members alone.
//Do not use std::tuple to directly pass parameters to testers
//by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is not intuitive and error-prone

Arguments setup_transpose_arguments(transpose_tuple tup)
{

    vector<int> size = std::get<0>(tup);

    Arguments arg;

    arg.rows = size[0];
    arg.cols = size[1];

    arg.lda = size[2];
    arg.ldb = size[3];

    return arg;
}


class transpose_gtest: public :: TestWithParam <transpose_tuple>
{
    protected:
        transpose_gtest(){}
        virtual ~transpose_gtest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};


TEST_P(transpose_gtest, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_transpose_arguments( GetParam() );

    rocblas_status status = testing_transpose<float>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.rows < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.cols < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.rows){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb < arg.cols){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(transpose_gtest, float2)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_transpose_arguments( GetParam() );

    rocblas_status status = testing_transpose<float2>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.rows < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.cols < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.rows){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb < arg.cols){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p


INSTANTIATE_TEST_CASE_P(rocblas_transpose,
                        transpose_gtest,
                        Combine(
                                  ValuesIn(size_range)
                               )
                       );


