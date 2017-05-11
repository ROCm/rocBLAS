/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_auxiliary.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

//only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<int, vector<int>> auxiliary_tuple;

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


//vector of vector, each vector is a {M};
//add/delete as a group
const
int M_range[] = { 600, 600000 };

//vector of vector, each triple is a {incx, incy, incd};
//add/delete this list in pairs, like {1, 1, 1}
const
vector<vector<int>> incx_incy_incd_range = {
                                            {1, 1, 1},
                                            {1, 1, 2},
                                            {1, 1, 3},
                                            {1, 2, 1},
                                            {1, 2, 2},
                                            {1, 2, 3},
                                            {1, 3, 1},
                                            {1, 3, 2},
                                            {1, 3, 3},
                                            {2, 1, 1},
                                            {2, 1, 2},
                                            {2, 1, 3},
                                            {2, 2, 1},
                                            {2, 2, 2},
                                            {2, 2, 3},
                                            {2, 3, 1},
                                            {2, 3, 2},
                                            {2, 3, 3},
                                            {3, 1, 1},
                                            {3, 1, 2},
                                            {3, 1, 3},
                                            {3, 2, 1},
                                            {3, 2, 2},
                                            {3, 2, 3},
                                            {3, 3, 1},
                                            {3, 3, 2},
                                            {3, 3, 3}
                                          };



/* ===============Google Unit Test==================================================== */


/* =====================================================================
     BLAS auxiliary:
=================================================================== */

/* ============================Setup Arguments======================================= */

//Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
//Some routines may not touch/use certain "members" of objects "argus".
//like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
//That is fine. These testers & routines will leave untouched members alone.
//Do not use std::tuple to directly pass parameters to testers
//by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is not intuitive and error-prone

Arguments setup_auxiliary_arguments(auxiliary_tuple tup)
{

    int M = std::get<0>(tup);
    vector<int> incx_incy_incd = std::get<1>(tup);

    Arguments arg;

    // see the comments about vector_size_range above
    arg.M = M;

    // see the comments about matrix_size_range above
    arg.incx = incx_incy_incd[0];
    arg.incy = incx_incy_incd[1];
    arg.incd = incx_incy_incd[2];

    return arg;
}


class auxiliary_gtest: public :: TestWithParam <auxiliary_tuple>
{
    protected:
        auxiliary_gtest(){}
        virtual ~auxiliary_gtest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};


TEST_P(auxiliary_gtest, auxiliary_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.


    Arguments arg = setup_auxiliary_arguments( GetParam() );

    rocblas_status status = testing_auxiliary<float>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){

        if( arg.M < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incx <= 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incy <= 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incd <= 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }

}

//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_CASE_P(rocblas_auxiliary,
                        auxiliary_gtest,
                        Combine(
                                  ValuesIn(M_range), ValuesIn(incx_incy_incd_range)
                               )
                        );
