/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_scal.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;



//only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;
typedef std::tuple<int, vector<double>, vector<int>> blas1_tuple;


                    /* =====================================================================
                         README: This file contains testers to verify the correctness of
                                 BLAS routines with google test

                                 It is supposed to be played/used by advance / expert users
                                 Normal users only need to get the library routines without testers
                               =================================================================== */
/*
from ‘testing::internal::CartesianProductHolder3<testing::internal::ParamGenerator<int>,
testing::internal::ParamGenerator<std::vector<double> >,
testing::internal::ParamGenerator<std::vector<int> > >’

to ‘testing::internal::ParamGenerator<std::tuple<int, std::vector<double, std::allocator<double> >, std::vector<int, std::allocator<int> > > >’


*/


/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

int N_range[] = {-1, 10, 500, 1000};

//vector of vector, each pair is a {alpha, beta};
//add/delete this list in pairs, like {2.0, 4.0}
vector<vector<double>> alpha_beta_range = { {1.0, 0.0},
                                            {2.0, -1.0}
                                          };

//vector of vector, each pair is a {incx, incy};
//add/delete this list in pairs, like {1, 2}
//incx , incy must > 0, otherwise there is no real computation taking place,
//but throw a message, which will still be detected by gtest
vector<vector<int>> incx_incy_range = { {1, 1},
                                        {-1, -1},
                                       };


/* ===============Google Unit Test==================================================== */


/* =====================================================================
     BLAS-1: Scal, Swap, Copy
=================================================================== */

class test_scal: public :: TestWithParam <blas1_tuple>
{
    protected:
        test_scal(){}
        virtual ~test_scal(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};

class test_swap: public :: TestWithParam <blas1_tuple>{
    protected:
        test_swap(){}
        virtual ~test_swap(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};

class test_copy: public :: TestWithParam <blas1_tuple>{
    protected:
        test_copy(){}
        virtual ~test_copy(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
};


Arguments setup_blas1_arguments(blas1_tuple tup)
{

    int N = std::get<0>(tup);
    vector<double> alpha_beta = std::get<1>(tup);
    vector<int> incx_incy = std::get<2>(tup);

    //the first element of alpha_beta_range is always alpha, and the second is always beta
    double alpha = alpha_beta[0];
    double beta = alpha_beta[1];

    int incx = incx_incy[0];
    int incy = incx_incy[1];

    Arguments arg;
    arg.N = N;
    arg.alpha = alpha;
    arg.beta = beta;
    arg.incx = incx;
    arg.incy = incy;

    arg.timing = 0;//disable timing data print out. Not supposed to collect performance data in gtest

    return arg;
}


TEST_P(test_scal, scal_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );

    rocblas_status status = testing_scal<float>( arg );

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_success)
    {
        if( arg.N < 0 )
        {
            EXPECT_EQ(rocblas_invalid_dim, status);
        }
        else if( arg.incx < 0)
        {
            EXPECT_EQ(rocblas_invalid_incx, status);
        }
    }

}



TEST_P(test_swap, swap_float)
{
    // argument automatically transferred to testing_swap
    //testing_swap<float>( GetParam() );
    //testing_swap<double>( GetParam() );
}


TEST_P(test_copy, copy_float)
{
    // argument automatically transferred to testing_copy
    //testing_copy<float>( GetParam() );
    //testing_copy<double>( GetParam() );
}


//Values is for a single item; ValuesIn is for an array
//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { N, {alpha, beta}, {incx, incy} }
INSTANTIATE_TEST_CASE_P(accuracy_test_BLAS1_scal,
                        test_scal,
                        Combine(
                                  ValuesIn(N_range), ValuesIn(alpha_beta_range), ValuesIn(incx_incy_range)
                               )
                        );

/*
INSTANTIATE_TEST_CASE_P(accuracy_test_BLAS1_swap,
                        test_swap,
                        Combine(
                                  ValuesIn(N_range), ValuesIn(alpha_beta_range), ValuesIn(incx_incy_range)
                               )
                        );


INSTANTIATE_TEST_CASE_P(accuracy_test_BLAS1_copy,
                        test_copy,
                        Combine(
                                  ValuesIn(N_range), ValuesIn(alpha_beta_range), ValuesIn(incx_incy_range)
                               )
                        );
*/
