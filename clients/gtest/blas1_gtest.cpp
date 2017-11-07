/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_iamax.hpp"
#include "testing_asum.hpp"
#include "testing_axpy.hpp"
#include "testing_copy.hpp"
#include "testing_dot.hpp"
#include "testing_nrm2.hpp"
#include "testing_scal.hpp"
#include "testing_swap.hpp"
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

When you see this error, do not hack this source code, hack the Makefile. It is due to compilation.

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

  int N_range[] = {
                    -1,
                    0,
                    5,
                    10,
                    500,
                    1000,
                    7111,
                    10000,
                  };

//vector of vector, each pair is a {alpha, beta};
//add/delete this list in pairs, like {2.0, 4.0}
vector<vector<double>> alpha_beta_range = { 
                                            {1.0, 0.0},
                                            {2.0, -1.0}
                                          };

//vector of vector, each pair is a {incx, incy};
//add/delete this list in pairs, like {1, 2}
//incx , incy must > 0, otherwise there is no real computation taking place,
//but throw a message, which will still be detected by gtest
vector<vector<int>> incx_incy_range = {
                                        {1, 1},
                                        {1, 2},
                                        {2, 1},
                                        { 1, -1},
                                        {-1,  1},
                                        {-1, -1},
                                       };


/* ===============Google Unit Test==================================================== */


/* =====================================================================
     BLAS-1:  iamax, asum, axpy, copy, dot, nrm2, scal, swap 
=================================================================== */

class blas1_gtest: public :: TestWithParam <blas1_tuple>
{
    protected:
        blas1_gtest(){}
        virtual ~blas1_gtest(){}
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

TEST(blas1_gtest, iamax_float_bad_arg)
{
    testing_iamax_bad_arg<float>();
}

TEST(blas1_gtest, asum_float_bad_arg)
{
    testing_asum_bad_arg<float, float>();
}

TEST(blas1_gtest, axpy_float_bad_arg)
{
    testing_axpy_bad_arg<float>();
}
TEST(blas1_gtest, copy_float_bad_arg)
{
    testing_copy_bad_arg<float>();
}
TEST(blas1_gtest, dot_float_bad_arg)
{
    testing_dot_bad_arg<float>();
}
TEST(blas1_gtest, scal_float_bad_arg)
{
    testing_scal_bad_arg<float>();
}
TEST(blas1_gtest, swap_float_bad_arg)
{
    testing_swap_bad_arg<float>();
}


TEST_P(blas1_gtest, iamax_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_iamax<float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incx < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(blas1_gtest, asum_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_asum<float, float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incx < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(blas1_gtest, axpy_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_axpy<float>( arg );
    EXPECT_EQ(rocblas_status_success, status);
}

TEST_P(blas1_gtest, axpy_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_axpy<double>( arg );
    EXPECT_EQ(rocblas_status_success, status);
}

TEST_P(blas1_gtest, axpy_half)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_axpy<rocblas_half>( arg );
    EXPECT_EQ(rocblas_status_success, status);
}

TEST_P(blas1_gtest, copy_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_copy<float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incx < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incy < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(blas1_gtest, dot_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_dot<float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else 
        {
            ASSERT_EQ(status, rocblas_status_success) << "incorrect return status";
        }
    }
}

TEST_P(blas1_gtest, dot_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_dot<double>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else 
        {
            ASSERT_EQ(status, rocblas_status_success) << "incorrect return status";
        }
    }
}

TEST_P(blas1_gtest, nrm2_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_nrm2<float, float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incx < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incy < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(blas1_gtest, scal_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_scal<float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( 0 == arg.N ){
            EXPECT_EQ(rocblas_status_success, status);
        }
        if( arg.N < 0 || arg.incx <= 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(blas1_gtest, swap_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments arg = setup_blas1_arguments( GetParam() );
    rocblas_status status = testing_swap<float>( arg );
    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success){
        if( arg.N < 0 ){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incx < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if( arg.incy < 0){
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


//Values is for a single item; ValuesIn is for an array
//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { N, {alpha, beta}, {incx, incy} }
INSTANTIATE_TEST_CASE_P(rocblas_blas1,
                        blas1_gtest,
                        Combine(
                                  ValuesIn(N_range), ValuesIn(alpha_beta_range), ValuesIn(incx_incy_range)
                               )
                        );
