/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_syr.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, int, double, char> syr_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with N, and "lda must >= N". case "lda < N" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {N, lda};
// add/delete as a group
const vector<vector<int>> small_matrix_size_range = {{1, 1},
                                                     {-1, 1},
                                                     {1, -1},
                                                     {0, 1},
                                                     {1, 0},
                                                     {0, 0},
                                                     {10, 9},
                                                     {11, 12},
                                                     {16, 16},
                                                     {33, 32},
                                                     {65, 66},
                                                     {1000, 1000}};

const vector<vector<int>> large_matrix_size_range = {{2000, 2000}, {4011, 4011}, {8000, 8000}};

// vector of incx;
// add/delete single values
const vector<int> small_incx_range = {-1, 1, 0, 2, -2, 10, 99};
const vector<int> large_incx_range = {
    1000,
};

// vector, each entry is  {alpha};
// add/delete single values
const vector<double> small_alpha_range = {-0.5, 2.0, 0.0};
const vector<double> large_alpha_range = {
    1000.6,
};

// vector, each entry is  {uplo};
const vector<char> uplo_range = {
    'U', 'L',
};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 syr:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_syr_arguments(syr_tuple tup)
{

    vector<int> matrix_size = std::get<0>(tup);
    int incx                = std::get<1>(tup);
    double alpha            = std::get<2>(tup);
    char uplo               = std::get<3>(tup);

    Arguments arg;

    // see the comments about small_matrix_size_range above
    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];

    // see the comments about small_matrix_size_range above
    arg.incx = incx;

    arg.alpha = alpha;

    arg.uplo_option = uplo;

    arg.timing = 0;

    return arg;
}

class parameterized_syr : public ::TestWithParam<syr_tuple>
{
    protected:
    parameterized_syr() {}
    virtual ~parameterized_syr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_syr, parameterized_syr_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr_arguments(GetParam());

    rocblas_status status = testing_syr<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {

        if(arg.N < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incx == 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(parameterized_syr, parameterized_syr_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr_arguments(GetParam());

    rocblas_status status = testing_syr<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {

        if(arg.N < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incx == 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {N, lda}, {incx} {alpha} }

TEST(checkin_blas2_bad_arg, syr_float) { testing_syr_bad_arg<float>(); }

INSTANTIATE_TEST_CASE_P(checkin_blas2,
                        parameterized_syr,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(small_alpha_range),
                                ValuesIn(uplo_range)));

INSTANTIATE_TEST_CASE_P(daily_rocblas_blas2,
                        parameterized_syr,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(large_alpha_range),
                                ValuesIn(uplo_range)));
