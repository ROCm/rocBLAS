/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_set_get_vector.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<int, vector<int>> set_get_vector_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {M};
// add/delete as a group
const int small_M_range[] = {10, 600};

const int medium_M_range[] = {600000};

const int large_M_range[] = {1000000, 6000000};

// vector of vector, each triple is a {incx, incy, incb};
// add/delete this list in pairs, like {1, 1, 1}
const vector<vector<int>> small_incx_incy_incb_range = {{1, 1, 1},
                                                        {1, 1, 2},
                                                        {1, 1, 3},
                                                        {1, 2, 1},
                                                        {1, 2, 2},
                                                        {1, 2, 3},
                                                        {1, 3, 1},
                                                        {1, 3, 2},
                                                        {1, 3, 3},
                                                        {3, 1, 1},
                                                        {3, 1, 2},
                                                        {3, 1, 3},
                                                        {3, 2, 1},
                                                        {3, 2, 2},
                                                        {3, 2, 3},
                                                        {3, 3, 1},
                                                        {3, 3, 2},
                                                        {3, 3, 3}};

const vector<vector<int>> medium_incx_incy_incb_range = {
    {1, 1, 1}, {1, 1, 3}, {1, 3, 1}, {1, 3, 3}, {3, 1, 1}, {3, 1, 3}, {3, 3, 1}, {3, 3, 3}};

const vector<vector<int>> large_incx_incy_incb_range = {
    {1, 1, 1}, {1, 1, 3}, {1, 3, 1}, {1, 3, 3}, {3, 1, 1}, {3, 1, 3}, {3, 3, 1}, {3, 3, 3}};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS set_get_vector:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_set_get_vector_arguments(set_get_vector_tuple tup)
{

    int M                      = std::get<0>(tup);
    vector<int> incx_incy_incb = std::get<1>(tup);

    Arguments arg;

    // see the comments about vector_size_range above
    arg.M = M;

    // see the comments about matrix_size_range above
    arg.incx = incx_incy_incb[0];
    arg.incy = incx_incy_incb[1];
    arg.incb = incx_incy_incb[2];

    return arg;
}

class parameterized_set_vector_get_vector : public ::TestWithParam<set_get_vector_tuple>
{
    protected:
    parameterized_set_vector_get_vector() {}
    virtual ~parameterized_set_vector_get_vector() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// TEST_P(parameterized_set_vector_get_vector, set_get_vector_float)
TEST_P(parameterized_set_vector_get_vector, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_vector_arguments(GetParam());

    rocblas_status status = testing_set_get_vector<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incx <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incy <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incb <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else
        {
            EXPECT_EQ(rocblas_status_success, status);
        }
    }
}

TEST_P(parameterized_set_vector_get_vector, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_vector_arguments(GetParam());

    rocblas_status status = testing_set_get_vector<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incx <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incy <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.incb <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else
        {
            EXPECT_EQ(rocblas_status_success, status);
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_CASE_P(quick_auxiliary,
                        parameterized_set_vector_get_vector,
                        Combine(ValuesIn(small_M_range), ValuesIn(small_incx_incy_incb_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_auxiliary,
                        parameterized_set_vector_get_vector,
                        Combine(ValuesIn(medium_M_range), ValuesIn(medium_incx_incy_incb_range)));

INSTANTIATE_TEST_CASE_P(nightly_auxiliary,
                        parameterized_set_vector_get_vector,
                        Combine(ValuesIn(large_M_range), ValuesIn(large_incx_incy_incb_range)));
