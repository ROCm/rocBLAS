/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_potf2.hpp"
#include "utility.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, char> potf2_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {N, lda};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {
    {-1, 1}, {10, 20}, {500, 600},
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192}, {640, 960}, {1000, 1000}, {1024, 1024}, {2000, 2000},
};

// vector of char, each is an uplo, which can be "Lower (L) or Upper (U)"

// Each letter is capitalizied, e.g. do not use 'l', but use 'L' instead.

const vector<char> uplo_range = {'L', 'U'};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     LAPACK potf2:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_potf2_arguments(potf2_tuple tup)
{

    vector<int> matrix_size = std::get<0>(tup);
    char uplo               = std::get<1>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];

    arg.uplo_option = uplo;

    arg.timing = 0;

    return arg;
}

class potf2_gtest : public ::TestWithParam<potf2_tuple>
{
    protected:
    potf2_gtest() {}
    virtual ~potf2_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(potf2_gtest, potf2_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_potf2_arguments(GetParam());

    rocblas_status status = testing_potf2<float>(arg);

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
    }
}

TEST_P(potf2_gtest, potf2_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_potf2_arguments(GetParam());

    rocblas_status status = testing_potf2<double>(arg);

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
    }
}

// notice we are using vector of vector
// so each element in xxx_range is a vector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {N, lda}, uplo }

// This function mainly test the scope of matrix_size. the scope of uplo_range is
// small
// Testing order: uplo_range first, full_matrix_size last
// i.e fix the matrix size and alpha, test all the uplo_range first.
INSTANTIATE_TEST_CASE_P(daily_lapack,
                        potf2_gtest,
                        Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

// THis function mainly test the scope of uplo_range, the scope of
// matrix_size_range is small
INSTANTIATE_TEST_CASE_P(checkin_lapack,
                        potf2_gtest,
                        Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
