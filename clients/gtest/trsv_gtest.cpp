/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "rocblas.hpp"
#include "testing_trsv.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, int, vector<char> > trsv_tuple;

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

// vector of vector, each vector is a {M, lda};
// add/delete as a group
const vector<vector<int>> small_matrix_size_range = {
    {-1, 1}, {4, 4},{10,20}
};

const vector<vector<int>> medium_matrix_size_range = {
    {192, 192}, {600, 600}, {800, 801},
};

const vector<vector<int>> large_matrix_size_range = {
    {1000, 1000}, {2000, 2000}, {4011, 4011}, {8000, 8000},
};

// vector of vector, each item is a {incx};
// add/delete this list in pairs, like {1}
const vector<int> small_incx_range = {
    2, -1, 1, -1, 3, 0, 1, 0, 10, 100,
};

const vector<int> medium_incx_range = {
    2, -1, 1, -1, 3, 0, 1, 0, 10, 100,
};

const vector<int> large_incx_range = {
    2, -1, 3, 0, 1,
};

// vector of vector, each pair is a {uplo, transA, diag};
// uplo has two "Lower (L), Upper (U)"
// transA has three ("Nontranspose (N), conjTranspose(C), transpose (T)")
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) automatically
// in strsv/dtrsv,
// so we use 'C'
// Diag has two options ("Non-unit (N), Unit (U)")

// Each letter is capitalizied, e.g. do not use 'l', but use 'L' instead.

const vector<vector<char>> uplo_transA_diag_range = {
    {'L', 'N', 'N'}, {'L', 'N', 'U'}, {'U', 'C', 'N'},
};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 trsv:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 trsv does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_trsv_arguments(trsv_tuple tup)
{
    vector<int> matrix_size   = std::get<0>(tup);
    int         incx          = std::get<1>(tup);
    vector<char> uplo_transA_diag       = std::get<2>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M   = matrix_size[0];
    arg.lda = matrix_size[1];

    // see the comments about matrix_size_range above
    arg.incx = incx;
    arg.uplo_option   = uplo_transA_diag[0];
    arg.transA_option = uplo_transA_diag[1];
    arg.diag_option   = uplo_transA_diag[2];
    arg.timing = 0;

    return arg;
}

class parameterized_trsv : public ::TestWithParam<trsv_tuple>
{
    protected:
    parameterized_trsv() {}
    virtual ~parameterized_trsv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_trsv, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trsv_arguments(GetParam());

    rocblas_status status = testing_trsv<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.M || arg.lda < 1)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(0 == arg.incx)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(parameterized_trsv, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trsv_arguments(GetParam());

    rocblas_status status = testing_trsv<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.M || arg.lda < 1)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(0 == arg.incx)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a a vector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, lda}, {incx}, {transA} }


INSTANTIATE_TEST_CASE_P(quick_blas2_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas2_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas2_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));
