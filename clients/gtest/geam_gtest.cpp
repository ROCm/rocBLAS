/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_geam.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, vector<double>, vector<char>> geam_tuple;

// vector of vector, each vector is a {M, N, lda, ldb, ldc};
// add/delete as a group
const vector<vector<int>> small_matrix_size_range = {
    {-1, -1, -1, 1, 1},
    {0, 0, 1, 1, 1},
    {3, 33, 35, 35, 35},
    {5, 5, 5, 5, 5},
    {10, 11, 100, 12, 13},
    {600, 500, 601, 602, 603},
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 192, 192, 192},
    {192, 193, 194, 195, 196},
    {640, 641, 960, 961, 962},
    {1001, 1000, 1003, 1002, 1001},
};
const vector<vector<int>> huge_matrix_size_range = {
    {4011, 4012, 4012, 4013, 4014},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> small_alpha_beta_range = {
    {1.0, 0.0}, {0.0, 1.0}, {3.0, 1.0}, {-1.0, -1.0}, {-1.0, 0.0}, {0.0, -1.0},
};
const vector<vector<double>> large_alpha_beta_range = {
    {1.0, 0.0}, {1.0, 3.0}, {0.0, 1.0}, {0.0, 0.0},
};
const vector<vector<double>> huge_alpha_beta_range = {
    {1.0, 3.0},
};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgeam/dgeam,
const vector<vector<char>> transA_transB_range = {
    {'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'T', 'T'},
    //     {'C', 'N'},
    //     {'T', 'C'}
};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-3 GEAM:
=================================================================== */
/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_geam_arguments(geam_tuple tup)
{
    vector<int> matrix_size    = std::get<0>(tup);
    vector<double> alpha_beta  = std::get<1>(tup);
    vector<char> transA_transB = std::get<2>(tup);

    Arguments arg;

    // see the comments about small_matrix_size_range above
    arg.M   = matrix_size[0];
    arg.N   = matrix_size[1];
    arg.lda = matrix_size[2];
    arg.ldb = matrix_size[3];
    arg.ldc = matrix_size[4];

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    arg.alpha = alpha_beta[0];
    arg.beta  = alpha_beta[1];

    arg.transA_option = transA_transB[0];
    arg.transB_option = transA_transB[1];

    arg.timing = 0;

    return arg;
}

class parameterized_geam : public ::TestWithParam<geam_tuple>
{
    protected:
    parameterized_geam() {}
    virtual ~parameterized_geam() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_geam, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    rocblas_status status = testing_geam<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0 || arg.N < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.M : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else
        {
            EXPECT_EQ(rocblas_status_success, status);
        }
    }
}

TEST_P(parameterized_geam, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    rocblas_status status = testing_geam<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0 || arg.N < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.M : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
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
// The combinations are  { {M, N, lda, ldb, ldc}, {alpha, beta}, {transA, transB} }

TEST(checkin_blas3_bad_arg, geam_float) { testing_geam_bad_arg<float>(); }

INSTANTIATE_TEST_CASE_P(quick_blas3,
                        parameterized_geam,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3,
                        parameterized_geam,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3,
                        parameterized_geam,
                        Combine(ValuesIn(huge_matrix_size_range),
                                ValuesIn(huge_alpha_beta_range),
                                ValuesIn(transA_transB_range)));
