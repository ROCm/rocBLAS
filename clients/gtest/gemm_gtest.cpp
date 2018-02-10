/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_gemm.hpp"
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

typedef std::tuple<vector<int>, vector<double>, vector<char>> gemm_tuple;

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc};
// add/delete as a group
const vector<vector<int>> tiny_matrix_size_range = {
    {1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6}, {7, 9, 15, 17, 18, 19},
};

const vector<vector<int>> small_matrix_size_range = {
    {-1, -1, -1, -1, 1, 1},
    {2, 2, 2, 2, 2, 2},
    {3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4},
    {5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6},
    {7, 7, 7, 7, 7, 7},
    {8, 8, 8, 8, 8, 8},
    {9, 9, 9, 9, 9, 9},
    {10, 10, 10, 10, 10, 10},
    {11, 11, 11, 11, 11, 11},
    {12, 12, 12, 12, 12, 12},
    {13, 13, 13, 13, 13, 13},
    {14, 14, 14, 14, 14, 14},
    {15, 15, 15, 15, 15, 15},
    {16, 16, 16, 16, 16, 16},
    {17, 17, 17, 17, 17, 17},
    {18, 18, 18, 18, 18, 18},
    {19, 19, 19, 19, 19, 19},
    {20, 20, 20, 20, 20, 20},
    {2, 3, 4, 5, 6, 7},
    {3, 4, 5, 6, 7, 8},
    {4, 5, 6, 6, 6, 6},
    {5, 6, 7, 7, 8, 9},
    {6, 7, 8, 10, 9, 8},
    {7, 8, 9, 11, 9, 10},
    {8, 9, 10, 10, 11, 12},
    {9, 10, 11, 12, 11, 13},
    {13, 12, 11, 15, 14, 13},
    {13, 14, 12, 12, 13, 14},
    {15, 16, 17, 17, 18, 19},
    {18, 17, 16, 18, 18, 18},
    {16, 17, 18, 20, 19, 18},
    {3, 33, 3, 33, 35, 35},
    {5, 6, 7, 9, 11, 13},
    {10, 10, 20, 100, 21, 22},
    {500, 501, 502, 503, 604, 505},
    {500, 501, 502, 203, 204, 205},
};

const vector<vector<int>> large_matrix_size_range = {
    {191, 193, 194, 195, 196, 197},
    {640, 640, 347, 960, 961, 962},
    {1000, 1001, 101, 1002, 1003, 1004},
    {1025, 1026, 1027, 1028, 1029, 1031},
    {4011, 4012, 103, 4014, 4015, 4016},
};

const vector<vector<int>> NaN_matrix_size_range = {
    {5, 6, 7, 8, 9, 10}, {4011, 4012, 111, 4013, 4014, 4015},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> NaN_alpha_beta_range = {
    {1.0, 0.0},
};

const vector<vector<double>> alpha_beta_range = {
    {5.0, 0.0}, {0.0, 3.0}, {1.0, 3.0},
};

const vector<vector<double>> full_alpha_beta_range = {
    {1.0, 0.0}, {-1.0, -1.0}, {2.0, 1.0}, {0.0, 1.0}};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgemm/dgemm,
const vector<vector<char>> transA_transB_range = {{'N', 'N'}, {'N', 'T'}, {'C', 'N'}, {'T', 'C'}};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-3 GEMM:
=================================================================== */
/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_gemm_arguments(gemm_tuple tup)
{
    vector<int> matrix_size    = std::get<0>(tup);
    vector<double> alpha_beta  = std::get<1>(tup);
    vector<char> transA_transB = std::get<2>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M   = matrix_size[0];
    arg.N   = matrix_size[1];
    arg.K   = matrix_size[2];
    arg.lda = matrix_size[3];
    arg.ldb = matrix_size[4];
    arg.ldc = matrix_size[5];

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    arg.alpha = alpha_beta[0];
    arg.beta  = alpha_beta[1];

    arg.transA_option = transA_transB[0];
    arg.transB_option = transA_transB[1];

    arg.timing = 0;

    return arg;
}

class parameterized_gemm_NaN : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_gemm_NaN() {}
    virtual ~parameterized_gemm_NaN() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm_NaN, rocblas_half)
{
    Arguments arg = setup_gemm_arguments(GetParam());

    testing_gemm_NaN<rocblas_half>(arg);
}

TEST_P(parameterized_gemm_NaN, float)
{
    Arguments arg = setup_gemm_arguments(GetParam());

    testing_gemm_NaN<float>(arg);
}

TEST_P(parameterized_gemm_NaN, double)
{
    Arguments arg = setup_gemm_arguments(GetParam());

    testing_gemm_NaN<double>(arg);
}

class parameterized_gemm : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_gemm() {}
    virtual ~parameterized_gemm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm, rocblas_half)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_arguments(GetParam());

    rocblas_status status = testing_gemm<rocblas_half>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(parameterized_gemm, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_arguments(GetParam());

    rocblas_status status = testing_gemm<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(parameterized_gemm, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_arguments(GetParam());

    rocblas_status status = testing_gemm<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST(checkin_blas3_bad_arg, gemm_half) { testing_gemm_bad_arg<rocblas_half>(); }

TEST(checkin_blas3_bad_arg, gemm_float) { testing_gemm_bad_arg<float>(); }

TEST(checkin_blas3_bad_arg, gemm_double) { testing_gemm_bad_arg<double>(); }

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, K, lda, ldb, ldc}, {alpha, beta}, {transA, transB} }

// INSTANTIATE_TEST_CASE_P(rocblas_gemm_beta_eq_0, parameterized_gemm_NaN,
INSTANTIATE_TEST_CASE_P(checkin_blas3_NaN,
                        parameterized_gemm_NaN,
                        Combine(ValuesIn(NaN_matrix_size_range),
                                ValuesIn(NaN_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

// THis function mainly test the scope of matrix_size. the scope of alpha_beta, transA_transB is
// small
INSTANTIATE_TEST_CASE_P(daily_blas3_large,
                        parameterized_gemm,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

// THis function mainly test the scope of alpha_beta, transA_transB,.the scope of matrix_size_range
// is small

INSTANTIATE_TEST_CASE_P(checkin_blas3_small,
                        parameterized_gemm,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(full_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(checkin_blas3_tiny,
                        parameterized_gemm,
                        Combine(ValuesIn(tiny_matrix_size_range),
                                ValuesIn(full_alpha_beta_range),
                                ValuesIn(transA_transB_range)));
