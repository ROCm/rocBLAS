/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_gemm_strided_batched.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

typedef std::tuple<vector<int>, vector<double>, vector<char>, int> gemm_strided_batched_tuple;

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

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc};
// add/delete as a group, in batched gemm, the matrix is much smaller than standard gemm
const vector<vector<int>> matrix_size_range = {
    {-1, -1, -1, -1, 1, 1},
    {31, 33, 35, 101, 102, 103},
    {59, 61, 63, 129, 131, 137},
    {129, 130, 131, 132, 133, 134},
    {501, 502, 103, 504, 605, 506},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> alpha_beta_range = {
    {1.0, 0.0}, {-1.0, -1.0},
};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgemm_strided_batched/dgemm_strided_batched,
const vector<vector<char>> transA_transB_range = {{'N', 'N'}, {'N', 'T'}, {'C', 'N'}, {'T', 'C'}};

// number of gemms in batched gemm
const vector<int> batch_count_range = {
    -1, 0, 1, 3,
};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-3 gemm_strided_batched:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_gemm_strided_batched_arguments(gemm_strided_batched_tuple tup)
{

    vector<int> matrix_size    = std::get<0>(tup);
    vector<double> alpha_beta  = std::get<1>(tup);
    vector<char> transA_transB = std::get<2>(tup);
    int batch_count            = std::get<3>(tup);

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

    arg.batch_count = batch_count;
    arg.timing      = 0;

    return arg;
}

class gemm_strided_batched : public ::TestWithParam<gemm_strided_batched_tuple>
{
    protected:
    gemm_strided_batched() {}
    virtual ~gemm_strided_batched() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(gemm_strided_batched, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_strided_batched_arguments(GetParam());

    rocblas_status status = testing_gemm_strided_batched<float>(arg);

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
        else if(arg.batch_count < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(gemm_strided_batched, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_strided_batched_arguments(GetParam());

    rocblas_status status = testing_gemm_strided_batched<double>(arg);

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
        else if(arg.batch_count < 0)
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
// The combinations are  { {M, N, K, lda, ldb, ldc}, {alpha, beta}, {transA, transB}, {batch_count}
// }

INSTANTIATE_TEST_CASE_P(checkin_blas3,
                        gemm_strided_batched,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(batch_count_range)));
