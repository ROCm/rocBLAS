/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_gemm_ex.hpp"
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

typedef std::tuple<vector<int>, vector<double>, vector<char>, vector<rocblas_precision>>
    gemm_ex_tuple;

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
// add/delete as a group
const vector<vector<int>> tiny_matrix_size_range = {
    {1, 1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6, 6}, {7, 9, 15, 17, 18, 19, 19},
};

const vector<vector<int>> small_known_bug = {
    {1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2},
    {3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4},
    {5, 5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6, 6},
    {7, 7, 7, 7, 7, 7, 7},
    {9, 9, 9, 9, 9, 9, 9},
    {10, 10, 10, 10, 10, 10, 10},
    {11, 11, 11, 11, 11, 11, 11},
    {12, 12, 12, 12, 12, 12, 12},
    {13, 13, 13, 13, 13, 13, 13},
    {14, 14, 14, 14, 14, 14, 14},
    {15, 15, 15, 15, 15, 15, 15},
};

const vector<vector<int>> small_multiple_8_matrix_size_range = {
    {8, 8, 8, 8, 8, 8, 8},
    {16, 16, 16, 16, 16, 16, 16},
    {24, 24, 24, 24, 24, 24, 24},
    {32, 32, 32, 32, 32, 32, 32},
    {40, 40, 40, 40, 40, 40, 40},
    {48, 48, 48, 48, 48, 48, 48},
    {56, 56, 56, 56, 56, 56, 56},
    {64, 64, 64, 64, 64, 64, 64},
    {72, 72, 72, 72, 72, 72, 72},
};

const vector<vector<int>> small_matrix_size_range = {
    {1, 1, 1, 1, 1, 1, 1},         {2, 2, 2, 2, 2, 2, 2},
    {3, 3, 3, 3, 3, 3, 3},         {4, 4, 4, 4, 4, 4, 4},
    {4, 4, 4, 5, 4, 4, 4},         {4, 4, 4, 4, 5, 4, 4},
    {8, 8, 8, 8, 9, 10, 11},       {5, 6, 7, 8, 9, 10, 11},
    {7, 5, 6, 8, 9, 10, 11},       {6, 7, 5, 8, 9, 10, 11},
    {4, 4, 4, 4, 4, 4, 5},         {4, 4, 4, 4, 4, 6, 5},
    {5, 5, 5, 5, 5, 5, 5},         {6, 6, 6, 6, 6, 6, 6},
    {7, 7, 7, 7, 7, 7, 7},         {8, 8, 8, 8, 8, 8, 8},
    {9, 9, 9, 9, 9, 9, 9},         {10, 10, 10, 10, 10, 10, 10},
    {11, 11, 11, 11, 11, 11, 11},  {12, 12, 12, 12, 12, 12, 12},
    {13, 13, 13, 13, 13, 13, 13},  {14, 14, 14, 14, 14, 14, 14},
    {15, 15, 15, 15, 15, 15, 15},  {16, 16, 16, 16, 16, 16, 16},
    {17, 17, 17, 17, 17, 17, 17},  {18, 18, 18, 18, 18, 18, 18},
    {19, 19, 19, 19, 19, 19, 19},  {20, 20, 20, 20, 20, 20, 20},
    {2, 3, 4, 5, 6, 7, 8},         {3, 4, 5, 6, 7, 8, 9},
    {4, 5, 6, 6, 6, 6, 6},         {5, 6, 7, 7, 8, 9, 9},
    {6, 7, 8, 10, 9, 8, 8},        {7, 8, 9, 11, 9, 10, 10},
    {8, 9, 10, 10, 11, 12, 12},    {9, 10, 11, 12, 11, 13, 13},
    {13, 12, 11, 15, 14, 13, 13},  {15, 16, 17, 17, 18, 19, 19},
    {18, 17, 16, 18, 18, 18, 18},  {16, 17, 18, 20, 19, 18, 18},
    {3, 33, 3, 33, 35, 35, 35},    {5, 6, 7, 9, 11, 13, 13},
    {10, 10, 20, 100, 21, 22, 22}, {500, 501, 502, 503, 604, 505, 505},
};

const vector<vector<int>> large_matrix_size_range = {
    {191, 193, 194, 195, 196, 197, 197},
    {639, 640, 347, 960, 961, 1062, 1062},
    {1000, 1001, 101, 2002, 1003, 1004, 1004},
    {925, 1026, 1027, 1028, 2029, 1031, 1031},
    {4011, 4012, 103, 4014, 4015, 4016, 4016},
};

const vector<vector<int>> chunk_matrix_size_range = {
    {24000, 256, 256, 24010, 256, 24000, 24000},
    {24000, 256, 256, 24000, 256, 24020, 24020},
    {256, 24001, 256, 256, 24030, 24000, 24000},
    {256, 24001, 256, 256, 24000, 24040, 24040},
};

const vector<vector<int>> NaN_matrix_size_range = {
    {5, 6, 7, 8, 9, 10, 10}, {4011, 4012, 111, 4013, 4014, 4015, 4015},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> alpha_beta_2_3_range = {
    {2.0, 3.0},
};

const vector<vector<double>> NaN_alpha_beta_range = {
    {1.0, 2.0},
};

const vector<vector<double>> alpha_beta_range = {
    {5.0, 0.0}, {0.0, 3.0}, {1.0, 3.0},
};

const vector<vector<double>> small_alpha_beta_range = {
    {1.0, 2.0},
};

const vector<vector<double>> full_alpha_beta_range = {
    {1.0, 0.0}, {-1.0, -1.0}, {2.0, 1.0}, {0.0, 1.0}};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgemm/dgemm,
const vector<vector<char>> small_transA_transB_range = {{'N', 'N'}};
const vector<vector<char>> transA_transB_range = {{'N', 'N'}, {'N', 'T'}, {'C', 'N'}, {'T', 'C'}};

// a_type, b_type, c_type, d_type, compute_type
const vector<vector<rocblas_precision>> precision_half = {{rocblas_precision_half,
                                                           rocblas_precision_half,
                                                           rocblas_precision_half,
                                                           rocblas_precision_half,
                                                           rocblas_precision_half}};

const vector<vector<rocblas_precision>> precision_hpa_half = {{rocblas_precision_half,
                                                               rocblas_precision_half,
                                                               rocblas_precision_half,
                                                               rocblas_precision_half,
                                                               rocblas_precision_single}};

const vector<vector<rocblas_precision>> precision_single = {{rocblas_precision_single,
                                                             rocblas_precision_single,
                                                             rocblas_precision_single,
                                                             rocblas_precision_single,
                                                             rocblas_precision_single}};

const vector<vector<rocblas_precision>> precision_double = {{rocblas_precision_double,
                                                             rocblas_precision_double,
                                                             rocblas_precision_double,
                                                             rocblas_precision_double,
                                                             rocblas_precision_double}};

const vector<vector<rocblas_precision>> precision_type_range = {{rocblas_precision_half,
                                                                 rocblas_precision_half,
                                                                 rocblas_precision_half,
                                                                 rocblas_precision_half,
                                                                 rocblas_precision_half},
                                                                {rocblas_precision_half,
                                                                 rocblas_precision_half,
                                                                 rocblas_precision_half,
                                                                 rocblas_precision_half,
                                                                 rocblas_precision_single},
                                                                {rocblas_precision_single,
                                                                 rocblas_precision_single,
                                                                 rocblas_precision_single,
                                                                 rocblas_precision_single,
                                                                 rocblas_precision_single},
                                                                {rocblas_precision_double,
                                                                 rocblas_precision_double,
                                                                 rocblas_precision_double,
                                                                 rocblas_precision_double,
                                                                 rocblas_precision_double}};

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

Arguments setup_gemm_ex_arguments(gemm_ex_tuple tup)
{
    vector<int> matrix_size                   = std::get<0>(tup);
    vector<double> alpha_beta                 = std::get<1>(tup);
    vector<char> transA_transB                = std::get<2>(tup);
    vector<rocblas_precision> precision_types = std::get<3>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M   = matrix_size[0];
    arg.N   = matrix_size[1];
    arg.K   = matrix_size[2];
    arg.lda = matrix_size[3];
    arg.ldb = matrix_size[4];
    arg.ldc = matrix_size[5];
    arg.ldd = matrix_size[6];

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    arg.alpha = alpha_beta[0];
    arg.beta  = alpha_beta[1];

    arg.transA_option = transA_transB[0];
    arg.transB_option = transA_transB[1];

    arg.timing = 0;

    arg.a_type       = precision_types[0];
    arg.b_type       = precision_types[1];
    arg.c_type       = precision_types[2];
    arg.d_type       = precision_types[3];
    arg.compute_type = precision_types[4];

    return arg;
}

// class parameterized_gemm_ex_NaN : public ::TestWithParam<gemm_ex_tuple>
//{
//    protected:
//    parameterized_gemm_ex_NaN() {}
//    virtual ~parameterized_gemm_ex_NaN() {}
//    virtual void SetUp() {}
//    virtual void TearDown() {}
//};
//
// TEST_P(parameterized_gemm_ex_NaN, rocblas_half)
//{
//    Arguments arg = setup_gemm_ex_arguments(GetParam());
//
//    testing_gemm_ex_NaN<rocblas_half>(arg);
//}
//
// TEST_P(parameterized_gemm_ex_NaN, float)
//{
//    Arguments arg = setup_gemm_ex_arguments(GetParam());
//
//    testing_gemm_ex_NaN<float>(arg);
//}
//
// TEST_P(parameterized_gemm_ex_NaN, double)
//{
//    Arguments arg = setup_gemm_ex_arguments(GetParam());
//
//    testing_gemm_ex_NaN<double>(arg);
//}
//
class parameterized_gemm_ex : public ::TestWithParam<gemm_ex_tuple>
{
    protected:
    parameterized_gemm_ex() {}
    virtual ~parameterized_gemm_ex() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm_ex, standard)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_ex_arguments(GetParam());

    //  rocblas_status status = testing_gemm_ex<float>(arg);
    rocblas_status status = testing_gemm_ex(arg);

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
        else if(arg.ldc < arg.M || arg.ldd < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

class parameterized_chunk_gemm_ex : public ::TestWithParam<gemm_ex_tuple>
{
    protected:
    parameterized_chunk_gemm_ex() {}
    virtual ~parameterized_chunk_gemm_ex() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_chunk_gemm_ex, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_ex_arguments(GetParam());

    rocblas_status status = testing_gemm_ex(arg);

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
        else if(arg.ldc < arg.M || arg.ldd < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

class parameterized_half_gemm_ex : public ::TestWithParam<gemm_ex_tuple>
{
    protected:
    parameterized_half_gemm_ex() {}
    virtual ~parameterized_half_gemm_ex() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// TEST(checkin_blas3_bad_arg, gemm_ex_half) { testing_gemm_ex_bad_arg<rocblas_half>(); }

// TEST(checkin_blas3_bad_arg, gemm_ex_float) { testing_gemm_ex_bad_arg<float>(); }

// TEST(checkin_blas3_bad_arg, gemm_ex_double) { testing_gemm_ex_bad_arg<double>(); }

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, K, lda, ldb, ldc}, {alpha, beta}, {transA, transB} }

// INSTANTIATE_TEST_CASE_P(rocblas_gemm_ex_beta_eq_0, parameterized_gemm_ex_NaN,
// INSTANTIATE_TEST_CASE_P(checkin_blas3_NaN,
//                        parameterized_gemm_ex_NaN,
//                        Combine(ValuesIn(NaN_matrix_size_range),
//                                ValuesIn(NaN_alpha_beta_range),
//                                ValuesIn(transA_transB_range)));

// THis function mainly test the scope of matrix_size. the scope of alpha_beta, transA_transB is
// small
// INSTANTIATE_TEST_CASE_P(daily_blas3_large,
//                        parameterized_gemm_ex,
//                        Combine(ValuesIn(large_matrix_size_range),
//                                ValuesIn(alpha_beta_range),
//                                ValuesIn(transA_transB_range)));

// THis function mainly test the scope of alpha_beta, transA_transB,.the scope of matrix_size_range
// is small

// INSTANTIATE_TEST_CASE_P(checkin_blas3_small_half,
//                        parameterized_gemm_ex,
//                        Combine(ValuesIn(small_matrix_size_range),
//                                ValuesIn(alpha_beta_range),
//                                ValuesIn(transA_transB_range),
//                                ValuesIn(precision_half)));

INSTANTIATE_TEST_CASE_P(known_bug_blas_ex_small_hpa_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_known_bug),
                                ValuesIn(small_alpha_beta_range),
                                ValuesIn(small_transA_transB_range),
                                ValuesIn(precision_hpa_half)));

INSTANTIATE_TEST_CASE_P(checkin_blas_ex_small_hpa_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_multiple_8_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_hpa_half)));

// INSTANTIATE_TEST_CASE_P(checkin_blas3_small_float,
//                        parameterized_gemm_ex,
//                        Combine(ValuesIn(small_matrix_size_range),
//                                ValuesIn(alpha_beta_range),
//                                ValuesIn(transA_transB_range),
//                                ValuesIn(precision_single)));

// INSTANTIATE_TEST_CASE_P(checkin_blas3_small_double,
//                        parameterized_gemm_ex,
//                        Combine(ValuesIn(small_matrix_size_range),
//                                ValuesIn(alpha_beta_range),
//                                ValuesIn(transA_transB_range),
//                                ValuesIn(precision_double)));

// INSTANTIATE_TEST_CASE_P(checkin_blas3_small,
//                        parameterized_gemm_ex,
//                        Combine(ValuesIn(small_matrix_size_range),
//                                ValuesIn(alpha_beta_range),
//                                ValuesIn(transA_transB_range),
//                                ValuesIn(precision_type_range)));

// INSTANTIATE_TEST_CASE_P(checkin_blas3_tiny,
//                        parameterized_gemm_ex,
//                        Combine(ValuesIn(tiny_matrix_size_range),
//                                ValuesIn(full_alpha_beta_range),
//                                ValuesIn(transA_transB_range)));
//
// INSTANTIATE_TEST_CASE_P(checkin_blas3_small,
//                        parameterized_half_gemm_ex,
//                        Combine(ValuesIn(small_matrix_size_range),
//                                ValuesIn(full_alpha_beta_range),
//                                ValuesIn(transA_transB_range)));
//
// INSTANTIATE_TEST_CASE_P(checkin_blas3_tiny,
//                        parameterized_half_gemm_ex,
//                        Combine(ValuesIn(tiny_matrix_size_range),
//                                ValuesIn(full_alpha_beta_range),
//                                ValuesIn(transA_transB_range)));
//
// INSTANTIATE_TEST_CASE_P(daily_blas3_large,
//                        parameterized_half_gemm_ex,
//                        Combine(ValuesIn(large_matrix_size_range),
//                                ValuesIn(alpha_beta_range),
//                                ValuesIn(transA_transB_range)));
//
// INSTANTIATE_TEST_CASE_P(daily_blas3_chunk,
//                        parameterized_chunk_gemm_ex,
//                        Combine(ValuesIn(chunk_matrix_size_range),
//                                ValuesIn(alpha_beta_2_3_range),
//                                ValuesIn(transA_transB_range)));
