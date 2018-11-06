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

typedef std::tuple<vector<int>, vector<double>, vector<char>, vector<rocblas_datatype>>
    gemm_ex_tuple;

// clang-format off
// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
// add/delete as a group
const vector<vector<int>> small_matrix_size_range = {
    {1, 1,  1,  1,  1,  1,  1},
    {1, 2,  3,  4,  5,  6,  6},
    {7, 9, 15, 17, 18, 19, 19},
    {8, 1,  1,  8,  8,  8,  8},
    { 2,  2,  2,  2,  2,  2,  2},
    { 3,  3,  3,  3,  3,  3,  3},
    { 4,  4,  4,  4,  4,  4,  4},
    { 5,  5,  5,  5,  5,  5,  5},
    { 6,  6,  6,  6,  6,  6,  6},
    { 7,  7,  7,  7,  7,  7,  7},
    { 8,  8,  8,  8,  8,  8,  8},
    { 9,  9,  9,  9,  9,  9,  9},
    {10, 10, 10, 10, 10, 10, 10},
    {11, 11, 11, 11, 11, 11, 11},
    {12, 12, 12, 12, 12, 12, 12},
    {13, 13, 13, 13, 13, 13, 13},
    {14, 14, 14, 14, 14, 14, 14},
    {15, 15, 15, 15, 15, 15, 15},
    {16, 16, 16, 16, 16, 16, 16},
    {17, 17, 17, 17, 17, 17, 17},
    {18, 18, 18, 18, 18, 18, 18},
    {19, 19, 19, 19, 19, 19, 19},
    {20, 20, 20, 20, 20, 20, 20},
    { 2,  3,  4,  5,  6,  7,  8},
    { 3,  4,  5,  6,  7,  8,  9},
    { 4,  5,  6,  6,  6,  6,  6},
    { 5,  6,  7,  7,  8,  9,  9},
    { 6,  7,  8, 10,  9,  8,  8},
    { 7,  8,  9, 11,  9, 10, 10},
    { 8,  9, 10, 10, 11, 12, 12},
    { 9, 10, 11, 12, 11, 13, 13},
    {13, 12, 11, 15, 14, 13, 13},
    {15, 16, 17, 17, 18, 19, 19},
    {18, 17, 16, 18, 18, 18, 18},
    {16, 17, 18, 20, 19, 18, 18},
    { 8,  2,  2,  8,  8,  8,  8},
    { 8,  3,  3,  8,  8,  8,  8},
    { 8,  4,  4,  8,  8,  8,  8},
    { 8,  5,  5,  8,  8,  8,  8},
    { 8,  6,  6,  8,  8,  8,  8},
    { 8,  7,  7,  8,  8,  8,  8},
    { 8,  9,  9,  9,  9,  9,  9},
    { 8, 10, 10, 10, 10, 10, 10},
    { 8, 11, 11, 11, 11, 11, 11},
    { 8, 12, 12, 12, 12, 12, 12},
    { 8, 13, 13, 13, 13, 13, 13},
    { 8, 14, 14, 14, 14, 14, 14},
    { 8, 15, 15, 15, 15, 15, 15},
    {16, 15, 15, 16, 16, 16, 16},
    {16, 17, 17, 17, 17, 17, 17},
    {17, 16, 16, 17, 17, 17, 17},
    {16, 18, 18, 18, 18, 18, 18},
    {24, 24, 24, 24, 24, 24, 24},
    {32, 32, 32, 32, 32, 32, 32},
    {40, 40, 40, 40, 40, 40, 40},
    {48, 48, 48, 48, 48, 48, 48},
    {56, 56, 56, 56, 56, 56, 56},
    {64, 64, 64, 64, 64, 64, 64},
    {72, 72, 72, 72, 72, 72, 72},
};
const vector<vector<int>> medium_matrix_size_range = {
    {127, 127,  63, 127, 127, 127, 127},
    {128, 127,  63, 128, 128, 128, 128},
    {129, 127,  63, 129, 129, 129, 129},
    {127, 128,  63, 128, 127, 127, 127},
    {128, 128,  63, 128, 127, 127, 127},
    {129, 128,  63, 129, 129, 129, 129},
    {127, 129,  63, 129, 129, 129, 129},
    {128, 129,  63, 129, 129, 129, 129},
    {129, 129,  63, 129, 129, 129, 129},
    {127, 127,  64, 127, 127, 127, 127},
    {128, 127,  64, 128, 128, 128, 128},
    {129, 127,  64, 129, 129, 129, 129},
    {127, 128,  64, 128, 127, 127, 127},
    {128, 128,  64, 128, 127, 127, 127},
    {129, 128,  64, 129, 129, 129, 129},
    {127, 129,  64, 129, 129, 129, 129},
    {128, 129,  64, 129, 129, 129, 129},
    {129, 129,  64, 129, 129, 129, 129},
    {127, 127,  65, 127, 127, 127, 127},
    {128, 127,  65, 128, 128, 128, 128},
    {129, 127,  65, 129, 129, 129, 129},
    {127, 128,  65, 128, 127, 127, 127},
    {128, 128,  65, 128, 127, 127, 127},
    {129, 128,  65, 129, 129, 129, 129},
    {127, 129,  65, 129, 129, 129, 129},
    {128, 129,  65, 129, 129, 129, 129},
    {129, 129,  65, 129, 129, 129, 129},
    {191, 193, 194, 195, 196, 197, 197},
    {500, 501, 502, 503, 604, 505, 505},
    {639, 640, 347, 960, 961,1062,1062},
};

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> large_matrix_size_range = {
    {1000, 1001,  101, 2002, 1003, 1004, 1004},
    { 925, 1026, 1027, 1028, 2029, 1031, 1031},
    {4011, 4012,  103, 4014, 4015, 4016, 4016},
};

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> chunk_matrix_size_range = {
    {24000,   256, 256, 24010,   256, 24000, 24000},
    {24000,   256, 256, 24000,   256, 24020, 24020},
    {  256, 24001, 256,   256, 24030, 24000, 24000},
    {  256, 24001, 256,   256, 24000, 24040, 24040},
};

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> NaN_matrix_size_range = {
    {   5,    6,   7,    8,    9,   10,   10},
    {4011, 4012, 111, 4013, 4014, 4015, 4015},
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
const vector<vector<rocblas_datatype>> precision_half = {{ rocblas_datatype_f16_r,
                                                            rocblas_datatype_f16_r,
                                                            rocblas_datatype_f16_r,
                                                            rocblas_datatype_f16_r,
                                                            rocblas_datatype_f16_r  }};

const vector<vector<rocblas_datatype>> precision_hpa_half = {{ rocblas_datatype_f16_r,
                                                                rocblas_datatype_f16_r,
                                                                rocblas_datatype_f16_r,
                                                                rocblas_datatype_f16_r,
                                                                rocblas_datatype_f32_r  }};

const vector<vector<rocblas_datatype>> precision_single = {{ rocblas_datatype_f32_r,
                                                              rocblas_datatype_f32_r,
                                                              rocblas_datatype_f32_r,
                                                              rocblas_datatype_f32_r,
                                                              rocblas_datatype_f32_r  }};

const vector<vector<rocblas_datatype>> precision_double = {{ rocblas_datatype_f64_r,
                                                              rocblas_datatype_f64_r,
                                                              rocblas_datatype_f64_r,
                                                              rocblas_datatype_f64_r,
                                                              rocblas_datatype_f64_r  }};

const vector<vector<rocblas_datatype>> precision_int8 = {{ rocblas_datatype_i8_r,
                                                           rocblas_datatype_i8_r,
                                                           rocblas_datatype_i32_r,
                                                           rocblas_datatype_i32_r,
                                                           rocblas_datatype_i32_r  }};

const vector<vector<rocblas_datatype>> precision_type_range = {{rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r},
                                                                {rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f16_r,
                                                                 rocblas_datatype_f32_r},
                                                                {rocblas_datatype_f32_r,
                                                                 rocblas_datatype_f32_r,
                                                                 rocblas_datatype_f32_r,
                                                                 rocblas_datatype_f32_r,
                                                                 rocblas_datatype_f32_r},
                                                                {rocblas_datatype_f64_r,
                                                                 rocblas_datatype_f64_r,
                                                                 rocblas_datatype_f64_r,
                                                                 rocblas_datatype_f64_r,
                                                                 rocblas_datatype_f64_r}};
// clang-format on

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
    vector<int> matrix_size                  = std::get<0>(tup);
    vector<double> alpha_beta                = std::get<1>(tup);
    vector<char> transA_transB               = std::get<2>(tup);
    vector<rocblas_datatype> precision_types = std::get<3>(tup);

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

class parameterized_gemm_ex : public ::TestWithParam<gemm_ex_tuple>
{
    protected:
    parameterized_gemm_ex() {}
    virtual ~parameterized_gemm_ex() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST(Int8_Test, SmallOnesMatrix)
{
    // A simple test case consisting of multiply two 8x8 matrices full of ones
    // (Result is expected to be 8x8 matrix full of 8s

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    int M = 8;
    int N = 8;
    int K = 8;

    const rocblas_int lda = 8;
    const rocblas_int ldb = 8;
    const rocblas_int ldc = 8;
    const rocblas_int ldd = 8;

    std::unique_ptr<int8_t[]>  hA(new  int8_t[M * N]());
    std::unique_ptr<int8_t[]>  hB(new  int8_t[M * N]());
    std::unique_ptr<int32_t[]> hC(new int32_t[M * N]());
    std::unique_ptr<int32_t[]> hD(new int32_t[M * N]());

    rocblas_datatype a_type       = rocblas_datatype_i8_r;
    rocblas_datatype b_type       = rocblas_datatype_i8_r;
    rocblas_datatype c_type       = rocblas_datatype_i32_r;
    rocblas_datatype d_type       = rocblas_datatype_i32_r;
    rocblas_datatype compute_type = rocblas_datatype_i32_r;

    int32_t alpha = 1;
    int32_t beta  = 1;

    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    int32_t solution_index;
    rocblas_int flags;
    size_t* workspace_size = 0;
    void* workspace;

    rocblas_status status;

    // allocate memory on CPU
    for (int i = 0; i < M * N; i++)
    {
        hA[i] = 1;
        hB[i] = 1;
        hC[i] = 0;
        hD[i] = 0;
    }

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int8_t) * M * N),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int8_t) * M * N),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int32_t) * M * N),
                                         rocblas_test::device_free};
    auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int32_t) * M * N),
                                         rocblas_test::device_free};
    int8_t*  dA = (int8_t*)dA_managed.get();
    int8_t*  dB = (int8_t*)dB_managed.get();
    int32_t* dC = (int32_t*)dC_managed.get();
    int32_t* dD = (int32_t*)dD_managed.get();

    hipMemcpy(dA, hA.get(), M * N * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.get(), M * N * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.get(), M * N * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(dD, hD.get(), M * N * sizeof(int32_t), hipMemcpyHostToDevice);

    status = rocblas_gemm_ex(handle,
                             transA,
                             transB,
                             M,
                             N,
                             K,
                             &alpha,
                             dA,
                             a_type,
                             lda,
                             dB,
                             b_type,
                             ldb,
                             &beta,
                             dC,
                             c_type,
                             ldc,
                             dD,
                             d_type,
                             ldd,
                             compute_type,
                             algo,
                             solution_index,
                             flags,
                             workspace_size,
                             workspace);

    hipMemcpy(hD.get(), dD, M * N * sizeof(int32_t), hipMemcpyDeviceToHost);

    /*for (int i = 0; i < N; i++)
    {
        std::cout << std::endl;
        for(int j = 0; j < M; j++)
        {
            std::cout << hD[i*8 + j] << " ";
        }
    }
    std::cout << std::endl;*/

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            EXPECT_EQ(hD[i*8 + j], 8);
}

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

TEST(pre_checkin_blas_ex_bad_arg, float) { testing_gemm_ex_bad_arg(); }

//----small
INSTANTIATE_TEST_CASE_P(quick_blas_ex_small_hpa_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_hpa_half)));

INSTANTIATE_TEST_CASE_P(quick_blas_ex_small_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_half)));

INSTANTIATE_TEST_CASE_P(quick_blas_ex_small_single,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_single)));

INSTANTIATE_TEST_CASE_P(quick_blas_ex_small_double,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_double)));

INSTANTIATE_TEST_CASE_P(quick_blas_ex_small_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_int8)));
//----medium
INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_medium_hpa_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_hpa_half)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_medium_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_half)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_medium_float,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_single)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_medium_double,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_double)));
//----large
INSTANTIATE_TEST_CASE_P(nightly_blas_ex_large_hpa_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_hpa_half)));

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_large_half,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_half)));

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_large_float,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_single)));

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_large_double,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_double)));
//----chunk
INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_chunk,
                        parameterized_chunk_gemm_ex,
                        Combine(ValuesIn(chunk_matrix_size_range),
                                ValuesIn(alpha_beta_2_3_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_single)));
