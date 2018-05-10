/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <functional>
#include "testing_set_get_vector.hpp"
#include "testing_set_get_matrix.hpp"
#include "utility.h"

using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, vector<int>> set_get_matrix_tuple;

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

// small sizes

// vector of vector, each triple is a {M, N};
// add/delete this list in pairs, like {3, 4}

const vector<vector<int>> M_N_range = {{3, 3}, {3, 30}, {30, 3}};

// vector of vector, each triple is a {lda, ldb, ldc};
// add/delete this list in pairs, like {3, 4, 3}

const vector<vector<int>> lda_ldb_ldc_range = {
    {3, 3, 3}, {3, 3, 4}, {3, 3, 5}, {3, 4, 3}, {3, 4, 4},    {3, 4, 5},   {3, 5, 3},
    {3, 5, 4}, {3, 5, 5}, {5, 3, 3}, {5, 3, 4}, {5, 3, 5},    {5, 4, 3},   {5, 4, 4},
    {5, 4, 5}, {5, 5, 3}, {5, 5, 4}, {5, 5, 5}, {30, 30, 30}, {31, 32, 33}};

// large sizes   {{M, N},{lda,ldb,ldc}}
set_get_matrix_tuple gemm_values1{{300000, 21}, {300000, 300001, 300002}};
set_get_matrix_tuple gemm_values2{{300001, 22}, {300001, 300001, 300010}};
set_get_matrix_tuple gemm_values3{{300002, 23}, {300002, 300020, 300002}};
set_get_matrix_tuple gemm_values4{{300003, 24}, {300003, 300021, 300011}};
set_get_matrix_tuple gemm_values5{{300004, 25}, {300030, 300004, 300000}};
set_get_matrix_tuple gemm_values6{{300005, 26}, {300031, 300005, 300012}};
set_get_matrix_tuple gemm_values7{{300006, 27}, {300032, 300022, 300006}};
set_get_matrix_tuple gemm_values8{{300007, 28}, {300033, 300023, 300013}};

set_get_matrix_tuple gemm_values11{{20, 300000}, {20, 21, 22}};
set_get_matrix_tuple gemm_values12{{21, 300001}, {21, 21, 40}};
set_get_matrix_tuple gemm_values13{{22, 300011}, {22, 31, 22}};
set_get_matrix_tuple gemm_values14{{23, 300111}, {23, 32, 33}};
set_get_matrix_tuple gemm_values15{{24, 301111}, {34, 24, 24}};
set_get_matrix_tuple gemm_values16{{25, 311111}, {35, 25, 36}};
set_get_matrix_tuple gemm_values17{{26, 300002}, {37, 38, 36}};
set_get_matrix_tuple gemm_values18{{27, 300022}, {39, 40, 41}};

set_get_matrix_tuple gemm_values21{{3, 3000222}, {4, 4, 4}};

const vector<set_get_matrix_tuple> small_gemm_values_vec = {gemm_values1, gemm_values11};

const vector<set_get_matrix_tuple> large_gemm_values_vec = {gemm_values1,
                                                            gemm_values2,
                                                            gemm_values3,
                                                            gemm_values4,
                                                            gemm_values5,
                                                            gemm_values6,
                                                            gemm_values7,
                                                            gemm_values8,
                                                            gemm_values11,
                                                            gemm_values12,
                                                            gemm_values13,
                                                            gemm_values14,
                                                            gemm_values15,
                                                            gemm_values16,
                                                            gemm_values17,
                                                            gemm_values18,
                                                            gemm_values21};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS auxiliary:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_set_get_matrix_arguments(set_get_matrix_tuple tup)
{

    vector<int> M_N   = std::get<0>(tup);
    vector<int> lda_ldb_ldc = std::get<1>(tup);

    Arguments arg;

    arg.M = M_N[0];
    arg.N = M_N[1];

    arg.lda = lda_ldb_ldc[0];
    arg.ldb = lda_ldb_ldc[1];
    arg.ldc = lda_ldb_ldc[2];

    return arg;
}

class parameterized_set_matrix_get_matrix : public ::TestWithParam<set_get_matrix_tuple>
{
    protected:
    parameterized_set_matrix_get_matrix() {}
    virtual ~parameterized_set_matrix_get_matrix() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_set_matrix_get_matrix, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_matrix_arguments(GetParam());

    rocblas_status status = testing_set_get_matrix<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.N <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb < arg.M)
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

TEST_P(parameterized_set_matrix_get_matrix, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_matrix_arguments(GetParam());

    rocblas_status status = testing_set_get_matrix<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != rocblas_status_success)
    {
        if(arg.M < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.N <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc <= 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.lda < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldb < arg.M)
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
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_CASE_P(checkin_auxilliary,
                        parameterized_set_matrix_get_matrix,
                        Combine(ValuesIn(M_N_range), ValuesIn(lda_ldb_ldc_range)));

INSTANTIATE_TEST_CASE_P(checkin_auxilliary_2,
                        parameterized_set_matrix_get_matrix,
                        ValuesIn(small_gemm_values_vec));

INSTANTIATE_TEST_CASE_P(daily_auxilliary,
                        parameterized_set_matrix_get_matrix,
                        ValuesIn(large_gemm_values_vec));
