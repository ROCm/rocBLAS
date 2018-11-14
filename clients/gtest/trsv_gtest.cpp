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
    {-1, 1}, {10, 20},
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
    // 1,
};

const vector<int> medium_incx_range = {
    2, -1, 1, -1, 3, 0, 1, 0, 10, 100,
    // 1,
};

const vector<int> large_incx_range = {
    2, -1, 3, 0, 1,
    // 1,
};


//// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
//// strsv/dtrsv,
//const vector<char> transA_range = {
//    'N', 'T', 'C',
//};

// vector of vector, each pair is a {side, uplo, transA, diag};
// side has two option "Lefe (L), Right (R)"
// uplo has two "Lower (L), Upper (U)"
// transA has three ("Nontranspose (N), conjTranspose(C), transpose (T)")
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) automatically
// in strsm/dtrsm,
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
    arg.lda = matrix_size[2];

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

// TEST(trsv_Test, float)
// {
//     rocblas_int incx = 5;
//     char char_diag = 'U';
//     char char_uplo   = 'L';
//     std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
//     rocblas_handle handle = unique_ptr_handle->handle;

//     const rocblas_operation transA = rocblas_operation_none;
//     rocblas_diagonal diag    =rocblas_diagonal_unit;
//     rocblas_fill uplo        =rocblas_fill_lower;

//     int M = 640;
//     const rocblas_int lda = 640;
//     rocblas_int size_A = lda * M;
//     rocblas_int size_x= M*incx;
//     vector<float> hA(size_A);
//     vector<float> AAT(size_A);
//     vector<float> hb(size_x);
//     vector<float> hx(size_x);
//     vector<float> cpu_x_or_b(size_x);
//     vector<float> hx_or_b_1(size_x);
//     vector<float> hx_or_b_2(size_x);
    

//     auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * size_A),
//                                          rocblas_test::device_free};
//     auto dxorb_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * size_x),
//                                             rocblas_test::device_free};

//     float* dA      = (float*)dA_managed.get();
//     float* dx_or_b   = (float*)dxorb_managed.get();

//     rocblas_init<float>(hA, M, M, lda);

//         //  calculate AAT = hA * hA ^ T
//     cblas_gemm(rocblas_operation_none,
//                rocblas_operation_transpose,
//                M,
//                M,
//                M,
//                (float)1.0,
//                hA.data(),
//                lda,
//                hA.data(),
//                lda,
//                (float)0.0,
//                AAT.data(),
//                lda); 

//     //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
//     for(int i = 0; i < M; i++)
//     {
//         float t = 0.0;
//         for(int j = 0; j < M; j++)
//         {
//             hA[i + j * lda] = AAT[i + j * lda];
//             t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
//         }
//         hA[i + i * lda] = t;
//     }
//     //  calculate Cholesky factorization of SPD matrix hA
//     cblas_potrf(char_uplo, M, hA.data(), lda);

//     //  make hA unit diagonal if diag == rocblas_diagonal_unit
//     if(char_diag == 'U' || char_diag == 'u')
//     {
//         if('L' == char_uplo || 'l' == char_uplo)
//         {
//             for(int i = 0; i < M; i++)
//             {
//                 float diag = hA[i + i * lda];
//                 for(int j = 0; j <= i; j++)
//                 {
//                     hA[i + j * lda] = hA[i + j * lda] / diag;
//                 }
//             }
//         }
//         else
//         {
//             for(int j = 0; j < M; j++)
//             {
//                 float diag = hA[j + j * lda];
//                 for(int i = 0; i <= j; i++)
//                 {
//                     hA[i + j * lda] = hA[i + j * lda] / diag;
//                 }
//             }
//         }
//     }

//     rocblas_init<float>(hx, 1, M, incx); //incx = 1 for now

//     hb = hx;

//     // Calculate hb = hA*hx;
//     cblas_trmv<float>(
//         uplo, transA, diag, M, (const float*)hA.data(), lda, hb.data(), incx);

//     cpu_x_or_b = hb; // cpuXorB <- B
//     hx_or_b_1 = hb;
//     hx_or_b_2 = hb;

//     // copy data from CPU to device
//     CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(float) * size_A, hipMemcpyHostToDevice));
//     CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(float) * size_x, hipMemcpyHostToDevice));

//     float max_err_1 = 0.0;
//     float max_err_2 = 0.0;
//     float max_res_1 = 0.0;
//     float max_res_2 = 0.0;
//     float error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
//     float residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
//     float eps                     = std::numeric_limits<float>::epsilon();

//     // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
//     rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
//     rocblas_status status = rocblas_trsv<float>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);
//     CHECK_HIP_ERROR(
//         hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(float) * size_x, hipMemcpyDeviceToHost));

//     // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
//     rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
//     CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_2.data(), sizeof(float) * size_x, hipMemcpyHostToDevice))
//     status = rocblas_trsv<float>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);
//     CHECK_HIP_ERROR(hipMemcpy(hx_or_b_2.data(), dx_or_b, sizeof(float) * size_x, hipMemcpyDeviceToHost));


//     cblas_trsv<float>(uplo,
//                     transA,
//                     diag,
//                     M,
//                     (const float*)hA.data(),
//                     lda,
//                     cpu_x_or_b.data(),
//                     incx);

//     // unit_check_general<float>(1, size_x, incx, cpu_x_or_b.data(), hx_or_b_1.data());


//     float err_1 = 0.0;
//     float err_2 = 0.0;
//     for(int i = 0; i < M; i++)
//     {
//         if(hx[i * incx] != 0)
//         {
//             err_1 += std::abs((hx[i * incx] - hx_or_b_1[i * incx]) / hx[i * incx]);
//             err_2 += std::abs((hx[i * incx] - hx_or_b_2[i * incx]) / hx[i * incx]);
//         }
//         else
//         {
//             err_1 += std::abs(hx_or_b_1[i * incx]);
//             err_2 += std::abs(hx_or_b_2[i * incx]);
//         }
//     }
//     max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
//     max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;

//     trsm_err_res_check<float>(max_err_1, M, error_eps_multiplier, eps);
//     trsm_err_res_check<float>(max_err_2, M, error_eps_multiplier, eps);

//     cblas_trsv<float>(uplo,
//                 transA,
//                 diag,
//                 M,
//                 (const float*)hA.data(),
//                 lda,
//                 hx_or_b_1.data(),
//                 incx);
//     cblas_trsv<float>(uplo,
//                 transA,
//                 diag,
//                 M,
//                 (const float*)hA.data(),
//                 lda,
//                 hx_or_b_2.data(),
//                 incx);
//     // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
//     //                                                  = hx_or_b - hb
//     // res is the one norm of the scaled residual for each column

//     // float res_1 = 0.0;
//     // float res_2 = 0.0;
//     // for(int i = 0; i < M; i++)
//     // {
//     //     if(hb[i * incx] != 0)
//     //     {
//     //         res_1 += std::abs((hx_or_b_1[i * incx] - hb[i * incx]) / hb[i * incx]);
//     //         res_2 += std::abs((hx_or_b_2[i * incx] - hb[i * incx]) / hb[i * incx]);
//     //     }
//     //     else
//     //     {
//     //         res_1 += std::abs(hx_or_b_1[i * incx]);
//     //         res_2 += std::abs(hx_or_b_2[i * incx]);
//     //     }
//     // }
//     // max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
//     // max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;

//     // trsm_err_res_check<float>(max_res_1, M, residual_eps_multiplier, eps);
//     // trsm_err_res_check<float>(max_res_2, M, residual_eps_multiplier, eps);
// }

// notice we are using vector of vector
// so each elment in xxx_range is a a vector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, lda}, {incx}, {transA} }

// TEST(checkin_blas2_bad_arg, trsv_bad_arg_float) { testing_trsv_bad_arg<float>(); }

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

INSTANTIATE_TEST_CASE_P(quick_blas2_trsv2,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas2_trsv2,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas2_trsv2,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));                            

INSTANTIATE_TEST_CASE_P(quick_blas3_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blas4_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas4_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas4_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blas5_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas5_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas5_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blas6_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas6_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas6_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blasB2_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blasB2_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blasB2_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blasB2_trsv2,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blasB2_trsv2,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blasB2_trsv2,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));                            

INSTANTIATE_TEST_CASE_P(quick_blasB3_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blasB3_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blasB3_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blasB4_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blasB4_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blasB4_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blasB5_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blasB5_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blasB5_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(quick_blasB6_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(small_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blasB6_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(medium_incx_range),
                                ValuesIn(uplo_transA_diag_range)));

INSTANTIATE_TEST_CASE_P(nightly_blasB6_trsv,
                        parameterized_trsv,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_incx_range),
                                ValuesIn(uplo_transA_diag_range)));