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

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c};
// add/delete as a group, in batched gemm, the matrix is much smaller than standard gemm
// clang-format off
const vector<vector<int>> small_matrix_size_range = {
    { -1,  -1,  -1,  -1,   1,   1,      1,      1,     1},
    { 31,  33,  35, 101, 102, 103,   3605,   3605,   3605},
    { 59,  61,  63, 129, 131, 137,   8631,   8631,   8631},
    {  3,   3,   3,   3,   3,   3,      9,     9,      9},
    { 15,  15,  15,  15,  15,  15,    225,   225,    225},
    { 16,  16,  16,  16,  16,  16,    256,   256,    256},
    { 17,  17,  17,  17,  17,  17,    289,   289,    289},
    { 63,  63,  63,  63,  63,  63,   3969,  3969,   3969},
    { 64,  64,  64,  64,  64,  64,   4096,  4096,   4096},
    { 65,  65,  65,  65,  65,  65,   4225,  4225,   4225},
    {127, 127, 127, 127, 127, 127,  16129, 16129,  16129},
    {128, 128, 128, 128, 128, 128,  16384, 16384,  16384},
    {129, 129, 129, 129, 129, 129,  16641, 16641,  16641},
};

const vector<vector<int>> small_matrix_size_stride_a_range = {
    {  3,   3,   3,   3,   3,   3,    9,      9,      9},
    {  3,   3,   3,   3,   3,   3,    0,      9,      9},
    { 15,  15,  15,  15,  15,  15,  225,      0,    225},
    { 16,  16,  16,  16,  16,  16,    0,    256,    256},
    { 17,  17,  17,  17,  17,  17,  289,      0,    289},
    { 63,  63,  63,  63,  63,  63,    0,   3969,   3969},
    { 64,  64,  64,  64,  64,  64, 4096,      0,   4096},
    { 65,  65,  65,  65,  65,  65,     0,  4225,   4225},
    {127, 127, 127, 127, 127, 127, 16129,     0,  16129},
    {128, 128, 128, 128, 128, 128,     0, 16384,  16384},
    {129, 129, 129, 129, 129, 129, 16641,     0,  16641},
};
const vector<vector<int>> medium_matrix_size_range = {
    {129, 130, 131, 132, 133, 134,  17554,  17554,  17554},
    {255, 255, 255, 255, 255, 255,  65025,  65025,  65025},
    {256, 256, 256, 256, 256, 256,  65536,  65536,  65536},
    {257, 257, 257, 257, 257, 257,  66049,  66049,  66049},
    {501, 502, 103, 504, 605, 506, 340010, 340010, 340010},
};

const vector<vector<int>> medium_matrix_size_stride_a_range = {
    {255, 255, 255, 255, 255, 255, 65025,     0, 65025},
    {256, 256, 256, 256, 256, 256,     0, 65536, 65536},
    {257, 257, 257, 257, 257, 257, 66049,     0, 66049},
};

const vector<vector<int>> large_matrix_size_range = {
    {511, 511, 511,  511, 511, 511, 261121, 261121, 261121},
    {512, 512, 512,  512, 512, 512, 262144, 262144, 262144},
    {513, 513, 513,  513, 513, 513, 263169, 263169, 263169},
    {513, 514, 515,  516, 517, 518, 266771, 266772, 266773},
};
const vector<vector<int>> large_matrix_size_stride_a_range = {
    {511, 511, 511,  511, 511, 511,      0, 261121, 261121},
    {512, 512, 512,  512, 512, 512, 262144,      0, 262144},
    {513, 513, 513,  513, 513, 513,      0, 263169, 263169},
    {513, 514, 515,  516, 517, 518, 266771,      0, 266773},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}

const vector<vector<double>> alpha_beta_range          = { {1.0, 0.0}, {-1.0, -1.0}, {0.0, 1.0}, };
const vector<vector<double>> alpha_beta_stride_a_range = { {2.0, 3.0}};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgemm_strided_batched/dgemm_strided_batched,
const vector<vector<char>> transA_transB_range          = {{'N', 'N'}, {'N', 'T'}, {'C', 'N'}, {'T', 'C'}};
const vector<vector<char>> transA_transB_stride_a_range = {{'N', 'N'}};

// number of gemms in batched gemm
const vector<int> small_batch_count_range           = { -1,   0,  1, 3, };
const vector<int> medium_batch_count_range          = { 63,  64, 65,    };
const vector<int> small_batch_count_stride_a_range  = {  1,   3,        };
const vector<int> medium_batch_count_stride_a_range = {  31, 32, 33,    };

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c};
gemm_strided_batched_tuple db_sb_1{ {12544, 64, 64, 12544, 64, 12544, 802816, 0, 802816}, {1, 0}, {'N', 'N'}, 16};
gemm_strided_batched_tuple db_sb_2{ {12544, 64, 64, 12544, 64, 12544, 802816, 0, 802816}, {1, 0}, {'N', 'N'}, 8};
gemm_strided_batched_tuple db_sb_3{ {3136, 256, 64, 3136, 64, 3136, 200704, 0, 802816}, {1, 0}, {'N', 'N'}, 16};
gemm_strided_batched_tuple db_sb_4{ {3136, 256, 64, 3136, 64, 3136, 200704, 0, 802816}, {1, 0}, {'N', 'N'}, 8};
gemm_strided_batched_tuple db_sb_5{ {3136, 64, 256, 3136, 256, 3136, 802816, 0, 200704}, {1, 0}, {'N', 'N'}, 16};
gemm_strided_batched_tuple db_sb_6{ {3136, 64, 256, 3136, 256, 3136, 802816, 0, 200704}, {1, 0}, {'N', 'N'}, 8};
gemm_strided_batched_tuple db_sb_7{ {784, 128, 512, 784, 512, 784, 401408, 0, 100352}, {1, 0}, {'N', 'N'}, 16};
gemm_strided_batched_tuple db_sb_8{ {784, 128, 512, 784, 512, 784, 401408, 0, 100352}, {1, 0}, {'N', 'N'}, 8};
gemm_strided_batched_tuple db_sb_9{ {784, 512, 128, 784, 128, 784, 100352, 0, 401408}, {1, 0}, {'N', 'N'}, 16};
gemm_strided_batched_tuple db_sb_10{ {784, 512, 128, 784, 128, 784, 100352, 0, 401408}, {1, 0}, {'N', 'N'}, 8};
gemm_strided_batched_tuple db_sb_11{ {784, 64, 192, 784, 192, 784, 150528, 0, 50176}, {1, 0}, {'N', 'N'}, 16};
gemm_strided_batched_tuple db_sb_12{ {12544, 64, 64, 12544, 64, 12544, 802816, 0, 802816}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_13{ {12544, 64, 64, 12544, 64, 12544, 802816, 0, 802816}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_14{ {196, 1024, 256, 196, 1024, 196, 50176, 0, 200704}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_15{ {196, 1024, 256, 196, 1024, 196, 50176, 0, 200704}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_16{ {196, 256, 1024, 196, 256, 196, 200704, 0, 50176}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_17{ {196, 256, 1024, 196, 256, 196, 200704, 0, 50176}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_18{ {196, 256, 256, 196, 256, 196, 50176, 0, 50176}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_19{ {196, 256, 256, 196, 256, 196, 50176, 0, 50176}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_20{ {196, 512, 192, 196, 512, 196, 37632, 0, 100352}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_21{ {3136, 256, 64, 3136, 256, 3136, 200704, 0, 802816}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_22{ {3136, 256, 64, 3136, 256, 3136, 200704, 0, 802816}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_23{ {3136, 64, 256, 3136, 64, 3136, 802816, 0, 200704}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_24{ {3136, 64, 256, 3136, 64, 3136, 802816, 0, 200704}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_25{ {49, 2048, 512, 49, 2048, 49, 25088, 0, 100352}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_26{ {49, 2048, 512, 49, 2048, 49, 25088, 0, 100352}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_27{ {49, 512, 2048, 49, 512, 49, 100352, 0, 25088}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_28{ {49, 512, 2048, 49, 512, 49, 100352, 0, 25088}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_29{ {49, 512, 512, 49, 512, 49, 25088, 0, 25088}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_30{ {49, 512, 512, 49, 512, 49, 25088, 0, 25088}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_31{ {49, 832, 256, 49, 832, 49, 12544, 0, 40768}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_32{ {784, 128, 512, 784, 128, 784, 401408, 0, 100352}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_33{ {784, 128, 512, 784, 128, 784, 401408, 0, 100352}, {1, 0}, {'N', 'T'}, 8};
gemm_strided_batched_tuple db_sb_34{ {784, 192, 64, 784, 192, 784, 50176, 0, 150528}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_35{ {784, 512, 128, 784, 512, 784, 100352, 0, 401408}, {1, 0}, {'N', 'T'}, 16};
gemm_strided_batched_tuple db_sb_36{ {784, 512, 128, 784, 512, 784, 100352, 0, 401408}, {1, 0}, {'N', 'T'}, 8};

const vector<gemm_strided_batched_tuple> deepbench_sb_vec = {
    db_sb_1,  db_sb_2,  db_sb_3,  db_sb_4,  db_sb_5,  db_sb_6,  db_sb_7,  db_sb_8,  db_sb_9,
    db_sb_10, db_sb_11, db_sb_12, db_sb_13, db_sb_14, db_sb_15, db_sb_16, db_sb_17, db_sb_18,
    db_sb_19, db_sb_20, db_sb_21, db_sb_22, db_sb_23, db_sb_24, db_sb_25, db_sb_26, db_sb_27,
    db_sb_28, db_sb_29, db_sb_30, db_sb_31, db_sb_32, db_sb_33, db_sb_34, db_sb_35, db_sb_36};

gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_001 {{3025, 256, 64, 3025, 64, 3025, 193600, 0, 774400}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_002 {{3025, 64, 256, 3025, 256, 3025, 774400, 0, 193600}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_003 {{3025, 64, 64, 3025, 64, 3025, 193600, 0, 193600}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_004 {{3136, 256, 64, 3136, 64, 3136, 200704, 0, 802816}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_005 {{3136, 64, 256, 3136, 256, 3136, 802816, 0, 200704}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_006 {{3136, 64, 64, 3136, 64, 3136, 200704, 0, 200704}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_007 {{784, 128, 512, 784, 512, 784, 401408, 0, 100352}, {1, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp32_sb_008 {{784, 512, 128, 784, 128, 784, 100352, 0, 401408}, {1, 0}, {'N', 'N'}, 64};

const vector<gemm_strided_batched_tuple> conv_resnet50_fwd_fp32_sb = {
    conv_resnet50_fwd_fp32_sb_001, conv_resnet50_fwd_fp32_sb_002,
                                   conv_resnet50_fwd_fp32_sb_004,
    conv_resnet50_fwd_fp32_sb_005, conv_resnet50_fwd_fp32_sb_006, 
    conv_resnet50_fwd_fp32_sb_007, conv_resnet50_fwd_fp32_sb_008,
};
const vector<gemm_strided_batched_tuple> known_bug_conv_resnet50_fwd_fp32_sb = {
    conv_resnet50_fwd_fp32_sb_003,
};

gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_001 {{3025, 256, 64, 3025, 64, 3025, 193600, 0, 774400}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_002 {{3025, 64, 256, 3025, 256, 3025, 774400, 0, 193600}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_003 {{3025, 64, 64, 3025, 64, 3025, 193600, 0, 193600}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_004 {{3136, 256, 64, 3136, 64, 3136, 200704, 0, 802816}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_005 {{3136, 64, 256, 3136, 256, 3136, 802816, 0, 200704}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_006 {{3136, 64, 64, 3136, 64, 3136, 200704, 0, 200704}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_007 {{784, 128, 512, 784, 512, 784, 401408, 0, 100352}, {15360, 0}, {'N', 'N'}, 64};
gemm_strided_batched_tuple conv_resnet50_fwd_fp16_sb_008 {{784, 512, 128, 784, 128, 784, 100352, 0, 401408}, {15360, 0}, {'N', 'N'}, 64};

const vector<gemm_strided_batched_tuple> conv_resnet50_fwd_fp16_sb = {
    conv_resnet50_fwd_fp16_sb_001, conv_resnet50_fwd_fp16_sb_002,
    conv_resnet50_fwd_fp16_sb_003, conv_resnet50_fwd_fp16_sb_004, 
    conv_resnet50_fwd_fp16_sb_005, conv_resnet50_fwd_fp16_sb_006,
    conv_resnet50_fwd_fp16_sb_007, conv_resnet50_fwd_fp16_sb_008, 
};

gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_001 {{196, 1024, 256, 196, 1024, 196, 50176, 0, 200704}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_002 {{196, 256, 1024, 196, 256, 196, 200704, 0, 50176}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_003 {{3025, 256, 64, 3025, 256, 3025, 193600, 0, 774400}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_004 {{3025, 64, 256, 3025, 64, 3025, 774400, 0, 193600}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_005 {{3025, 64, 64, 3025, 64, 3025, 193600, 0, 193600}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_006 {{3136, 256, 64, 3136, 256, 3136, 200704, 0, 802816}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_007 {{3136, 64, 256, 3136, 64, 3136, 802816, 0, 200704}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_008 {{3136, 64, 64, 3136, 64, 3136, 200704, 0, 200704}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_009 {{49, 2048, 512, 49, 2048, 49, 25088, 0, 100352}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_010 {{49, 512, 2048, 49, 512, 49, 100352, 0, 25088}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_011 {{784, 128, 512, 784, 128, 784, 401408, 0, 100352}, {1, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp32_sb_012 {{784, 512, 128, 784, 512, 784, 100352, 0, 401408}, {1, 0}, {'N', 'T'}, 64};

const vector<gemm_strided_batched_tuple> conv_resnet50_bwddata_fp32_sb = {
    conv_resnet50_bwddata_fp32_sb_001, conv_resnet50_bwddata_fp32_sb_002, 
    conv_resnet50_bwddata_fp32_sb_003, conv_resnet50_bwddata_fp32_sb_004, 
    conv_resnet50_bwddata_fp32_sb_005, conv_resnet50_bwddata_fp32_sb_006, 
    conv_resnet50_bwddata_fp32_sb_007, conv_resnet50_bwddata_fp32_sb_008, 
    conv_resnet50_bwddata_fp32_sb_009, conv_resnet50_bwddata_fp32_sb_010, 
    conv_resnet50_bwddata_fp32_sb_011, conv_resnet50_bwddata_fp32_sb_012, 
};

gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_001 {{196, 1024, 256, 196, 1024, 196, 50176, 0, 200704}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_002 {{196, 256, 1024, 196, 256, 196, 200704, 0, 50176}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_003 {{3025, 256, 64, 3025, 256, 3025, 193600, 0, 774400}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_004 {{3025, 64, 256, 3025, 64, 3025, 774400, 0, 193600}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_005 {{3025, 64, 64, 3025, 64, 3025, 193600, 0, 193600}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_006 {{3136, 256, 64, 3136, 256, 3136, 200704, 0, 802816}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_007 {{3136, 64, 256, 3136, 64, 3136, 802816, 0, 200704}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_008 {{3136, 64, 64, 3136, 64, 3136, 200704, 0, 200704}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_009 {{49, 2048, 512, 49, 2048, 49, 25088, 0, 100352}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_010 {{49, 512, 2048, 49, 512, 49, 100352, 0, 25088}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_011 {{784, 128, 512, 784, 128, 784, 401408, 0, 100352}, {15360, 0}, {'N', 'T'}, 64};
gemm_strided_batched_tuple conv_resnet50_bwddata_fp16_sb_012 {{784, 512, 128, 784, 512, 784, 100352, 0, 401408}, {15360, 0}, {'N', 'T'}, 64};

const vector<gemm_strided_batched_tuple> conv_resnet50_bwddata_fp16_sb = {
    conv_resnet50_bwddata_fp16_sb_001, conv_resnet50_bwddata_fp16_sb_002, 
    conv_resnet50_bwddata_fp16_sb_003, conv_resnet50_bwddata_fp16_sb_004, 
    conv_resnet50_bwddata_fp16_sb_005, conv_resnet50_bwddata_fp16_sb_006, 
    conv_resnet50_bwddata_fp16_sb_007, conv_resnet50_bwddata_fp16_sb_008, 
    conv_resnet50_bwddata_fp16_sb_009, conv_resnet50_bwddata_fp16_sb_010, 
    conv_resnet50_bwddata_fp16_sb_011, conv_resnet50_bwddata_fp16_sb_012, 
};

gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_001 {{1225, 192, 384, 1225, 384, 1225, 470400, 0, 235200}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_002 {{1225, 64, 384, 1225, 384, 1225, 470400, 0, 78400}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_003 {{1225, 96, 384, 1225, 384, 1225, 470400, 0, 117600}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_004 {{289, 128, 1024, 289, 1024, 289, 295936, 0, 36992}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_005 {{289, 192, 1024, 289, 1024, 289, 295936, 0, 55488}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_006 {{289, 256, 1024, 289, 1024, 289, 295936, 0, 73984}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_007 {{289, 384, 1024, 289, 1024, 289, 295936, 0, 110976}, {15360, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp16_sb_008 {{5329, 64, 160, 5329, 160, 5329, 852640, 0, 341056}, {15360, 0}, {'N', 'N'}, 32};

const vector<gemm_strided_batched_tuple> conv_inception4_fwd_fp16_sb = {
    conv_inception4_fwd_fp16_sb_001, conv_inception4_fwd_fp16_sb_002, 
    conv_inception4_fwd_fp16_sb_003, conv_inception4_fwd_fp16_sb_004, 
    conv_inception4_fwd_fp16_sb_005, conv_inception4_fwd_fp16_sb_006, 
    conv_inception4_fwd_fp16_sb_007, conv_inception4_fwd_fp16_sb_008, 
};

gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_001 {{1225, 192, 384, 1225, 384, 1225, 470400, 0, 235200}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_002 {{1225, 64, 384, 1225, 384, 1225, 470400, 0, 78400}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_003 {{1225, 96, 384, 1225, 384, 1225, 470400, 0, 117600}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_004 {{289, 128, 1024, 289, 1024, 289, 295936, 0, 36992}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_005 {{289, 192, 1024, 289, 1024, 289, 295936, 0, 55488}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_006 {{289, 256, 1024, 289, 1024, 289, 295936, 0, 73984}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_007 {{289, 384, 1024, 289, 1024, 289, 295936, 0, 110976}, {1, 0}, {'N', 'N'}, 32};
gemm_strided_batched_tuple conv_inception4_fwd_fp32_sb_008 {{5329, 64, 160, 5329, 160, 5329, 852640, 0, 341056}, {1, 0}, {'N', 'N'}, 32};

const vector<gemm_strided_batched_tuple> conv_inception4_fwd_fp32_sb = {
    conv_inception4_fwd_fp32_sb_001, conv_inception4_fwd_fp32_sb_002, 
    conv_inception4_fwd_fp32_sb_003, conv_inception4_fwd_fp32_sb_004, 
    conv_inception4_fwd_fp32_sb_005, conv_inception4_fwd_fp32_sb_006, 
    conv_inception4_fwd_fp32_sb_007, conv_inception4_fwd_fp32_sb_008, 
};

gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_001 {{1225, 384, 192, 1225, 384, 1225, 235200, 0, 470400}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_002 {{1225, 384, 64, 1225, 384, 1225, 78400, 0, 470400}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_003 {{1225, 384, 96, 1225, 384, 1225, 117600, 0, 470400}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_004 {{289, 1024, 128, 289, 1024, 289, 36992, 0, 295936}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_005 {{289, 1024, 192, 289, 1024, 289, 55488, 0, 295936}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_006 {{289, 1024, 256, 289, 1024, 289, 73984, 0, 295936}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_007 {{289, 1024, 384, 289, 1024, 289, 110976, 0, 295936}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_008 {{5329, 160, 64, 5329, 160, 5329, 341056, 0, 852640}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_009 {{64, 1536, 256, 64, 1536, 64, 16384, 0, 98304}, {1, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp32_sb_010 {{64, 1536, 384, 64, 1536, 64, 24576, 0, 98304}, {1, 0}, {'N', 'T'}, 32};

const vector<gemm_strided_batched_tuple> conv_inception4_bwddata_fp32_sb = {
    conv_inception4_bwddata_fp32_sb_001, conv_inception4_bwddata_fp32_sb_002, 
    conv_inception4_bwddata_fp32_sb_003, conv_inception4_bwddata_fp32_sb_004, 
    conv_inception4_bwddata_fp32_sb_005, conv_inception4_bwddata_fp32_sb_006, 
    conv_inception4_bwddata_fp32_sb_007, conv_inception4_bwddata_fp32_sb_008, 
    conv_inception4_bwddata_fp32_sb_009, conv_inception4_bwddata_fp32_sb_010, 
};

gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_001 {{1225, 384, 192, 1225, 384, 1225, 235200, 0, 470400}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_002 {{1225, 384, 64, 1225, 384, 1225, 78400, 0, 470400}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_003 {{1225, 384, 96, 1225, 384, 1225, 117600, 0, 470400}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_004 {{289, 1024, 128, 289, 1024, 289, 36992, 0, 295936}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_005 {{289, 1024, 192, 289, 1024, 289, 55488, 0, 295936}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_006 {{289, 1024, 256, 289, 1024, 289, 73984, 0, 295936}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_007 {{289, 1024, 384, 289, 1024, 289, 110976, 0, 295936}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_008 {{5329, 160, 64, 5329, 160, 5329, 341056, 0, 852640}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_009 {{64, 1536, 256, 64, 1536, 64, 16384, 0, 98304}, {15360, 0}, {'N', 'T'}, 32};
gemm_strided_batched_tuple conv_inception4_bwddata_fp16_sb_010 {{64, 1536, 384, 64, 1536, 64, 24576, 0, 98304}, {15360, 0}, {'N', 'T'}, 32};

const vector<gemm_strided_batched_tuple> conv_inception4_bwddata_fp16_sb = {
    conv_inception4_bwddata_fp16_sb_001, conv_inception4_bwddata_fp16_sb_002, 
    conv_inception4_bwddata_fp16_sb_003, conv_inception4_bwddata_fp16_sb_004, 
    conv_inception4_bwddata_fp16_sb_005, conv_inception4_bwddata_fp16_sb_006, 
    conv_inception4_bwddata_fp16_sb_007, conv_inception4_bwddata_fp16_sb_008, 
    conv_inception4_bwddata_fp16_sb_009, conv_inception4_bwddata_fp16_sb_010, 
};

gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_001 {{121, 2048, 1, 121, 2048, 121, 121, 0, 247808}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_002 {{12544, 64, 1, 12544, 64, 12544, 12544, 0, 802816}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_003 {{144, 1024, 1, 144, 1024, 144, 144, 0, 147456}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_004 {{144, 256, 1, 144, 256, 144, 144, 0, 36864}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_005 {{144, 512, 1, 144, 512, 144, 144, 0, 73728}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_006 {{169, 256, 1, 169, 256, 169, 169, 0, 43264}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_007 {{16, 512, 1, 16, 512, 16, 16, 0, 8192}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_008 {{16, 528, 1, 16, 528, 16, 16, 0, 8448}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_009 {{16, 576, 1, 16, 576, 16, 16, 0, 9216}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_010 {{16, 608, 1, 16, 608, 16, 16, 0, 9728}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_011 {{196, 128, 1, 196, 128, 196, 196, 0, 25088}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_012 {{196, 192, 1, 196, 192, 196, 196, 0, 37632}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_013 {{196, 256, 1, 196, 256, 196, 196, 0, 50176}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_014 {{196, 480, 1, 196, 480, 196, 196, 0, 94080}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_015 {{196, 512, 1, 196, 512, 196, 196, 0, 100352}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_016 {{196, 528, 1, 196, 528, 196, 196, 0, 103488}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_017 {{196, 576, 1, 196, 576, 196, 196, 0, 112896}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_018 {{196, 608, 1, 196, 608, 196, 196, 0, 119168}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_019 {{196, 64, 1, 196, 64, 196, 196, 0, 12544}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_020 {{3136, 128, 1, 3136, 128, 3136, 3136, 0, 401408}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_021 {{3136, 256, 1, 3136, 256, 3136, 3136, 0, 802816}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_022 {{3136, 64, 1, 3136, 64, 3136, 3136, 0, 200704}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_023 {{32768, 480, 1, 32768, 480, 32768, 32768, 0, 15728640}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_024 {{49, 1024, 1, 49, 1024, 49, 49, 0, 50176}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_025 {{49, 1056, 1, 49, 1056, 49, 49, 0, 51744}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_026 {{49, 192, 1, 49, 192, 49, 49, 0, 9408}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_027 {{49, 512, 1, 49, 512, 49, 49, 0, 25088}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_028 {{49, 832, 1, 49, 832, 49, 49, 0, 40768}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_029 {{729, 64, 1, 729, 64, 729, 729, 0, 46656}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_030 {{784, 128, 1, 784, 128, 784, 784, 0, 100352}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_031 {{784, 192, 1, 784, 192, 784, 784, 0, 150528}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_032 {{784, 256, 1, 784, 256, 784, 784, 0, 200704}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_033 {{784, 320, 1, 784, 320, 784, 784, 0, 250880}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_034 {{784, 512, 1, 784, 512, 784, 784, 0, 401408}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_035 {{784, 64, 1, 784, 64, 784, 784, 0, 50176}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_036 {{8192, 480, 1, 8192, 480, 8192, 8192, 0, 3932160}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_037 {{8192, 512, 1, 8192, 512, 8192, 8192, 0, 4194304}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_038 {{8192, 528, 1, 8192, 528, 8192, 8192, 0, 4325376}, {1, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp32_sb_039 {{8192, 832, 1, 8192, 832, 8192, 8192, 0, 6815744}, {1, 0}, {'N', 'T'}, 1};

const vector<gemm_strided_batched_tuple> conv_ctest_bwddata_fp32_sb = {
    conv_ctest_bwddata_fp32_sb_001, conv_ctest_bwddata_fp32_sb_002,
    conv_ctest_bwddata_fp32_sb_003, conv_ctest_bwddata_fp32_sb_004,
    conv_ctest_bwddata_fp32_sb_005, conv_ctest_bwddata_fp32_sb_006,
    conv_ctest_bwddata_fp32_sb_007, conv_ctest_bwddata_fp32_sb_008,
    conv_ctest_bwddata_fp32_sb_009, conv_ctest_bwddata_fp32_sb_010,
    conv_ctest_bwddata_fp32_sb_011, conv_ctest_bwddata_fp32_sb_012,
    conv_ctest_bwddata_fp32_sb_013, conv_ctest_bwddata_fp32_sb_014,
    conv_ctest_bwddata_fp32_sb_015, conv_ctest_bwddata_fp32_sb_016,
    conv_ctest_bwddata_fp32_sb_017, conv_ctest_bwddata_fp32_sb_018,
    conv_ctest_bwddata_fp32_sb_019, conv_ctest_bwddata_fp32_sb_020,
    conv_ctest_bwddata_fp32_sb_021, conv_ctest_bwddata_fp32_sb_022,
    conv_ctest_bwddata_fp32_sb_023, conv_ctest_bwddata_fp32_sb_024,
    conv_ctest_bwddata_fp32_sb_025, conv_ctest_bwddata_fp32_sb_026,
    conv_ctest_bwddata_fp32_sb_027, conv_ctest_bwddata_fp32_sb_028,
    conv_ctest_bwddata_fp32_sb_029, conv_ctest_bwddata_fp32_sb_030,
    conv_ctest_bwddata_fp32_sb_031, conv_ctest_bwddata_fp32_sb_032,
    conv_ctest_bwddata_fp32_sb_033, conv_ctest_bwddata_fp32_sb_034,
    conv_ctest_bwddata_fp32_sb_035, conv_ctest_bwddata_fp32_sb_036,
    conv_ctest_bwddata_fp32_sb_037, conv_ctest_bwddata_fp32_sb_038,
    conv_ctest_bwddata_fp32_sb_039,
};

gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_001 {{121, 2048, 1, 121, 2048, 121, 121, 0, 247808}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_002 {{12544, 64, 1, 12544, 64, 12544, 12544, 0, 802816}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_003 {{144, 1024, 1, 144, 1024, 144, 144, 0, 147456}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_004 {{144, 256, 1, 144, 256, 144, 144, 0, 36864}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_005 {{144, 512, 1, 144, 512, 144, 144, 0, 73728}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_006 {{169, 256, 1, 169, 256, 169, 169, 0, 43264}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_007 {{16, 512, 1, 16, 512, 16, 16, 0, 8192}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_008 {{16, 528, 1, 16, 528, 16, 16, 0, 8448}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_009 {{16, 576, 1, 16, 576, 16, 16, 0, 9216}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_010 {{16, 608, 1, 16, 608, 16, 16, 0, 9728}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_011 {{196, 128, 1, 196, 128, 196, 196, 0, 25088}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_012 {{196, 192, 1, 196, 192, 196, 196, 0, 37632}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_013 {{196, 256, 1, 196, 256, 196, 196, 0, 50176}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_014 {{196, 480, 1, 196, 480, 196, 196, 0, 94080}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_015 {{196, 512, 1, 196, 512, 196, 196, 0, 100352}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_016 {{196, 528, 1, 196, 528, 196, 196, 0, 103488}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_017 {{196, 576, 1, 196, 576, 196, 196, 0, 112896}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_018 {{196, 608, 1, 196, 608, 196, 196, 0, 119168}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_019 {{196, 64, 1, 196, 64, 196, 196, 0, 12544}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_020 {{3136, 128, 1, 3136, 128, 3136, 3136, 0, 401408}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_021 {{3136, 256, 1, 3136, 256, 3136, 3136, 0, 802816}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_022 {{3136, 64, 1, 3136, 64, 3136, 3136, 0, 200704}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_023 {{32768, 480, 1, 32768, 480, 32768, 32768, 0, 15728640}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_024 {{49, 1024, 1, 49, 1024, 49, 49, 0, 50176}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_025 {{49, 1056, 1, 49, 1056, 49, 49, 0, 51744}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_026 {{49, 192, 1, 49, 192, 49, 49, 0, 9408}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_027 {{49, 512, 1, 49, 512, 49, 49, 0, 25088}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_028 {{49, 832, 1, 49, 832, 49, 49, 0, 40768}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_029 {{729, 64, 1, 729, 64, 729, 729, 0, 46656}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_030 {{784, 128, 1, 784, 128, 784, 784, 0, 100352}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_031 {{784, 192, 1, 784, 192, 784, 784, 0, 150528}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_032 {{784, 256, 1, 784, 256, 784, 784, 0, 200704}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_033 {{784, 320, 1, 784, 320, 784, 784, 0, 250880}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_034 {{784, 512, 1, 784, 512, 784, 784, 0, 401408}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_035 {{784, 64, 1, 784, 64, 784, 784, 0, 50176}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_036 {{8192, 480, 1, 8192, 480, 8192, 8192, 0, 3932160}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_037 {{8192, 512, 1, 8192, 512, 8192, 8192, 0, 4194304}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_038 {{8192, 528, 1, 8192, 528, 8192, 8192, 0, 4325376}, {15360, 0}, {'N', 'T'}, 1};
gemm_strided_batched_tuple conv_ctest_bwddata_fp16_sb_039 {{8192, 832, 1, 8192, 832, 8192, 8192, 0, 6815744}, {15360, 0}, {'N', 'T'}, 1};

const vector<gemm_strided_batched_tuple> conv_ctest_bwddata_fp16_sb = {
    conv_ctest_bwddata_fp16_sb_001, conv_ctest_bwddata_fp16_sb_002, 
    conv_ctest_bwddata_fp16_sb_003, conv_ctest_bwddata_fp16_sb_004, 
    conv_ctest_bwddata_fp16_sb_005, conv_ctest_bwddata_fp16_sb_006, 
    conv_ctest_bwddata_fp16_sb_007, conv_ctest_bwddata_fp16_sb_008, 
    conv_ctest_bwddata_fp16_sb_009, conv_ctest_bwddata_fp16_sb_010, 
    conv_ctest_bwddata_fp16_sb_011, conv_ctest_bwddata_fp16_sb_012, 
    conv_ctest_bwddata_fp16_sb_013, conv_ctest_bwddata_fp16_sb_014, 
    conv_ctest_bwddata_fp16_sb_015, conv_ctest_bwddata_fp16_sb_016, 
    conv_ctest_bwddata_fp16_sb_017, conv_ctest_bwddata_fp16_sb_018, 
    conv_ctest_bwddata_fp16_sb_019, conv_ctest_bwddata_fp16_sb_020, 
    conv_ctest_bwddata_fp16_sb_021, conv_ctest_bwddata_fp16_sb_022, 
    conv_ctest_bwddata_fp16_sb_023, conv_ctest_bwddata_fp16_sb_024, 
    conv_ctest_bwddata_fp16_sb_025, conv_ctest_bwddata_fp16_sb_026, 
    conv_ctest_bwddata_fp16_sb_027, conv_ctest_bwddata_fp16_sb_028, 
    conv_ctest_bwddata_fp16_sb_029, conv_ctest_bwddata_fp16_sb_030, 
    conv_ctest_bwddata_fp16_sb_031, conv_ctest_bwddata_fp16_sb_032, 
    conv_ctest_bwddata_fp16_sb_033, conv_ctest_bwddata_fp16_sb_034, 
    conv_ctest_bwddata_fp16_sb_035, conv_ctest_bwddata_fp16_sb_036, 
    conv_ctest_bwddata_fp16_sb_037, conv_ctest_bwddata_fp16_sb_038, 
    conv_ctest_bwddata_fp16_sb_039, 
};

gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_001 {{12544, 1, 64, 12544, 64, 12544, 802816, 0, 12544}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_002 {{3136, 1, 128, 3136, 128, 3136, 401408, 0, 3136}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_003 {{3136, 1, 256, 3136, 256, 3136, 802816, 0, 3136}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_004 {{3136, 1, 64, 3136, 64, 3136, 200704, 0, 3136}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_005 {{32768, 1, 480, 32768, 480, 32768, 15728640, 0, 32768}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_006 {{729, 1, 64, 729, 64, 729, 46656, 0, 729}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_007 {{784, 1, 128, 784, 128, 784, 100352, 0, 784}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_008 {{784, 1, 192, 784, 192, 784, 150528, 0, 784}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_009 {{784, 1, 256, 784, 256, 784, 200704, 0, 784}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_010 {{784, 1, 320, 784, 320, 784, 250880, 0, 784}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_011 {{784, 1, 512, 784, 512, 784, 401408, 0, 784}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_012 {{784, 1, 64, 784, 64, 784, 50176, 0, 784}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_013 {{8192, 1, 480, 8192, 480, 8192, 3932160, 0, 8192}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_014 {{8192, 1, 512, 8192, 512, 8192, 4194304, 0, 8192}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_015 {{8192, 1, 528, 8192, 528, 8192, 4325376, 0, 8192}, {1, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp32_sb_016 {{8192, 1, 832, 8192, 832, 8192, 6815744, 0, 8192}, {1, 0}, {'N', 'N'}, 1};

const vector<gemm_strided_batched_tuple> conv_ctest_fwd_fp32_sb = {
    conv_ctest_fwd_fp32_sb_001, conv_ctest_fwd_fp32_sb_002, 
    conv_ctest_fwd_fp32_sb_003, conv_ctest_fwd_fp32_sb_004, 
    conv_ctest_fwd_fp32_sb_005, conv_ctest_fwd_fp32_sb_006, 
    conv_ctest_fwd_fp32_sb_007, conv_ctest_fwd_fp32_sb_008, 
    conv_ctest_fwd_fp32_sb_009, conv_ctest_fwd_fp32_sb_010, 
    conv_ctest_fwd_fp32_sb_011, conv_ctest_fwd_fp32_sb_012, 
    conv_ctest_fwd_fp32_sb_013, conv_ctest_fwd_fp32_sb_014, 
    conv_ctest_fwd_fp32_sb_015, conv_ctest_fwd_fp32_sb_016, 
};

gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_001 {{12544, 1, 64, 12544, 64, 12544, 802816, 0, 12544}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_002 {{3136, 1, 128, 3136, 128, 3136, 401408, 0, 3136}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_003 {{3136, 1, 256, 3136, 256, 3136, 802816, 0, 3136}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_004 {{3136, 1, 64, 3136, 64, 3136, 200704, 0, 3136}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_005 {{32768, 1, 480, 32768, 480, 32768, 15728640, 0, 32768}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_006 {{729, 1, 64, 729, 64, 729, 46656, 0, 729}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_007 {{784, 1, 128, 784, 128, 784, 100352, 0, 784}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_008 {{784, 1, 192, 784, 192, 784, 150528, 0, 784}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_009 {{784, 1, 256, 784, 256, 784, 200704, 0, 784}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_010 {{784, 1, 320, 784, 320, 784, 250880, 0, 784}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_011 {{784, 1, 512, 784, 512, 784, 401408, 0, 784}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_012 {{784, 1, 64, 784, 64, 784, 50176, 0, 784}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_013 {{8192, 1, 480, 8192, 480, 8192, 3932160, 0, 8192}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_014 {{8192, 1, 512, 8192, 512, 8192, 4194304, 0, 8192}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_015 {{8192, 1, 528, 8192, 528, 8192, 4325376, 0, 8192}, {15360, 0}, {'N', 'N'}, 1};
gemm_strided_batched_tuple conv_ctest_fwd_fp16_sb_016 {{8192, 1, 832, 8192, 832, 8192, 6815744, 0, 8192}, {15360, 0}, {'N', 'N'}, 1};

const vector<gemm_strided_batched_tuple> conv_ctest_fwd_fp16_sb = {
    conv_ctest_fwd_fp16_sb_001, conv_ctest_fwd_fp16_sb_002,
    conv_ctest_fwd_fp16_sb_003, conv_ctest_fwd_fp16_sb_004,
    conv_ctest_fwd_fp16_sb_005, conv_ctest_fwd_fp16_sb_006,
    conv_ctest_fwd_fp16_sb_007, conv_ctest_fwd_fp16_sb_008,
    conv_ctest_fwd_fp16_sb_009, conv_ctest_fwd_fp16_sb_010,
    conv_ctest_fwd_fp16_sb_011, conv_ctest_fwd_fp16_sb_012,
    conv_ctest_fwd_fp16_sb_013, conv_ctest_fwd_fp16_sb_014,
    conv_ctest_fwd_fp16_sb_015, conv_ctest_fwd_fp16_sb_016,
};

// clang-format on

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
    arg.M        = matrix_size[0];
    arg.N        = matrix_size[1];
    arg.K        = matrix_size[2];
    arg.lda      = matrix_size[3];
    arg.ldb      = matrix_size[4];
    arg.ldc      = matrix_size[5];
    arg.stride_a = matrix_size[6];
    arg.stride_b = matrix_size[7];
    arg.stride_c = matrix_size[8];

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

class gemm_strided_batched_half : public ::TestWithParam<gemm_strided_batched_tuple>
{
    protected:
    gemm_strided_batched_half() {}
    virtual ~gemm_strided_batched_half() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class gemm_strided_batched_float : public ::TestWithParam<gemm_strided_batched_tuple>
{
    protected:
    gemm_strided_batched_float() {}
    virtual ~gemm_strided_batched_float() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class gemm_strided_batched_double : public ::TestWithParam<gemm_strided_batched_tuple>
{
    protected:
    gemm_strided_batched_double() {}
    virtual ~gemm_strided_batched_double() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(gemm_strided_batched_half, standard)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_strided_batched_arguments(GetParam());

    rocblas_status status = testing_gemm_strided_batched<rocblas_half>(arg);

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

TEST_P(gemm_strided_batched_float, standard)
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

TEST_P(gemm_strided_batched_double, standard)
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
    }
}

TEST_P(gemm_strided_batched, half)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_strided_batched_arguments(GetParam());

    rocblas_status status = testing_gemm_strided_batched<rocblas_half>(arg);

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

//--- small
// tests with stride_a == 0
INSTANTIATE_TEST_CASE_P(quick_blas3_small_stride_zero,
                        gemm_strided_batched,
                        Combine(ValuesIn(small_matrix_size_stride_a_range),
                                ValuesIn(alpha_beta_stride_a_range),
                                ValuesIn(transA_transB_stride_a_range),
                                ValuesIn(small_batch_count_stride_a_range)));

INSTANTIATE_TEST_CASE_P(quick_blas3_small,
                        gemm_strided_batched,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(small_batch_count_range)));
// tests with stride_a == 0
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_small_stride_zero,
                        gemm_strided_batched,
                        Combine(ValuesIn(small_matrix_size_stride_a_range),
                                ValuesIn(alpha_beta_stride_a_range),
                                ValuesIn(transA_transB_stride_a_range),
                                ValuesIn(medium_batch_count_stride_a_range)));
//--- medium
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_medium,
                        gemm_strided_batched,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(small_batch_count_range)));
// tests with stride_a == 0
INSTANTIATE_TEST_CASE_P(nightly_blas3_medium_stride_zero,
                        gemm_strided_batched,
                        Combine(ValuesIn(medium_matrix_size_stride_a_range),
                                ValuesIn(alpha_beta_stride_a_range),
                                ValuesIn(transA_transB_stride_a_range),
                                ValuesIn(medium_batch_count_stride_a_range)));

INSTANTIATE_TEST_CASE_P(nightly_checkin_blas3_medium,
                        gemm_strided_batched,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(medium_batch_count_range)));
//--- large
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_large,
                        gemm_strided_batched,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(small_batch_count_range)));
// tests with stride_a == 0
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_large_stride_zero,
                        gemm_strided_batched,
                        Combine(ValuesIn(large_matrix_size_stride_a_range),
                                ValuesIn(alpha_beta_stride_a_range),
                                ValuesIn(transA_transB_stride_a_range),
                                ValuesIn(small_batch_count_range)));

// clang-format off
INSTANTIATE_TEST_CASE_P(nightly_blas3_deepbench_sizes, gemm_strided_batched, ValuesIn(deepbench_sb_vec));
INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_fwd_fp32_sb, gemm_strided_batched_float, ValuesIn(conv_resnet50_fwd_fp32_sb));

INSTANTIATE_TEST_CASE_P(known_bug_conv_resnet50_fwd_fp32_sb, gemm_strided_batched_float, ValuesIn(known_bug_conv_resnet50_fwd_fp32_sb));
INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_fwd_fp16_sb, gemm_strided_batched_half, ValuesIn(conv_resnet50_fwd_fp16_sb));

INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_bwddata_fp32_sb, gemm_strided_batched_float, ValuesIn(conv_resnet50_bwddata_fp32_sb));
INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_bwddata_fp16_sb, gemm_strided_batched_half, ValuesIn(conv_resnet50_bwddata_fp16_sb));

INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_fwd_fp32_sb, gemm_strided_batched_float, ValuesIn(conv_inception4_fwd_fp32_sb));
INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_fwd_fp16_sb, gemm_strided_batched_half, ValuesIn(conv_inception4_fwd_fp16_sb));

INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_bwddata_fp32_sb, gemm_strided_batched_float, ValuesIn(conv_inception4_bwddata_fp32_sb));
INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_bwddata_fp16_sb, gemm_strided_batched_half, ValuesIn(conv_inception4_bwddata_fp16_sb));

INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_bwddata_fp32_sb, gemm_strided_batched_float, ValuesIn(conv_ctest_bwddata_fp32_sb));
INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_bwddata_fp16_sb, gemm_strided_batched_half, ValuesIn(conv_ctest_bwddata_fp16_sb));

INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_fwd_fp32_sb, gemm_strided_batched_float, ValuesIn(conv_ctest_fwd_fp32_sb));
INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_fwd_fp16_sb, gemm_strided_batched_half, ValuesIn(conv_ctest_fwd_fp16_sb));
// clang-format on
