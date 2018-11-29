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

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> deepbench_size_range = {
    {  192,    64,   784,   784,   784,   192,   192},
    {12544,   128,   256, 12544,   256, 12544, 12544},
    {12544,   256,    64, 12544,    64, 12544, 12544},
    {12544,    64,   147, 12544,   147, 12544, 12544},
    { 1568,  1024,   256,  1568,   256,  1568,  1568},
    { 1568,  1024,   512,  1568,   512,  1568,  1568},
    { 1568,   256,  1024,  1568,  1024,  1568,  1568},
    { 1568,   256,   256,  1568,   256,  1568,  1568},
    { 1568,   256,   512,  1568,   512,  1568,  1568},
    { 1568,   512,   128,  1568,   128,  1568,  1568},
    { 1653,   256,  3200,  1653,  3200,  1653,  1653},
    {  196,    48, 12800,   196, 12800,   196,   196},
    {23040,    16,     9, 23040,     9, 23040, 23040},
    {26939,    32,   100, 26939,   100, 26939, 26939},
    {27920,    64,    25, 27920,    25, 27920, 27920},
    { 2916,    64,    27,  2916,    27,  2916,  2916},
    { 3136,  1024,   256,  3136,   256,  3136,  3136},
    { 3136,  1024,   512,  3136,   512,  3136,  3136},
    { 3136,   192,   512,  3136,   512,  3136,  3136},
    { 3136,   256,  1024,  3136,  1024,  3136,  3136},
    { 3136,   256,   256,  3136,   256,  3136,  3136},
    { 3136,   256,   512,  3136,   512,  3136,  3136},
    { 3136,   512,   128,  3136,   128,  3136,  3136},
    {  369,   512,  6400,   369,  6400,   369,   369},
    {  392,  1024,   256,   392,   256,   392,   392},
    {  392,  2048,  1024,   392,  1024,   392,   392},
    {  392,  2048,   512,   392,   512,   392,   392},
    {  392,   512,  1024,   392,  1024,   392,   392},
    {  392,   512,  2048,   392,  2048,   392,   392},
    {  392,   512,   512,   392,   512,   392,   392},
    {   49,   128, 20800,    49, 20800,    49,    49},
    {   49,   512,  2048,    49,  2048,    49,    49},
    {50176,    64,    27, 50176,    27, 50176, 50176},
    { 5760,    32,   144,  5760,   144,  5760,  5760},
    { 6272,   128,   256,  6272,   256,  6272,  6272},
    { 6272,   256,    64,  6272,    64,  6272,  6272},
    { 6308,    32,  1600,  6308,  1600,  6308,  6308},
    { 6786,   128,  1600,  6786,  1600,  6786,  6786},
    {  784,  1024,   256,   784,   256,   784,   784},
    {  784,  2048,  1024,   784,  1024,   784,   784},
    {  784,  2048,   512,   784,   512,   784,   784},
    {  784,   256,   832,   784,   832,   784,   784},
    {  784,    32,  4800,   784,  4800,   784,   784},
    {  784,   512,  1024,   784,  1024,   784,   784},
    {  784,   512,  2048,   784,  2048,   784,   784},
    {  784,   512,   512,   784,   512,   784,   784},
    {12544,   147,    64, 12544,   147, 12544, 12544},
    {12544,   256,   128, 12544,   256, 12544, 12544},
    {12544,    64,   256, 12544,    64, 12544, 12544},
    { 1568,   128,   512,  1568,   128,  1568,  1568},
    { 1568,   512,  1024,  1568,   512,  1568,  1568},
    { 1568,   512,   256,  1568,   512,  1568,  1568},
    { 1653,  3200,   256,  1653,  3200,  1653,  1653},
    {  196, 12800,    48,   196, 12800,   196,   196},
    {23040,     9,    16, 23040,     9, 23040, 23040},
    {26939,   100,    32, 26939,   100, 26939, 26939},
    {27920,    25,    64, 27920,    25, 27920, 27920},
    { 2916,    27,    64,  2916,    27,  2916,  2916},
    { 3136,   128,   512,  3136,   128,  3136,  3136},
    { 3136,   512,  1024,  3136,   512,  3136,  3136},
    { 3136,   512,   256,  3136,   512,  3136,  3136},
    {  369,  6400,   512,   369,  6400,   369,   369},
    {  392,  1024,  2048,   392,  1024,   392,   392},
    {  392,  1024,   512,   392,  1024,   392,   392},
    {  392,   256,  1024,   392,   256,   392,   392},
    {   49,  2048,   512,    49,  2048,    49,    49},
    {   49, 20800,   128,    49, 20800,    49,    49},
    { 6272,   256,   128,  6272,   256,  6272,  6272},
    { 6272,    64,   256,  6272,    64,  6272,  6272},
    { 6308,  1600,    32,  6308,  1600,  6308,  6308},
    { 6786,  1600,   128,  6786,  1600,  6786,  6786},
    {  784,  1024,  2048,   784,  1024,   784,   784},
    {  784,  1024,   512,   784,  1024,   784,   784},
    {  784,   256,  1024,   784,   256,   784,   784},
    {  784,  4800,    32,   784,  4800,   784,   784},
    {  100,    32, 26939, 26939, 26939,   100,   100},
    { 1024,  2048,    49,    49,    49,  1024,  1024},
    { 1024,   256,   196,   196,   196,  1024,  1024},
    { 1024,   512,    49,    49,    49,  1024,  1024},
    { 1152,   128,  7000,  7000,  7000,  1152,  1152},
    { 1152,   128,   729,   729,   729,  1152,  1152},
    { 1152,   128,   784,   784,   784,  1152,  1152},
    { 1152,   256,   196,   196,   196,  1152,  1152},
    { 1152,   256,  3136,  3136,  3136, 1152,   1152},
    {12800,    48,   196,   196,   196, 12800, 12800},
    {  128,   512,   196,   196,   196,   128,   128},
    {  128,   512,   784,   784,   784,   128,   128},
    {  144,    32,  5760,  5760,  5760,   144,   144},
    {  147,    64, 12544, 12544, 12544,   147,   147},
    { 1600,   128,  6786,  6786,  6786,  1600,  1600},
    { 1600,    32,  6308,  6308,  6308,  1600,  1600},
    {  192,    64,   784,   784,   784,   192,   192},
    { 2048,   512,    49,    49,    49,  2048,  2048},
    {20800,   128,    49,    49,    49, 20800, 20800},
    { 2304,   256,  1680,  1680,  1680,  2304,  2304},
    { 2304,   256,   196,   196,   196,  2304,  2304},
    { 2304,   512,    49,    49,    49,  2304,  2304},
    { 2304,   512,   784,   784,   784,  2304,  2304},
    {  256,  1024,   196,   196,   196,   256,   256},
    {  256,  1024,    49,    49,    49,   256,   256},
    {  256,   128,   784,   784,   784,   256,   256},
    {  256,   256,   196,   196,   196,   256,   256},
    {  256,    64,  3136,  3136,  3136,   256,   256},
    {   25,    64, 27920, 27920, 27920,    25,    25},
    {   27,    64,  2916,  2916,  2916,    27,    27},
    {  288,    64,  1440,  1440,  1440,   288,   288},
    { 3200,   256,  1653,  1653,  1653,  3200,  3200},
    { 4608,   512,   196,   196,   196,  4608,  4608},
    { 4608,   512,   420,   420,   420,  4608,  4608},
    { 4608,   512,    49,    49,    49,  4608,  4608},
    { 4800,    32,   784,   784,   784,  4800,  4800},
    {  512,  1024,   196,   196,   196,   512,   512},
    {  512,   128,   784,   784,   784,   512,   512},
    {  512,   192,   196,   196,   196,   512,   512},
    {  512,  2048,    49,    49,    49,   512,   512},
    {  512,   256,   196,   196,   196,   512,   512},
    {  512,   512,    49,    49,    49,   512,   512},
    {  576,   128, 12544, 12544, 12544,   576,   576},
    {  576,   128,   360,   360,   360,   576,   576},
    {  576,    64, 28000, 28000, 28000,   576,   576},
    {  576,    64,  2916,  2916,  2916,   576,   576},
    {  576,    64,  3136,  3136,  3136,   576,   576},
    { 6400,   512,   369,   369,   369,  6400,  6400},
    {   64,   256,  3136,  3136,  3136,    64,    64},
    {   64,   256,   784,   784,   784,    64,    64},
    {   64,    64, 12544, 12544, 12544,    64,    64},
    {  832,   256,    49,    49,    49,   832,   832},
    {    9,    16, 23040, 23040, 23040,     9,     9}
};

// this test is too large for all orientations, only running the 
// one defined in the original suite
const vector<vector<int>> deepbench_large_size_range = {
    {   27,    64, 50176, 50176, 50176,    27,    27},
};

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> resnet50_fwd_size_range = {
    {12544,  1024,   256, 12544,   256, 12544, 12544},
    {12544,  1024,   512, 12544,   512, 12544, 12544},
    {12544,   256,  1024, 12544,  1024, 12544, 12544},
    {12544,   256,   512, 12544,   512, 12544, 12544},
    {12544,    64,   147, 12544,   147, 12544, 12544},
    {  196,   256,  2304,   196,  2304,   196,   196},
    { 3025,    64,   576,  3025,   576,  3025,  3025},
    { 3136,  2048,  1024,  3136,  1024,  3136,  3136},
    { 3136,  2048,   512,  3136,   512,  3136,  3136},
    { 3136,   512,  1024,  3136,  1024,  3136,  3136},
    { 3136,   512,  2048,  3136,  2048,  3136,  3136},
    { 3136,    64,   576,  3136,   576,  3136,  3136},
    {   49,   512,  4608,    49,  4608,    49,    49},
    {50176,   128,   256, 50176,   256, 50176, 50176},
    {50176,   512,   256, 50176,   256, 50176, 50176},
    {  784,   128,  1152,   784,  1152,   784,   784}
};

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> resnet50_bwdwrw_size_range = {
    { 1024,  2048,    49,    49,    49,  1024,  1024},
    { 1024,   256,   196,   196,   196,  1024,  1024},
    { 1024,   512,    49,    49,    49,  1024,  1024},
    { 1152,   128,   784,   784,   784,  1152,  1152},
    {  128,   512,   784,   784,   784,   128,   128},
    {  147,    64, 12544, 12544, 12544,   147,   147},
    { 2048,   512,    49,    49,    49,  2048,  2048},
    { 2304,   256,   196,   196,   196,  2304,  2304},
    {  256,  1024,   196,   196,   196,   256,   256},
    {  256,   128,   784,   784,   784,   256,   256},
    {  256,   512,   784,   784,   784,   256,   256},
    {  256,    64,  3025,  3025,  3025,   256,   256},
    {  256,    64,  3136,  3136,  3136,   256,   256},
    { 4608,   512,    49,    49,    49,  4608,  4608},
    {  512,  1024,   196,   196,   196,   512,   512},
    {  512,   128,   784,   784,   784,   512,   512},
    {  512,  2048,    49,    49,    49,   512,   512},
    {  512,   256,   196,   196,   196,   512,   512},
    {  576,    64,  3025,  3025,  3025,   576,   576},
    {  576,    64,  3136,  3136,  3136,   576,   576},
    {   64,   256,  3025,  3025,  3025,    64,    64},
    {   64,   256,  3136,  3136,  3136,    64,    64},
    {   64,    64,  3025,  3025,  3025,    64,    64},
    {   64,    64,  3136,  3136,  3136,    64,    64}
};

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc, ldd};
const vector<vector<int>> resnet50_bwddata_size_range = {
    {12544,   147,    64, 12544,   147, 12544, 12544},
    {12544,   512,  1024, 12544,   512, 12544, 12544},
    {12544,   512,   256, 12544,   512, 12544, 12544},
    {  196,  2304,   256,   196,  2304,   196,   196},
    { 3025,   576,    64,  3025,   576,  3025,  3025},
    { 3136,  1024,  2048,  3136,  1024,  3136,  3136},
    { 3136,  1024,   512,  3136,  1024,  3136,  3136},
    { 3136,   576,    64,  3136,   576,  3136,  3136},
    {   49,  4608,   512,    49,  4608,    49,    49},
    {50176,   256,   128, 50176,   256, 50176, 50176},
    {50176,   256,   512, 50176,   256, 50176, 50176},
    {  784,  1152,   128,   784,  1152,   784,   784}
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

const vector<vector<double>> deepbench_alpha_beta_range = {
    {1.0, 0.0}, {1.0, 1.0}
};

const vector<vector<double>> resnet50_fwd_alpha_beta_range = {
    {1.0, 0.0}
};

const vector<vector<double>> resnet50_bwdwrw_alpha_beta_range = {
    {1.0, 1.0}
};

const vector<vector<double>> resnet50_bwddata_alpha_beta_range = {
    {1.0, 0.0}
};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgemm/dgemm,
const vector<vector<char>> small_transA_transB_range = {{'N', 'N'}};
const vector<vector<char>> transA_transB_range = {{'N', 'N'}, {'N', 'T'}, {'C', 'N'}, {'T', 'C'}};
const vector<vector<char>> transA_transB_range_int8 =  {{'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'T', 'T'}};
const vector<vector<char>> deepbench_transA_transB_range =  {{'N', 'N'}, {'N', 'T'}, {'T', 'N'}};
const vector<vector<char>> deepbench_large_transA_transB_range =  {{'T', 'N'}};
const vector<vector<char>> resnet50_fwd_transA_transB_range =  {{'N', 'N'}};
const vector<vector<char>> resnet50_bwdwrw_transA_transB_range =  {{'T', 'N'}};
const vector<vector<char>> resnet50_bwddata_transA_transB_range =  {{'N', 'T'}};

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
        else if((arg.a_type == rocblas_datatype_i8_r) &&
                ((arg.K % 4 != 0) || (arg.transA_option == 'T' && arg.lda % 4 != 0) ||
                 (arg.transB_option == 'N' && arg.ldb % 4 != 0)))
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else
        {
            FAIL() << "Unexpected failure: " << status;
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
                                ValuesIn(transA_transB_range_int8),
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

INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_medium_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range_int8),
                                ValuesIn(precision_int8)));
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

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_large_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range_int8),
                                ValuesIn(precision_int8)));
//----chunk
INSTANTIATE_TEST_CASE_P(pre_checkin_blas_ex_chunk,
                        parameterized_chunk_gemm_ex,
                        Combine(ValuesIn(chunk_matrix_size_range),
                                ValuesIn(alpha_beta_2_3_range),
                                ValuesIn(transA_transB_range),
                                ValuesIn(precision_single)));

//----deepbench
INSTANTIATE_TEST_CASE_P(nightly_blas_ex_deepbench_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(deepbench_size_range),
                                ValuesIn(deepbench_alpha_beta_range),
                                ValuesIn(deepbench_transA_transB_range),
                                ValuesIn(precision_int8)));

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_deepbench_large_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(deepbench_large_size_range),
                                ValuesIn(deepbench_alpha_beta_range),
                                ValuesIn(deepbench_large_transA_transB_range),
                                ValuesIn(precision_int8)));

//----resnet50
INSTANTIATE_TEST_CASE_P(nightly_blas_ex_resnet50_fwd_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(resnet50_fwd_size_range),
                                ValuesIn(resnet50_fwd_alpha_beta_range),
                                ValuesIn(resnet50_fwd_transA_transB_range),
                                ValuesIn(precision_int8)));

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_resnet50_bwdwrw_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(resnet50_bwdwrw_size_range),
                                ValuesIn(resnet50_bwdwrw_alpha_beta_range),
                                ValuesIn(resnet50_bwdwrw_transA_transB_range),
                                ValuesIn(precision_int8)));

INSTANTIATE_TEST_CASE_P(nightly_blas_ex_resnet50_bwddata_int8,
                        parameterized_gemm_ex,
                        Combine(ValuesIn(resnet50_bwddata_size_range),
                                ValuesIn(resnet50_bwddata_alpha_beta_range),
                                ValuesIn(resnet50_bwddata_transA_transB_range),
                                ValuesIn(precision_int8)));
