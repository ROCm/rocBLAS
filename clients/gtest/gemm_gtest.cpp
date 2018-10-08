/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>
#include "testing_gemm.hpp"
#include "testing_gemm_sweep.hpp"
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

typedef std::tuple<int, int, int, vector<double>, vector<char>> gemm_sweep_tuple;
typedef std::tuple<vector<int>, vector<double>, vector<char>> gemm_tuple;

// clang-format off
const vector<int> size_range_1_4   = {   1,    2,    3,   4};
const vector<int> size_range_1_8   = {   1,    2,    3,   4,  5,  6,  7,  8};
const vector<int> size_range_5_8   = {   5,    6,    7,   8};
const vector<int> size_range_9_12  = {   9,   10,   11,  12};
const vector<int> size_range_13_16 = {  13,   14,   15,  16};
const vector<int> size_range_17_20 = {  17,   18,   19,  20};
const vector<int> size_range_20_23 = {  20,   21,   22,  23};
const vector<int> size_range_24_27 = {  24,   25,   26,  27};
const vector<int> size_range_28_31 = {  28,   29,   30,  31};
const vector<int> size_range_32    = {  31,   32,   33};
const vector<int> size_range_48    = {  47,   48,   49};
const vector<int> size_range_64    = {  63,   64,   65};
const vector<int> size_range_96    = {  95,   96,   97};
const vector<int> size_range_128   = { 127,  128,  129};
const vector<int> size_range_256   = { 255,  256,  257};
const vector<int> size_range_512   = { 511,  512,  513};
const vector<int> size_range_1024  = {1023, 1024, 1025};
const vector<int> size_range_9_129 = {9, 10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                                         20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                                         30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                                         40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
                                         50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                                         60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                                         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                                         80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                                         90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                                        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                        120, 121, 122, 123, 124, 125, 126, 127, 128, 129};
// clang-format on

// vector of vector, each vector is a {M, N, K, lda, ldb, ldc};
// add/delete as a group
const vector<vector<int>> small_matrix_size_range = {
    {1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6}, {7, 9, 15, 17, 18, 19},
};

const vector<vector<int>> medium_matrix_size_range = {
    {-1, -1, -1, -1, 1, 1},
    {1, 1, 1, 1, 1, 1},
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
    {191, 193, 194, 195, 196, 197},
    {500, 501, 502, 503, 604, 505},
    {500, 501, 502, 203, 204, 205},
    {639, 640, 347, 960, 961, 1062},
};

const vector<vector<int>> large_matrix_size_range = {
    {1000, 1001, 101, 2002, 1003, 1004},
    {925, 1026, 1027, 1028, 2029, 1031},
    {4011, 4012, 103, 4014, 4015, 4016},
};

const vector<vector<int>> chunk_matrix_size_range = {
    {24000, 256, 256, 24010, 256, 24000},
    {24000, 256, 256, 24000, 256, 24020},
    {256, 24001, 256, 256, 24030, 24000},
    {256, 24001, 256, 256, 24000, 24040},
};

const vector<vector<int>> NaN_matrix_size_range = {
    {5, 6, 7, 8, 9, 10}, {4011, 4012, 111, 4013, 4014, 4015},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> alpha_beta_2_3_range = {
    {2.0, 3.0},
};

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
const vector<vector<char>> transA_transB_N_N_range = {{'N', 'N'}};

// clang-format off

gemm_tuple deepbench0{{192, 64, 784, 784, 784, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench1{{12544, 128, 256, 12544, 256, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench2{{12544, 256, 64, 12544, 64, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench3{{12544, 64, 147, 12544, 147, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench4{{1568, 1024, 256, 1568, 256, 1568}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench5{{1568, 1024, 512, 1568, 512, 1568}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench6{{1568, 256, 1024, 1568, 1024, 1568}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench7{{1568, 256, 256, 1568, 256, 1568}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench8{{1568, 256, 512, 1568, 512, 1568}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench9{{1568, 512, 128, 1568, 128, 1568}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench10{{1653, 256, 3200, 1653, 3200, 1653}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench11{{196, 48, 12800, 196, 12800, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench12{{23040, 16, 9, 23040, 9, 23040}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench13{{26939, 32, 100, 26939, 100, 26939}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench14{{27920, 64, 25, 27920, 25, 27920}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench15{{2916, 64, 27, 2916, 27, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench16{{3136, 1024, 256, 3136, 256, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench17{{3136, 1024, 512, 3136, 512, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench18{{3136, 192, 512, 3136, 512, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench19{{3136, 256, 1024, 3136, 1024, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench20{{3136, 256, 256, 3136, 256, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench21{{3136, 256, 512, 3136, 512, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench22{{3136, 512, 128, 3136, 128, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench23{{369, 512, 6400, 369, 6400, 369}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench24{{392, 1024, 256, 392, 256, 392}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench25{{392, 2048, 1024, 392, 1024, 392}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench26{{392, 2048, 512, 392, 512, 392}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench27{{392, 512, 1024, 392, 1024, 392}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench28{{392, 512, 2048, 392, 2048, 392}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench29{{392, 512, 512, 392, 512, 392}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench30{{49, 128, 20800, 49, 20800, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench31{{49, 512, 2048, 49, 2048, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench32{{50176, 64, 27, 50176, 27, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench33{{5760, 32, 144, 5760, 144, 5760}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench34{{6272, 128, 256, 6272, 256, 6272}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench35{{6272, 256, 64, 6272, 64, 6272}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench36{{6308, 32, 1600, 6308, 1600, 6308}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench37{{6786, 128, 1600, 6786, 1600, 6786}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench38{{784, 1024, 256, 784, 256, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench39{{784, 2048, 1024, 784, 1024, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench40{{784, 2048, 512, 784, 512, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench41{{784, 256, 832, 784, 832, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench42{{784, 32, 4800, 784, 4800, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench43{{784, 512, 1024, 784, 1024, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench44{{784, 512, 2048, 784, 2048, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench45{{784, 512, 512, 784, 512, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple deepbench46{{12544, 147, 64, 12544, 147, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench47{{12544, 256, 128, 12544, 256, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench48{{12544, 64, 256, 12544, 64, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench49{{1568, 128, 512, 1568, 128, 1568}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench50{{1568, 512, 1024, 1568, 512, 1568}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench51{{1568, 512, 256, 1568, 512, 1568}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench52{{1653, 3200, 256, 1653, 3200, 1653}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench53{{196, 12800, 48, 196, 12800, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench54{{23040, 9, 16, 23040, 9, 23040}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench55{{26939, 100, 32, 26939, 100, 26939}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench56{{27920, 25, 64, 27920, 25, 27920}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench57{{2916, 27, 64, 2916, 27, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench58{{3136, 128, 512, 3136, 128, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench59{{3136, 512, 1024, 3136, 512, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench60{{3136, 512, 256, 3136, 512, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench61{{369, 6400, 512, 369, 6400, 369}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench62{{392, 1024, 2048, 392, 1024, 392}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench63{{392, 1024, 512, 392, 1024, 392}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench64{{392, 256, 1024, 392, 256, 392}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench65{{49, 2048, 512, 49, 2048, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench66{{49, 20800, 128, 49, 20800, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench67{{6272, 256, 128, 6272, 256, 6272}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench68{{6272, 64, 256, 6272, 64, 6272}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench69{{6308, 1600, 32, 6308, 1600, 6308}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench70{{6786, 1600, 128, 6786, 1600, 6786}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench71{{784, 1024, 2048, 784, 1024, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench72{{784, 1024, 512, 784, 1024, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench73{{784, 256, 1024, 784, 256, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench74{{784, 4800, 32, 784, 4800, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple deepbench75{{100, 32, 26939, 26939, 26939, 100}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench76{{1024, 2048, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench77{{1024, 256, 196, 196, 196, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench78{{1024, 512, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench79{{1152, 128, 7000, 7000, 7000, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench80{{1152, 128, 729, 729, 729, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench81{{1152, 128, 784, 784, 784, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench82{{1152, 256, 196, 196, 196, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench83{{1152, 256, 3136, 3136, 3136, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench84{{12800, 48, 196, 196, 196, 12800}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench85{{128, 512, 196, 196, 196, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench86{{128, 512, 784, 784, 784, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench87{{144, 32, 5760, 5760, 5760, 144}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench88{{147, 64, 12544, 12544, 12544, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench89{{1600, 128, 6786, 6786, 6786, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench90{{1600, 32, 6308, 6308, 6308, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench91{{192, 64, 784, 784, 784, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench92{{2048, 512, 49, 49, 49, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench93{{20800, 128, 49, 49, 49, 20800}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench94{{2304, 256, 1680, 1680, 1680, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench95{{2304, 256, 196, 196, 196, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench96{{2304, 512, 49, 49, 49, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench97{{2304, 512, 784, 784, 784, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench98{{256, 1024, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench99{{256, 1024, 49, 49, 49, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench100{{256, 128, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench101{{256, 256, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench102{{256, 64, 3136, 3136, 3136, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench103{{25, 64, 27920, 27920, 27920, 25}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench104{{27, 64, 2916, 2916, 2916, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench105{{27, 64, 50176, 50176, 50176, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench106{{288, 64, 1440, 1440, 1440, 288}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench107{{3200, 256, 1653, 1653, 1653, 3200}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench108{{4608, 512, 196, 196, 196, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench109{{4608, 512, 420, 420, 420, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench110{{4608, 512, 49, 49, 49, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench111{{4800, 32, 784, 784, 784, 4800}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench112{{512, 1024, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench113{{512, 128, 784, 784, 784, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench114{{512, 192, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench115{{512, 2048, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench116{{512, 256, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench117{{512, 512, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench118{{576, 128, 12544, 12544, 12544, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench119{{576, 128, 360, 360, 360, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench120{{576, 64, 28000, 28000, 28000, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench121{{576, 64, 2916, 2916, 2916, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench122{{576, 64, 3136, 3136, 3136, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench123{{6400, 512, 369, 369, 369, 6400}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench124{{64, 256, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench125{{64, 256, 784, 784, 784, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench126{{64, 64, 12544, 12544, 12544, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench127{{832, 256, 49, 49, 49, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple deepbench128{{9, 16, 23040, 23040, 23040, 9}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> deepbench_vec = {
    deepbench0,   deepbench1,   deepbench2,   deepbench3,   deepbench4,   deepbench5,
    deepbench6,   deepbench7,   deepbench8,   deepbench9,   deepbench10,  deepbench11,
    deepbench12,  deepbench13,  deepbench14,  deepbench15,  deepbench16,  deepbench17,
    deepbench18,  deepbench19,  deepbench20,  deepbench21,  deepbench22,  deepbench23,
    deepbench24,  deepbench25,  deepbench26,  deepbench27,  deepbench28,  deepbench29,
    deepbench30,  deepbench31,  deepbench32,  deepbench33,  deepbench34,  deepbench35,
    deepbench36,  deepbench37,  deepbench38,  deepbench39,  deepbench40,  deepbench41,
    deepbench42,  deepbench43,  deepbench44,  deepbench45,  deepbench46,  deepbench47,
    deepbench48,  deepbench49,  deepbench50,  deepbench51,  deepbench52,  deepbench53,
    deepbench54,  deepbench55,  deepbench56,  deepbench57,  deepbench58,  deepbench59,
    deepbench60,  deepbench61,  deepbench62,  deepbench63,  deepbench64,  deepbench65,
    deepbench66,  deepbench67,  deepbench68,  deepbench69,  deepbench70,  deepbench71,
    deepbench72,  deepbench73,  deepbench74,  deepbench75,  deepbench76,  deepbench77,
    deepbench78,  deepbench79,  deepbench80,  deepbench81,  deepbench82,  deepbench83,
    deepbench84,  deepbench85,  deepbench86,  deepbench87,  deepbench88,  deepbench89,
    deepbench90,  deepbench91,  deepbench92,  deepbench93,  deepbench94,  deepbench95,
    deepbench96,  deepbench97,  deepbench98,  deepbench99,  deepbench100, deepbench101,
    deepbench102, deepbench103, deepbench104, deepbench105, deepbench106, deepbench107,
    deepbench108, deepbench109, deepbench110, deepbench111, deepbench112, deepbench113,
    deepbench114, deepbench115, deepbench116, deepbench117, deepbench118, deepbench119,
    deepbench120, deepbench121, deepbench122, deepbench123, deepbench124, deepbench125,
    deepbench126, deepbench127, deepbench128,
};

gemm_tuple fixed_bug0{{9, 1, 9, 9, 9, 9}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> fixed_bug_vec = {
    fixed_bug0,
};

gemm_tuple conv_resnet50_fwd_fp32_001 {{12544, 1024, 256, 12544, 256, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_002 {{12544, 1024, 512, 12544, 512, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_003 {{12544, 256, 1024, 12544, 1024, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_004 {{12544, 256, 512, 12544, 512, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_005 {{12544, 64, 147, 12544, 147, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_006 {{196, 256, 2304, 196, 2304, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_007 {{3025, 64, 576, 3025, 576, 3025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_008 {{3136, 2048, 1024, 3136, 1024, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_009 {{3136, 2048, 512, 3136, 512, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_010 {{3136, 512, 1024, 3136, 1024, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_011 {{3136, 512, 2048, 3136, 2048, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_012 {{3136, 64, 576, 3136, 576, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_013 {{49, 512, 4608, 49, 4608, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_014 {{50176, 128, 256, 50176, 256, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_015 {{50176, 512, 256, 50176, 256, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp32_016 {{784, 128, 1152, 784, 1152, 784}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_resnet50_fwd_fp32 = {
    conv_resnet50_fwd_fp32_001, conv_resnet50_fwd_fp32_002, 
    conv_resnet50_fwd_fp32_003, conv_resnet50_fwd_fp32_004,
    conv_resnet50_fwd_fp32_005, conv_resnet50_fwd_fp32_006, 
    conv_resnet50_fwd_fp32_007, conv_resnet50_fwd_fp32_008,
    conv_resnet50_fwd_fp32_009, conv_resnet50_fwd_fp32_010, 
    conv_resnet50_fwd_fp32_011, conv_resnet50_fwd_fp32_012,
    conv_resnet50_fwd_fp32_013, conv_resnet50_fwd_fp32_014, 
    conv_resnet50_fwd_fp32_015, conv_resnet50_fwd_fp32_016,
};

gemm_tuple conv_resnet50_fwd_fp16_001 {{12544, 1024, 256, 12544, 256, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_002 {{12544, 1024, 512, 12544, 512, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_003 {{12544, 256, 1024, 12544, 1024, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_004 {{12544, 256, 512, 12544, 512, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_005 {{12544, 64, 147, 12544, 147, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_006 {{196, 256, 2304, 196, 2304, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_007 {{3025, 64, 576, 3025, 576, 3025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_008 {{3136, 2048, 1024, 3136, 1024, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_009 {{3136, 2048, 512, 3136, 512, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_010 {{3136, 512, 1024, 3136, 1024, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_011 {{3136, 512, 2048, 3136, 2048, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_012 {{3136, 64, 576, 3136, 576, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_013 {{49, 512, 4608, 49, 4608, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_014 {{50176, 128, 256, 50176, 256, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_015 {{50176, 512, 256, 50176, 256, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_016 {{784, 128, 1152, 784, 1152, 784}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_resnet50_fwd_fp16 = {
    conv_resnet50_fwd_fp16_001, conv_resnet50_fwd_fp16_002, 
    conv_resnet50_fwd_fp16_003, conv_resnet50_fwd_fp16_004, 
    conv_resnet50_fwd_fp16_005, conv_resnet50_fwd_fp16_006, 
    conv_resnet50_fwd_fp16_007, conv_resnet50_fwd_fp16_008, 
    conv_resnet50_fwd_fp16_009, conv_resnet50_fwd_fp16_010, 
    conv_resnet50_fwd_fp16_011, conv_resnet50_fwd_fp16_012, 
    conv_resnet50_fwd_fp16_013, conv_resnet50_fwd_fp16_014, 
    conv_resnet50_fwd_fp16_015, conv_resnet50_fwd_fp16_016, 
};

gemm_tuple conv_resnet50_bwdwrw_fp32_001 {{1024, 2048, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_002 {{1024, 256, 196, 196, 196, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_003 {{1024, 512, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_004 {{1152, 128, 784, 784, 784, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_005 {{128, 512, 784, 784, 784, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_006 {{147, 64, 12544, 12544, 12544, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_007 {{2048, 512, 49, 49, 49, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_008 {{2304, 256, 196, 196, 196, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_009 {{256, 1024, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_010 {{256, 128, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_011 {{256, 512, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_012 {{256, 64, 3025, 3025, 3025, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_013 {{256, 64, 3136, 3136, 3136, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_014 {{4608, 512, 49, 49, 49, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_015 {{512, 1024, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_016 {{512, 128, 784, 784, 784, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_017 {{512, 2048, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_018 {{512, 256, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_019 {{576, 64, 3025, 3025, 3025, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_020 {{576, 64, 3136, 3136, 3136, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_021 {{64, 256, 3025, 3025, 3025, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_022 {{64, 256, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_023 {{64, 64, 3025, 3025, 3025, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp32_024 {{64, 64, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> conv_resnet50_bwdwrw_fp32 = {
    conv_resnet50_bwdwrw_fp32_001, conv_resnet50_bwdwrw_fp32_002,
    conv_resnet50_bwdwrw_fp32_003, conv_resnet50_bwdwrw_fp32_004,
    conv_resnet50_bwdwrw_fp32_005, conv_resnet50_bwdwrw_fp32_006,
    conv_resnet50_bwdwrw_fp32_007, conv_resnet50_bwdwrw_fp32_008,
    conv_resnet50_bwdwrw_fp32_009, conv_resnet50_bwdwrw_fp32_010,
    conv_resnet50_bwdwrw_fp32_011, conv_resnet50_bwdwrw_fp32_012,
    conv_resnet50_bwdwrw_fp32_013, conv_resnet50_bwdwrw_fp32_014,
    conv_resnet50_bwdwrw_fp32_015, conv_resnet50_bwdwrw_fp32_016,
    conv_resnet50_bwdwrw_fp32_017, conv_resnet50_bwdwrw_fp32_018,
    conv_resnet50_bwdwrw_fp32_019, conv_resnet50_bwdwrw_fp32_020,
    conv_resnet50_bwdwrw_fp32_021, conv_resnet50_bwdwrw_fp32_022,
    conv_resnet50_bwdwrw_fp32_023, conv_resnet50_bwdwrw_fp32_024,
};

gemm_tuple conv_resnet50_bwdwrw_fp16_001 {{1024, 2048, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_002 {{1024, 256, 196, 196, 196, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_003 {{1024, 512, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_004 {{1152, 128, 784, 784, 784, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_005 {{128, 512, 784, 784, 784, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_006 {{147, 64, 12544, 12544, 12544, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_007 {{2048, 512, 49, 49, 49, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_008 {{2304, 256, 196, 196, 196, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_009 {{256, 1024, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_010 {{256, 128, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_011 {{256, 512, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_012 {{256, 64, 3025, 3025, 3025, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_013 {{256, 64, 3136, 3136, 3136, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_014 {{4608, 512, 49, 49, 49, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_015 {{512, 1024, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_016 {{512, 128, 784, 784, 784, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_017 {{512, 2048, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_018 {{512, 256, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_019 {{576, 64, 3025, 3025, 3025, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_020 {{576, 64, 3136, 3136, 3136, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_021 {{64, 256, 3025, 3025, 3025, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_022 {{64, 256, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_023 {{64, 64, 3025, 3025, 3025, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_024 {{64, 64, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_025 {{1024, 2048, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_026 {{1024, 256, 196, 196, 196, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_027 {{1024, 512, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_028 {{1152, 128, 784, 784, 784, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_029 {{128, 512, 784, 784, 784, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_030 {{147, 64, 12544, 12544, 12544, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_031 {{2048, 512, 49, 49, 49, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_032 {{2304, 256, 196, 196, 196, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_033 {{256, 1024, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_034 {{256, 128, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_035 {{256, 512, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_036 {{256, 64, 3025, 3025, 3025, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_037 {{256, 64, 3136, 3136, 3136, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_038 {{4608, 512, 49, 49, 49, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_039 {{512, 1024, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_040 {{512, 128, 784, 784, 784, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_041 {{512, 2048, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_042 {{512, 256, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_043 {{576, 64, 3025, 3025, 3025, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_044 {{576, 64, 3136, 3136, 3136, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_045 {{64, 256, 3025, 3025, 3025, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_046 {{64, 256, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_047 {{64, 64, 3025, 3025, 3025, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_048 {{64, 64, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> conv_resnet50_bwdwrw_fp16 = {
    conv_resnet50_bwdwrw_fp16_001, conv_resnet50_bwdwrw_fp16_002, 
    conv_resnet50_bwdwrw_fp16_003, conv_resnet50_bwdwrw_fp16_004, 
    conv_resnet50_bwdwrw_fp16_005, conv_resnet50_bwdwrw_fp16_006, 
    conv_resnet50_bwdwrw_fp16_007, conv_resnet50_bwdwrw_fp16_008, 
    conv_resnet50_bwdwrw_fp16_009, conv_resnet50_bwdwrw_fp16_010, 
    conv_resnet50_bwdwrw_fp16_011, conv_resnet50_bwdwrw_fp16_012, 
    conv_resnet50_bwdwrw_fp16_013, conv_resnet50_bwdwrw_fp16_014, 
    conv_resnet50_bwdwrw_fp16_015, conv_resnet50_bwdwrw_fp16_016, 
    conv_resnet50_bwdwrw_fp16_017, conv_resnet50_bwdwrw_fp16_018, 
    conv_resnet50_bwdwrw_fp16_019, conv_resnet50_bwdwrw_fp16_020, 
    conv_resnet50_bwdwrw_fp16_021, conv_resnet50_bwdwrw_fp16_022, 
    conv_resnet50_bwdwrw_fp16_023, conv_resnet50_bwdwrw_fp16_024, 
    conv_resnet50_bwdwrw_fp16_025, conv_resnet50_bwdwrw_fp16_026, 
    conv_resnet50_bwdwrw_fp16_027, conv_resnet50_bwdwrw_fp16_028, 
    conv_resnet50_bwdwrw_fp16_029, conv_resnet50_bwdwrw_fp16_030, 
    conv_resnet50_bwdwrw_fp16_031, conv_resnet50_bwdwrw_fp16_032, 
    conv_resnet50_bwdwrw_fp16_033, conv_resnet50_bwdwrw_fp16_034, 
    conv_resnet50_bwdwrw_fp16_035, conv_resnet50_bwdwrw_fp16_036, 
    conv_resnet50_bwdwrw_fp16_037, conv_resnet50_bwdwrw_fp16_038, 
    conv_resnet50_bwdwrw_fp16_039, conv_resnet50_bwdwrw_fp16_040, 
    conv_resnet50_bwdwrw_fp16_041, conv_resnet50_bwdwrw_fp16_042, 
    conv_resnet50_bwdwrw_fp16_043, conv_resnet50_bwdwrw_fp16_044, 
    conv_resnet50_bwdwrw_fp16_045, conv_resnet50_bwdwrw_fp16_046, 
    conv_resnet50_bwdwrw_fp16_047, conv_resnet50_bwdwrw_fp16_048, 
};

gemm_tuple conv_resnet50_bwddata_fp32_001 {{12544, 147, 64, 12544, 147, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_002 {{12544, 512, 1024, 12544, 512, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_003 {{12544, 512, 256, 12544, 512, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_004 {{196, 2304, 256, 196, 2304, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_005 {{3025, 576, 64, 3025, 576, 3025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_006 {{3136, 1024, 2048, 3136, 1024, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_007 {{3136, 1024, 512, 3136, 1024, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_008 {{3136, 576, 64, 3136, 576, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_009 {{49, 4608, 512, 49, 4608, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_010 {{50176, 256, 128, 50176, 256, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_011 {{50176, 256, 512, 50176, 256, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp32_012 {{784, 1152, 128, 784, 1152, 784}, {1, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_resnet50_bwddata_fp32 = {
    conv_resnet50_bwddata_fp32_001, conv_resnet50_bwddata_fp32_002, 
    conv_resnet50_bwddata_fp32_003, conv_resnet50_bwddata_fp32_004, 
    conv_resnet50_bwddata_fp32_005, conv_resnet50_bwddata_fp32_006, 
    conv_resnet50_bwddata_fp32_007, conv_resnet50_bwddata_fp32_008, 
    conv_resnet50_bwddata_fp32_009, conv_resnet50_bwddata_fp32_010, 
    conv_resnet50_bwddata_fp32_011, conv_resnet50_bwddata_fp32_012, 
};

gemm_tuple conv_resnet50_bwddata_fp16_001 {{12544, 147, 64, 12544, 147, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_002 {{12544, 512, 1024, 12544, 512, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_003 {{12544, 512, 256, 12544, 512, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_004 {{196, 2304, 256, 196, 2304, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_005 {{3025, 576, 64, 3025, 576, 3025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_006 {{3136, 1024, 2048, 3136, 1024, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_007 {{3136, 1024, 512, 3136, 1024, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_008 {{3136, 576, 64, 3136, 576, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_009 {{49, 4608, 512, 49, 4608, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_010 {{50176, 256, 128, 50176, 256, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_011 {{50176, 256, 512, 50176, 256, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_012 {{784, 1152, 128, 784, 1152, 784}, {1, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_resnet50_bwddata_fp16 = {
    conv_resnet50_bwddata_fp16_001, conv_resnet50_bwddata_fp16_002, 
    conv_resnet50_bwddata_fp16_003, conv_resnet50_bwddata_fp16_004, 
    conv_resnet50_bwddata_fp16_005, conv_resnet50_bwddata_fp16_006, 
    conv_resnet50_bwddata_fp16_007, conv_resnet50_bwddata_fp16_008, 
    conv_resnet50_bwddata_fp16_009, conv_resnet50_bwddata_fp16_010, 
    conv_resnet50_bwddata_fp16_011, conv_resnet50_bwddata_fp16_012, 
};

gemm_tuple conv_inception4_fwd_fp16_001 {{1225, 192, 1728, 1225, 1728, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_002 {{1225, 224, 1728, 1225, 1728, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_003 {{1225, 96, 576, 1225, 576, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_004 {{1225, 96, 864, 1225, 864, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_005 {{2048, 256, 1536, 2048, 1536, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_006 {{2048, 384, 1536, 2048, 1536, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_007 {{21609, 32, 288, 21609, 288, 21609}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_008 {{21609, 64, 288, 21609, 288, 21609}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_009 {{22201, 32, 27, 22201, 27, 22201}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_010 {{289, 192, 1344, 289, 1344, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_011 {{289, 224, 1344, 289, 1344, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_012 {{289, 224, 1568, 289, 1568, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_013 {{289, 256, 1568, 289, 1568, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_014 {{289, 256, 1792, 289, 1792, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_015 {{289, 256, 2016, 289, 2016, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_016 {{289, 320, 1792, 289, 1792, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_017 {{289, 384, 3456, 289, 3456, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_018 {{5041, 96, 576, 5041, 576, 5041}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_019 {{5329, 64, 448, 5329, 448, 5329}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_020 {{5329, 96, 576, 5329, 576, 5329}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_021 {{64, 192, 1728, 64, 1728, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_022 {{64, 256, 1152, 64, 1152, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_023 {{64, 256, 1536, 64, 1536, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_024 {{64, 320, 2880, 64, 2880, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_025 {{64, 448, 1152, 64, 1152, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp16_026 {{64, 512, 1344, 64, 1344, 64}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_inception4_fwd_fp16 = {
    conv_inception4_fwd_fp16_001, conv_inception4_fwd_fp16_002, conv_inception4_fwd_fp16_003, conv_inception4_fwd_fp16_004, 
    conv_inception4_fwd_fp16_005, conv_inception4_fwd_fp16_006, conv_inception4_fwd_fp16_007, conv_inception4_fwd_fp16_008, 
    conv_inception4_fwd_fp16_009, conv_inception4_fwd_fp16_010, conv_inception4_fwd_fp16_011, conv_inception4_fwd_fp16_012, 
    conv_inception4_fwd_fp16_013, conv_inception4_fwd_fp16_014, conv_inception4_fwd_fp16_015, conv_inception4_fwd_fp16_016, 
    conv_inception4_fwd_fp16_017, conv_inception4_fwd_fp16_018, conv_inception4_fwd_fp16_019, conv_inception4_fwd_fp16_020, 
    conv_inception4_fwd_fp16_021, conv_inception4_fwd_fp16_022, conv_inception4_fwd_fp16_023, conv_inception4_fwd_fp16_024, 
    conv_inception4_fwd_fp16_025, conv_inception4_fwd_fp16_026, 
};

gemm_tuple conv_inception4_fwd_fp32_001 {{1225, 192, 1728, 1225, 1728, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_002 {{1225, 224, 1728, 1225, 1728, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_003 {{1225, 96, 576, 1225, 576, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_004 {{1225, 96, 864, 1225, 864, 1225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_005 {{2048, 256, 1536, 2048, 1536, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_006 {{2048, 384, 1536, 2048, 1536, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_007 {{21609, 32, 288, 21609, 288, 21609}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_008 {{21609, 64, 288, 21609, 288, 21609}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_009 {{22201, 32, 27, 22201, 27, 22201}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_010 {{289, 192, 1344, 289, 1344, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_011 {{289, 224, 1344, 289, 1344, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_012 {{289, 224, 1568, 289, 1568, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_013 {{289, 256, 1568, 289, 1568, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_014 {{289, 256, 1792, 289, 1792, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_015 {{289, 256, 2016, 289, 2016, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_016 {{289, 320, 1792, 289, 1792, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_017 {{289, 384, 3456, 289, 3456, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_018 {{5041, 96, 576, 5041, 576, 5041}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_019 {{5329, 64, 448, 5329, 448, 5329}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_020 {{5329, 96, 576, 5329, 576, 5329}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_021 {{64, 192, 1728, 64, 1728, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_022 {{64, 256, 1152, 64, 1152, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_023 {{64, 256, 1536, 64, 1536, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_024 {{64, 320, 2880, 64, 2880, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_025 {{64, 448, 1152, 64, 1152, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_inception4_fwd_fp32_026 {{64, 512, 1344, 64, 1344, 64}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_inception4_fwd_fp32 = {
    conv_inception4_fwd_fp32_001, conv_inception4_fwd_fp32_002, conv_inception4_fwd_fp32_003, conv_inception4_fwd_fp32_004, 
    conv_inception4_fwd_fp32_005, conv_inception4_fwd_fp32_006, conv_inception4_fwd_fp32_007, conv_inception4_fwd_fp32_008, 
    conv_inception4_fwd_fp32_009, conv_inception4_fwd_fp32_010, conv_inception4_fwd_fp32_011, conv_inception4_fwd_fp32_012, 
    conv_inception4_fwd_fp32_013, conv_inception4_fwd_fp32_014, conv_inception4_fwd_fp32_015, conv_inception4_fwd_fp32_016, 
    conv_inception4_fwd_fp32_017, conv_inception4_fwd_fp32_018, conv_inception4_fwd_fp32_019, conv_inception4_fwd_fp32_020, 
    conv_inception4_fwd_fp32_021, conv_inception4_fwd_fp32_022, conv_inception4_fwd_fp32_023, conv_inception4_fwd_fp32_024, 
    conv_inception4_fwd_fp32_025, conv_inception4_fwd_fp32_026, 
};

gemm_tuple conv_inception4_bwdwrw_fp32_001 {{1024, 128, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_002 {{1024, 192, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_003 {{1024, 256, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_004 {{1024, 384, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_005 {{1152, 256, 64, 64, 64, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_006 {{1152, 448, 64, 64, 64, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_007 {{1344, 192, 289, 289, 289, 1344}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_008 {{1344, 224, 289, 289, 289, 1344}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_009 {{1344, 512, 64, 64, 64, 1344}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_010 {{1536, 256, 64, 64, 64, 1536}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_011 {{1536, 384, 64, 64, 64, 1536}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_012 {{1568, 224, 289, 289, 289, 1568}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_013 {{1568, 256, 289, 289, 289, 1568}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_014 {{160, 64, 5329, 5329, 5329, 160}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_015 {{1728, 192, 1225, 1225, 1225, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_016 {{1728, 192, 64, 64, 64, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_017 {{1728, 224, 1225, 1225, 1225, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_018 {{1792, 256, 289, 289, 289, 1792}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_019 {{1792, 320, 289, 289, 289, 1792}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_020 {{2016, 256, 289, 289, 289, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_021 {{27, 32, 22201, 22201, 22201, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_022 {{2880, 320, 64, 64, 64, 2880}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_023 {{288, 32, 21609, 21609, 21609, 288}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_024 {{288, 64, 21609, 21609, 21609, 288}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_025 {{3456, 384, 289, 289, 289, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_026 {{384, 192, 1225, 1225, 1225, 384}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_027 {{384, 64, 1225, 1225, 1225, 384}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_028 {{384, 96, 1225, 1225, 1225, 384}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_029 {{448, 64, 5329, 5329, 5329, 448}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_030 {{576, 96, 1225, 1225, 1225, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_031 {{576, 96, 5041, 5041, 5041, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_032 {{576, 96, 5329, 5329, 5329, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp32_033 {{864, 96, 1225, 1225, 1225, 864}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> conv_inception4_bwdwrw_fp32 = {
    conv_inception4_bwdwrw_fp32_001, conv_inception4_bwdwrw_fp32_002, 
    conv_inception4_bwdwrw_fp32_003, conv_inception4_bwdwrw_fp32_004, 
    conv_inception4_bwdwrw_fp32_005, conv_inception4_bwdwrw_fp32_006, 
    conv_inception4_bwdwrw_fp32_007, conv_inception4_bwdwrw_fp32_008, 
    conv_inception4_bwdwrw_fp32_009, conv_inception4_bwdwrw_fp32_010, 
    conv_inception4_bwdwrw_fp32_011, conv_inception4_bwdwrw_fp32_012, 
    conv_inception4_bwdwrw_fp32_013, conv_inception4_bwdwrw_fp32_014, 
    conv_inception4_bwdwrw_fp32_015, conv_inception4_bwdwrw_fp32_016, 
    conv_inception4_bwdwrw_fp32_017, conv_inception4_bwdwrw_fp32_018, 
    conv_inception4_bwdwrw_fp32_019, conv_inception4_bwdwrw_fp32_020, 
    conv_inception4_bwdwrw_fp32_021, conv_inception4_bwdwrw_fp32_022, 
    conv_inception4_bwdwrw_fp32_023, conv_inception4_bwdwrw_fp32_024, 
    conv_inception4_bwdwrw_fp32_025, conv_inception4_bwdwrw_fp32_026, 
    conv_inception4_bwdwrw_fp32_027, conv_inception4_bwdwrw_fp32_028, 
    conv_inception4_bwdwrw_fp32_029, conv_inception4_bwdwrw_fp32_030, 
    conv_inception4_bwdwrw_fp32_031, conv_inception4_bwdwrw_fp32_032, 
    conv_inception4_bwdwrw_fp32_033, 
};

gemm_tuple conv_inception4_bwdwrw_fp16_001 {{1024, 128, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_002 {{1024, 192, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_003 {{1024, 256, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_004 {{1024, 384, 289, 289, 289, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_005 {{1152, 256, 64, 64, 64, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_006 {{1152, 448, 64, 64, 64, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_007 {{1344, 192, 289, 289, 289, 1344}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_008 {{1344, 224, 289, 289, 289, 1344}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_009 {{1344, 512, 64, 64, 64, 1344}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_010 {{1536, 256, 64, 64, 64, 1536}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_011 {{1536, 384, 64, 64, 64, 1536}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_012 {{1568, 224, 289, 289, 289, 1568}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_013 {{1568, 256, 289, 289, 289, 1568}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_014 {{160, 64, 5329, 5329, 5329, 160}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_015 {{1728, 192, 1225, 1225, 1225, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_016 {{1728, 192, 64, 64, 64, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_017 {{1728, 224, 1225, 1225, 1225, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_018 {{1792, 256, 289, 289, 289, 1792}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_019 {{1792, 320, 289, 289, 289, 1792}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_020 {{2016, 256, 289, 289, 289, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_021 {{27, 32, 22201, 22201, 22201, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_022 {{2880, 320, 64, 64, 64, 2880}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_023 {{288, 32, 21609, 21609, 21609, 288}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_024 {{288, 64, 21609, 21609, 21609, 288}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_025 {{3456, 384, 289, 289, 289, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_026 {{384, 192, 1225, 1225, 1225, 384}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_027 {{384, 64, 1225, 1225, 1225, 384}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_028 {{384, 96, 1225, 1225, 1225, 384}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_029 {{448, 64, 5329, 5329, 5329, 448}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_030 {{576, 96, 1225, 1225, 1225, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_031 {{576, 96, 5041, 5041, 5041, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_032 {{576, 96, 5329, 5329, 5329, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_inception4_bwdwrw_fp16_033 {{864, 96, 1225, 1225, 1225, 864}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> conv_inception4_bwdwrw_fp16 = {
    conv_inception4_bwdwrw_fp16_001, conv_inception4_bwdwrw_fp16_002, 
    conv_inception4_bwdwrw_fp16_003, conv_inception4_bwdwrw_fp16_004, 
    conv_inception4_bwdwrw_fp16_005, conv_inception4_bwdwrw_fp16_006, 
    conv_inception4_bwdwrw_fp16_007, conv_inception4_bwdwrw_fp16_008, 
    conv_inception4_bwdwrw_fp16_009, conv_inception4_bwdwrw_fp16_010, 
    conv_inception4_bwdwrw_fp16_011, conv_inception4_bwdwrw_fp16_012, 
    conv_inception4_bwdwrw_fp16_013, conv_inception4_bwdwrw_fp16_014, 
    conv_inception4_bwdwrw_fp16_015, conv_inception4_bwdwrw_fp16_016, 
    conv_inception4_bwdwrw_fp16_017, conv_inception4_bwdwrw_fp16_018, 
    conv_inception4_bwdwrw_fp16_019, conv_inception4_bwdwrw_fp16_020, 
    conv_inception4_bwdwrw_fp16_021, conv_inception4_bwdwrw_fp16_022, 
    conv_inception4_bwdwrw_fp16_023, conv_inception4_bwdwrw_fp16_024, 
    conv_inception4_bwdwrw_fp16_025, conv_inception4_bwdwrw_fp16_026, 
    conv_inception4_bwdwrw_fp16_027, conv_inception4_bwdwrw_fp16_028, 
    conv_inception4_bwdwrw_fp16_029, conv_inception4_bwdwrw_fp16_030, 
    conv_inception4_bwdwrw_fp16_031, conv_inception4_bwdwrw_fp16_032, 
    conv_inception4_bwdwrw_fp16_033, 
};

gemm_tuple conv_inception4_bwddata_fp32_001 {{1225, 1728, 192, 1225, 1728, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_002 {{1225, 1728, 224, 1225, 1728, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_003 {{1225, 576, 96, 1225, 576, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_004 {{1225, 864, 96, 1225, 864, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_005 {{21609, 288, 32, 21609, 288, 21609}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_006 {{21609, 288, 64, 21609, 288, 21609}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_007 {{22201, 27, 32, 22201, 27, 22201}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_008 {{289, 1344, 192, 289, 1344, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_009 {{289, 1344, 224, 289, 1344, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_010 {{289, 1568, 224, 289, 1568, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_011 {{289, 1568, 256, 289, 1568, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_012 {{289, 1792, 256, 289, 1792, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_013 {{289, 1792, 320, 289, 1792, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_014 {{289, 2016, 256, 289, 2016, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_015 {{289, 3456, 384, 289, 3456, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_016 {{5041, 576, 96, 5041, 576, 5041}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_017 {{5329, 448, 64, 5329, 448, 5329}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_018 {{5329, 576, 96, 5329, 576, 5329}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_019 {{64, 1152, 256, 64, 1152, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_020 {{64, 1152, 448, 64, 1152, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_021 {{64, 1344, 512, 64, 1344, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_022 {{64, 1536, 256, 64, 1536, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_023 {{64, 1728, 192, 64, 1728, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp32_024 {{64, 2880, 320, 64, 2880, 64}, {1, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_inception4_bwddata_fp32 = {
    conv_inception4_bwddata_fp32_001, conv_inception4_bwddata_fp32_002, 
    conv_inception4_bwddata_fp32_003, conv_inception4_bwddata_fp32_004, 
    conv_inception4_bwddata_fp32_005, conv_inception4_bwddata_fp32_006, 
    conv_inception4_bwddata_fp32_007, conv_inception4_bwddata_fp32_008, 
    conv_inception4_bwddata_fp32_009, conv_inception4_bwddata_fp32_010, 
    conv_inception4_bwddata_fp32_011, conv_inception4_bwddata_fp32_012, 
    conv_inception4_bwddata_fp32_013, conv_inception4_bwddata_fp32_014, 
    conv_inception4_bwddata_fp32_015, conv_inception4_bwddata_fp32_016, 
    conv_inception4_bwddata_fp32_017, conv_inception4_bwddata_fp32_018, 
    conv_inception4_bwddata_fp32_019, conv_inception4_bwddata_fp32_020, 
    conv_inception4_bwddata_fp32_021, conv_inception4_bwddata_fp32_022, 
    conv_inception4_bwddata_fp32_023, conv_inception4_bwddata_fp32_024, 
};

gemm_tuple conv_inception4_bwddata_fp16_001 {{1225, 1728, 192, 1225, 1728, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_002 {{1225, 1728, 224, 1225, 1728, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_003 {{1225, 576, 96, 1225, 576, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_004 {{1225, 864, 96, 1225, 864, 1225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_005 {{21609, 288, 32, 21609, 288, 21609}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_006 {{21609, 288, 64, 21609, 288, 21609}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_007 {{22201, 27, 32, 22201, 27, 22201}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_008 {{289, 1344, 192, 289, 1344, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_009 {{289, 1344, 224, 289, 1344, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_010 {{289, 1568, 224, 289, 1568, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_011 {{289, 1568, 256, 289, 1568, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_012 {{289, 1792, 256, 289, 1792, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_013 {{289, 1792, 320, 289, 1792, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_014 {{289, 2016, 256, 289, 2016, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_015 {{289, 3456, 384, 289, 3456, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_016 {{5041, 576, 96, 5041, 576, 5041}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_017 {{5329, 448, 64, 5329, 448, 5329}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_018 {{5329, 576, 96, 5329, 576, 5329}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_019 {{64, 1152, 256, 64, 1152, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_020 {{64, 1152, 448, 64, 1152, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_021 {{64, 1344, 512, 64, 1344, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_022 {{64, 1536, 256, 64, 1536, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_023 {{64, 1728, 192, 64, 1728, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_inception4_bwddata_fp16_024 {{64, 2880, 320, 64, 2880, 64}, {1, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_inception4_bwddata_fp16 = {
    conv_inception4_bwddata_fp16_001, conv_inception4_bwddata_fp16_002, 
    conv_inception4_bwddata_fp16_003, conv_inception4_bwddata_fp16_004, 
    conv_inception4_bwddata_fp16_005, conv_inception4_bwddata_fp16_006, 
    conv_inception4_bwddata_fp16_007, conv_inception4_bwddata_fp16_008, 
    conv_inception4_bwddata_fp16_009, conv_inception4_bwddata_fp16_010, 
    conv_inception4_bwddata_fp16_011, conv_inception4_bwddata_fp16_012, 
    conv_inception4_bwddata_fp16_013, conv_inception4_bwddata_fp16_014, 
    conv_inception4_bwddata_fp16_015, conv_inception4_bwddata_fp16_016, 
    conv_inception4_bwddata_fp16_017, conv_inception4_bwddata_fp16_018, 
    conv_inception4_bwddata_fp16_019, conv_inception4_bwddata_fp16_020, 
    conv_inception4_bwddata_fp16_021, conv_inception4_bwddata_fp16_022, 
    conv_inception4_bwddata_fp16_023, conv_inception4_bwddata_fp16_024, 
};

gemm_tuple conv_ctest_bwddata_fp32_001 {{10000, 363, 1, 10000, 363, 10000}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_002 {{100, 1008, 1, 100, 1008, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_003 {{100, 1152, 1, 100, 1152, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_004 {{100, 128, 1, 100, 128, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_005 {{100, 1296, 1, 100, 1296, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_006 {{100, 1440, 1, 100, 1440, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_007 {{100, 1600, 1, 100, 1600, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_008 {{100, 1728, 1, 100, 1728, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_009 {{100, 192, 1, 100, 192, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_010 {{100, 2304, 1, 100, 2304, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_011 {{100, 2400, 1, 100, 2400, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_012 {{100, 256, 1, 100, 256, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_013 {{100, 400, 1, 100, 400, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_014 {{100, 4608, 1, 100, 4608, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_015 {{100, 480, 1, 100, 480, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_016 {{100, 4, 1, 100, 4, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_017 {{100, 512, 1, 100, 512, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_018 {{100, 528, 1, 100, 528, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_019 {{100, 576, 1, 100, 576, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_020 {{100, 600, 1, 100, 600, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_021 {{100, 608, 1, 100, 608, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_022 {{100, 64, 1, 100, 64, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_023 {{100, 800, 1, 100, 800, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_024 {{100, 864, 1, 100, 864, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_025 {{100, 9216, 1, 100, 9216, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_026 {{100, 9, 1, 100, 9, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_027 {{1024, 128, 1, 1024, 128, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_028 {{1024, 147, 1, 1024, 147, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_029 {{1024, 192, 1, 1024, 192, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_030 {{1024, 256, 1, 1024, 256, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_031 {{1024, 27, 1, 1024, 27, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_032 {{1024, 320, 1, 1024, 320, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_033 {{1024, 363, 1, 1024, 363, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_034 {{1024, 512, 1, 1024, 512, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_035 {{1024, 64, 1, 1024, 64, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_036 {{1024, 75, 1, 1024, 75, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_037 {{10404, 363, 1, 10404, 363, 10404}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_038 {{10609, 147, 1, 10609, 147, 10609}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_039 {{10816, 147, 1, 10816, 147, 10816}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_040 {{10816, 1600, 1, 10816, 1600, 10816}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_041 {{11025, 147, 1, 11025, 147, 11025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_042 {{11236, 147, 1, 11236, 147, 11236}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_043 {{11449, 147, 1, 11449, 147, 11449}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_044 {{11449, 363, 1, 11449, 363, 11449}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_045 {{11449, 75, 1, 11449, 75, 11449}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_046 {{1156, 27, 1, 1156, 27, 1156}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_047 {{11664, 147, 1, 11664, 147, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_048 {{11664, 1600, 1, 11664, 1600, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_049 {{11664, 363, 1, 11664, 363, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_050 {{11664, 576, 1, 11664, 576, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_051 {{11881, 147, 1, 11881, 147, 11881}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_052 {{11881, 363, 1, 11881, 363, 11881}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_053 {{11881, 75, 1, 11881, 75, 11881}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_054 {{12100, 147, 1, 12100, 147, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_055 {{12100, 1600, 1, 12100, 1600, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_056 {{12100, 27, 1, 12100, 27, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_057 {{12100, 363, 1, 12100, 363, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_058 {{12100, 576, 1, 12100, 576, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_059 {{12100, 75, 1, 12100, 75, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_060 {{121, 1024, 1, 121, 1024, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_061 {{121, 1056, 1, 121, 1056, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_062 {{121, 192, 1, 121, 192, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_063 {{121, 2304, 1, 121, 2304, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_064 {{121, 3456, 1, 121, 3456, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_065 {{121, 363, 1, 121, 363, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_066 {{121, 4, 1, 121, 4, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_067 {{121, 512, 1, 121, 512, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_068 {{121, 75, 1, 121, 75, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_069 {{121, 832, 1, 121, 832, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_070 {{12321, 147, 1, 12321, 147, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_071 {{12321, 27, 1, 12321, 27, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_072 {{12321, 363, 1, 12321, 363, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_073 {{12321, 75, 1, 12321, 75, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_074 {{12544, 147, 1, 12544, 147, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_075 {{12544, 1600, 1, 12544, 1600, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_076 {{12544, 27, 1, 12544, 27, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_077 {{12544, 363, 1, 12544, 363, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_078 {{12544, 576, 1, 12544, 576, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_079 {{12544, 75, 1, 12544, 75, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_080 {{12769, 147, 1, 12769, 147, 12769}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_081 {{12769, 27, 1, 12769, 27, 12769}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_082 {{12769, 75, 1, 12769, 75, 12769}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_083 {{12996, 147, 1, 12996, 147, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_084 {{12996, 27, 1, 12996, 27, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_085 {{12996, 363, 1, 12996, 363, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_086 {{12996, 576, 1, 12996, 576, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_087 {{12996, 64, 1, 12996, 64, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_088 {{12996, 75, 1, 12996, 75, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_089 {{13225, 27, 1, 13225, 27, 13225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_090 {{13225, 75, 1, 13225, 75, 13225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_091 {{13456, 147, 1, 13456, 147, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_092 {{13456, 27, 1, 13456, 27, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_093 {{13456, 363, 1, 13456, 363, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_094 {{13456, 64, 1, 13456, 64, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_095 {{13456, 75, 1, 13456, 75, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_096 {{13689, 75, 1, 13689, 75, 13689}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_097 {{13924, 27, 1, 13924, 27, 13924}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_098 {{144, 1008, 1, 144, 1008, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_099 {{144, 1152, 1, 144, 1152, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_100 {{144, 1296, 1, 144, 1296, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_101 {{144, 1440, 1, 144, 1440, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_102 {{144, 1600, 1, 144, 1600, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_103 {{144, 1728, 1, 144, 1728, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_104 {{144, 2304, 1, 144, 2304, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_105 {{144, 2400, 1, 144, 2400, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_106 {{144, 363, 1, 144, 363, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_107 {{144, 400, 1, 144, 400, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_108 {{144, 4608, 1, 144, 4608, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_109 {{144, 4, 1, 144, 4, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_110 {{144, 576, 1, 144, 576, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_111 {{144, 600, 1, 144, 600, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_112 {{144, 800, 1, 144, 800, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_113 {{144, 864, 1, 144, 864, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_114 {{144, 9216, 1, 144, 9216, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_115 {{144, 9, 1, 144, 9, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_116 {{169, 1152, 1, 169, 1152, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_117 {{169, 147, 1, 169, 147, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_118 {{169, 1600, 1, 169, 1600, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_119 {{169, 1728, 1, 169, 1728, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_120 {{169, 2048, 1, 169, 2048, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_121 {{169, 2304, 1, 169, 2304, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_122 {{169, 2400, 1, 169, 2400, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_123 {{169, 3456, 1, 169, 3456, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_124 {{169, 400, 1, 169, 400, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_125 {{169, 4608, 1, 169, 4608, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_126 {{169, 4, 1, 169, 4, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_127 {{169, 576, 1, 169, 576, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_128 {{169, 800, 1, 169, 800, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_129 {{169, 864, 1, 169, 864, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_130 {{169, 9, 1, 169, 9, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_131 {{16, 1024, 1, 16, 1024, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_132 {{16, 1056, 1, 16, 1056, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_133 {{16, 1200, 1, 16, 1200, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_134 {{16, 1440, 1, 16, 1440, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_135 {{16, 1728, 1, 16, 1728, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_136 {{16, 192, 1, 16, 192, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_137 {{16, 2016, 1, 16, 2016, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_138 {{16, 2304, 1, 16, 2304, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_139 {{16, 4608, 1, 16, 4608, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_140 {{16, 4, 1, 16, 4, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_141 {{16, 512, 1, 16, 512, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_142 {{16, 800, 1, 16, 800, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_143 {{16, 832, 1, 16, 832, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_144 {{16, 9216, 1, 16, 9216, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_145 {{16, 9, 1, 16, 9, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_146 {{1860, 4608, 1, 1860, 4608, 1860}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_147 {{1953, 4608, 1, 1953, 4608, 1953}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_148 {{196, 1008, 1, 196, 1008, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_149 {{196, 1024, 1, 196, 1024, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_150 {{196, 1152, 1, 196, 1152, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_151 {{196, 128, 1, 196, 128, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_152 {{196, 1296, 1, 196, 1296, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_153 {{196, 1440, 1, 196, 1440, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_154 {{196, 147, 1, 196, 147, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_155 {{196, 1600, 1, 196, 1600, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_156 {{196, 1728, 1, 196, 1728, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_157 {{196, 192, 1, 196, 192, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_158 {{196, 2304, 1, 196, 2304, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_159 {{196, 2400, 1, 196, 2400, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_160 {{196, 256, 1, 196, 256, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_161 {{196, 27, 1, 196, 27, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_162 {{196, 320, 1, 196, 320, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_163 {{196, 363, 1, 196, 363, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_164 {{196, 400, 1, 196, 400, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_165 {{196, 4608, 1, 196, 4608, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_166 {{196, 4, 1, 196, 4, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_167 {{196, 512, 1, 196, 512, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_168 {{196, 576, 1, 196, 576, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_169 {{196, 600, 1, 196, 600, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_170 {{196, 64, 1, 196, 64, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_171 {{196, 75, 1, 196, 75, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_172 {{196, 800, 1, 196, 800, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_173 {{196, 864, 1, 196, 864, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_174 {{196, 9216, 1, 196, 9216, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_175 {{196, 9, 1, 196, 9, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_176 {{1, 1200, 1, 1, 1200, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_177 {{1, 363, 1, 1, 363, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_178 {{1, 4608, 1, 1, 4608, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_179 {{1, 4, 1, 1, 4, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_180 {{1, 800, 1, 1, 800, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_181 {{1, 9, 1, 1, 9, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_182 {{2048, 4608, 1, 2048, 4608, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_183 {{2048, 480, 1, 2048, 480, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_184 {{2048, 512, 1, 2048, 512, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_185 {{2048, 528, 1, 2048, 528, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_186 {{2048, 832, 1, 2048, 832, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_187 {{2145, 480, 1, 2145, 480, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_188 {{2145, 512, 1, 2145, 512, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_189 {{2145, 528, 1, 2145, 528, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_190 {{2145, 832, 1, 2145, 832, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_191 {{2244, 4608, 1, 2244, 4608, 2244}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_192 {{225, 128, 1, 225, 128, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_193 {{225, 1600, 1, 225, 1600, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_194 {{225, 192, 1, 225, 192, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_195 {{225, 2048, 1, 225, 2048, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_196 {{225, 2304, 1, 225, 2304, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_197 {{225, 2400, 1, 225, 2400, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_198 {{225, 256, 1, 225, 256, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_199 {{225, 27, 1, 225, 27, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_200 {{225, 320, 1, 225, 320, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_201 {{225, 3456, 1, 225, 3456, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_202 {{225, 400, 1, 225, 400, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_203 {{225, 4, 1, 225, 4, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_204 {{225, 512, 1, 225, 512, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_205 {{225, 64, 1, 225, 64, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_206 {{225, 75, 1, 225, 75, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_207 {{225, 800, 1, 225, 800, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_208 {{2304, 1600, 1, 2304, 1600, 2304}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_209 {{2345, 480, 1, 2345, 480, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_210 {{2345, 512, 1, 2345, 512, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_211 {{2345, 528, 1, 2345, 528, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_212 {{2345, 832, 1, 2345, 832, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_213 {{256, 1008, 1, 256, 1008, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_214 {{256, 1024, 1, 256, 1024, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_215 {{256, 1152, 1, 256, 1152, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_216 {{256, 128, 1, 256, 128, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_217 {{256, 1296, 1, 256, 1296, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_218 {{256, 1440, 1, 256, 1440, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_219 {{256, 147, 1, 256, 147, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_220 {{256, 1728, 1, 256, 1728, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_221 {{256, 192, 1, 256, 192, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_222 {{256, 2304, 1, 256, 2304, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_223 {{256, 256, 1, 256, 256, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_224 {{256, 27, 1, 256, 27, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_225 {{256, 363, 1, 256, 363, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_226 {{256, 4608, 1, 256, 4608, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_227 {{256, 480, 1, 256, 480, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_228 {{256, 4, 1, 256, 4, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_229 {{256, 512, 1, 256, 512, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_230 {{256, 528, 1, 256, 528, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_231 {{256, 576, 1, 256, 576, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_232 {{256, 608, 1, 256, 608, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_233 {{256, 64, 1, 256, 64, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_234 {{256, 75, 1, 256, 75, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_235 {{256, 800, 1, 256, 800, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_236 {{256, 864, 1, 256, 864, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_237 {{256, 9, 1, 256, 9, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_238 {{25, 1008, 1, 25, 1008, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_239 {{25, 1024, 1, 25, 1024, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_240 {{25, 1056, 1, 25, 1056, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_241 {{25, 1152, 1, 25, 1152, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_242 {{25, 1200, 1, 25, 1200, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_243 {{25, 1296, 1, 25, 1296, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_244 {{25, 1440, 1, 25, 1440, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_245 {{25, 1600, 1, 25, 1600, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_246 {{25, 1728, 1, 25, 1728, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_247 {{25, 192, 1, 25, 192, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_248 {{25, 2016, 1, 25, 2016, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_249 {{25, 2304, 1, 25, 2304, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_250 {{25, 2400, 1, 25, 2400, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_251 {{25, 3456, 1, 25, 3456, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_252 {{25, 400, 1, 25, 400, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_253 {{25, 4608, 1, 25, 4608, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_254 {{25, 4, 1, 25, 4, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_255 {{25, 512, 1, 25, 512, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_256 {{25, 528, 1, 25, 528, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_257 {{25, 576, 1, 25, 576, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_258 {{25, 600, 1, 25, 600, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_259 {{25, 608, 1, 25, 608, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_260 {{25, 800, 1, 25, 800, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_261 {{25, 832, 1, 25, 832, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_262 {{25, 864, 1, 25, 864, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_263 {{25, 9216, 1, 25, 9216, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_264 {{25, 9, 1, 25, 9, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_265 {{2601, 1600, 1, 2601, 1600, 2601}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_266 {{2704, 1152, 1, 2704, 1152, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_267 {{2704, 1600, 1, 2704, 1600, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_268 {{2704, 2304, 1, 2704, 2304, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_269 {{2704, 576, 1, 2704, 576, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_270 {{289, 128, 1, 289, 128, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_271 {{289, 192, 1, 289, 192, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_272 {{289, 256, 1, 289, 256, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_273 {{289, 320, 1, 289, 320, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_274 {{289, 4, 1, 289, 4, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_275 {{289, 512, 1, 289, 512, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_276 {{289, 64, 1, 289, 64, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_277 {{289, 75, 1, 289, 75, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_278 {{2916, 1152, 1, 2916, 1152, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_279 {{2916, 1600, 1, 2916, 1600, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_280 {{2916, 2304, 1, 2916, 2304, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_281 {{2916, 576, 1, 2916, 576, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_282 {{3025, 1600, 1, 3025, 1600, 3025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_283 {{3025, 576, 1, 3025, 576, 3025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_284 {{3136, 1152, 1, 3136, 1152, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_285 {{3136, 1600, 1, 3136, 1600, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_286 {{3136, 2304, 1, 3136, 2304, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_287 {{3136, 576, 1, 3136, 576, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_288 {{3136, 64, 1, 3136, 64, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_289 {{3249, 1600, 1, 3249, 1600, 3249}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_290 {{3249, 64, 1, 3249, 64, 3249}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_291 {{324, 128, 1, 324, 128, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_292 {{324, 192, 1, 324, 192, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_293 {{324, 256, 1, 324, 256, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_294 {{324, 27, 1, 324, 27, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_295 {{324, 480, 1, 324, 480, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_296 {{324, 512, 1, 324, 512, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_297 {{324, 528, 1, 324, 528, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_298 {{324, 576, 1, 324, 576, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_299 {{324, 608, 1, 324, 608, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_300 {{324, 64, 1, 324, 64, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_301 {{33540, 480, 1, 33540, 480, 33540}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_302 {{3364, 1152, 1, 3364, 1152, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_303 {{3364, 128, 1, 3364, 128, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_304 {{3364, 2304, 1, 3364, 2304, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_305 {{3364, 256, 1, 3364, 256, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_306 {{3364, 576, 1, 3364, 576, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_307 {{3364, 64, 1, 3364, 64, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_308 {{34320, 480, 1, 34320, 480, 34320}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_309 {{3481, 64, 1, 3481, 64, 3481}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_310 {{3600, 128, 1, 3600, 128, 3600}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_311 {{3600, 256, 1, 3600, 256, 3600}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_312 {{3600, 64, 1, 3600, 64, 3600}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_313 {{361, 1600, 1, 361, 1600, 361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_314 {{361, 2400, 1, 361, 2400, 361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_315 {{36, 1008, 1, 36, 1008, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_316 {{36, 1024, 1, 36, 1024, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_317 {{36, 1152, 1, 36, 1152, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_318 {{36, 1296, 1, 36, 1296, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_319 {{36, 1440, 1, 36, 1440, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_320 {{36, 1600, 1, 36, 1600, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_321 {{36, 1728, 1, 36, 1728, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_322 {{36, 2016, 1, 36, 2016, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_323 {{36, 2048, 1, 36, 2048, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_324 {{36, 2304, 1, 36, 2304, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_325 {{36, 2400, 1, 36, 2400, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_326 {{36, 256, 1, 36, 256, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_327 {{36, 3456, 1, 36, 3456, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_328 {{36, 400, 1, 36, 400, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_329 {{36, 4608, 1, 36, 4608, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_330 {{36, 4, 1, 36, 4, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_331 {{36, 512, 1, 36, 512, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_332 {{36, 528, 1, 36, 528, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_333 {{36, 576, 1, 36, 576, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_334 {{36, 600, 1, 36, 600, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_335 {{36, 608, 1, 36, 608, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_336 {{36, 800, 1, 36, 800, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_337 {{36, 864, 1, 36, 864, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_338 {{36, 9216, 1, 36, 9216, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_339 {{36, 9, 1, 36, 9, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_340 {{400, 147, 1, 400, 147, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_341 {{400, 1600, 1, 400, 1600, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_342 {{400, 2400, 1, 400, 2400, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_343 {{400, 400, 1, 400, 400, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_344 {{400, 800, 1, 400, 800, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_345 {{41616, 363, 1, 41616, 363, 41616}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_346 {{42849, 363, 1, 42849, 363, 42849}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_347 {{44521, 363, 1, 44521, 363, 44521}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_348 {{44944, 147, 1, 44944, 147, 44944}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_349 {{45796, 363, 1, 45796, 363, 45796}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_350 {{46225, 147, 1, 46225, 147, 46225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_351 {{46656, 363, 1, 46656, 363, 46656}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_352 {{46656, 75, 1, 46656, 75, 46656}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_353 {{47089, 363, 1, 47089, 363, 47089}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_354 {{47524, 147, 1, 47524, 147, 47524}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_355 {{47524, 363, 1, 47524, 363, 47524}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_356 {{47961, 147, 1, 47961, 147, 47961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_357 {{47961, 363, 1, 47961, 363, 47961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_358 {{47961, 75, 1, 47961, 75, 47961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_359 {{48400, 147, 1, 48400, 147, 48400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_360 {{48400, 27, 1, 48400, 27, 48400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_361 {{48400, 75, 1, 48400, 75, 48400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_362 {{484, 363, 1, 484, 363, 484}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_363 {{48841, 147, 1, 48841, 147, 48841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_364 {{48841, 363, 1, 48841, 363, 48841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_365 {{49284, 147, 1, 49284, 147, 49284}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_366 {{49284, 27, 1, 49284, 27, 49284}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_367 {{49284, 75, 1, 49284, 75, 49284}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_368 {{49729, 147, 1, 49729, 147, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_369 {{49729, 27, 1, 49729, 27, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_370 {{49729, 363, 1, 49729, 363, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_371 {{49729, 75, 1, 49729, 75, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_372 {{49, 1008, 1, 49, 1008, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_373 {{49, 1024, 1, 49, 1024, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_374 {{49, 1056, 1, 49, 1056, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_375 {{49, 1152, 1, 49, 1152, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_376 {{49, 1200, 1, 49, 1200, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_377 {{49, 128, 1, 49, 128, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_378 {{49, 1296, 1, 49, 1296, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_379 {{49, 1440, 1, 49, 1440, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_380 {{49, 147, 1, 49, 147, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_381 {{49, 1600, 1, 49, 1600, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_382 {{49, 1728, 1, 49, 1728, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_383 {{49, 192, 1, 49, 192, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_384 {{49, 2016, 1, 49, 2016, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_385 {{49, 2048, 1, 49, 2048, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_386 {{49, 2304, 1, 49, 2304, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_387 {{49, 2400, 1, 49, 2400, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_388 {{49, 256, 1, 49, 256, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_389 {{49, 3456, 1, 49, 3456, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_390 {{49, 400, 1, 49, 400, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_391 {{49, 4608, 1, 49, 4608, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_392 {{49, 480, 1, 49, 480, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_393 {{49, 4, 1, 49, 4, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_394 {{49, 512, 1, 49, 512, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_395 {{49, 528, 1, 49, 528, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_396 {{49, 576, 1, 49, 576, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_397 {{49, 600, 1, 49, 600, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_398 {{49, 608, 1, 49, 608, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_399 {{49, 64, 1, 49, 64, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_400 {{49, 800, 1, 49, 800, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_401 {{49, 832, 1, 49, 832, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_402 {{49, 864, 1, 49, 864, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_403 {{49, 9216, 1, 49, 9216, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_404 {{49, 9, 1, 49, 9, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_405 {{4, 1200, 1, 4, 1200, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_406 {{4, 1440, 1, 4, 1440, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_407 {{4, 1600, 1, 4, 1600, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_408 {{4, 1728, 1, 4, 1728, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_409 {{4, 2016, 1, 4, 2016, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_410 {{4, 2400, 1, 4, 2400, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_411 {{4, 363, 1, 4, 363, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_412 {{4, 400, 1, 4, 400, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_413 {{4, 4608, 1, 4, 4608, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_414 {{4, 4, 1, 4, 4, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_415 {{4, 512, 1, 4, 512, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_416 {{4, 528, 1, 4, 528, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_417 {{4, 576, 1, 4, 576, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_418 {{4, 600, 1, 4, 600, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_419 {{4, 608, 1, 4, 608, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_420 {{4, 800, 1, 4, 800, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_421 {{4, 9216, 1, 4, 9216, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_422 {{4, 9, 1, 4, 9, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_423 {{50176, 147, 1, 50176, 147, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_424 {{50176, 27, 1, 50176, 27, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_425 {{50176, 363, 1, 50176, 363, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_426 {{50176, 75, 1, 50176, 75, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_427 {{50625, 147, 1, 50625, 147, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_428 {{50625, 27, 1, 50625, 27, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_429 {{50625, 363, 1, 50625, 363, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_430 {{50625, 75, 1, 50625, 75, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_431 {{51076, 27, 1, 51076, 27, 51076}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_432 {{51529, 147, 1, 51529, 147, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_433 {{51529, 27, 1, 51529, 27, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_434 {{51529, 363, 1, 51529, 363, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_435 {{51529, 75, 1, 51529, 75, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_436 {{52441, 147, 1, 52441, 147, 52441}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_437 {{52441, 27, 1, 52441, 27, 52441}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_438 {{52441, 75, 1, 52441, 75, 52441}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_439 {{529, 1600, 1, 529, 1600, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_440 {{529, 2400, 1, 529, 2400, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_441 {{529, 576, 1, 529, 576, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_442 {{529, 864, 1, 529, 864, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_443 {{529, 9, 1, 529, 9, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_444 {{53361, 147, 1, 53361, 147, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_445 {{53361, 27, 1, 53361, 27, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_446 {{53361, 363, 1, 53361, 363, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_447 {{53361, 75, 1, 53361, 75, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_448 {{54289, 27, 1, 54289, 27, 54289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_449 {{576, 1152, 1, 576, 1152, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_450 {{576, 1600, 1, 576, 1600, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_451 {{576, 1728, 1, 576, 1728, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_452 {{576, 2304, 1, 576, 2304, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_453 {{576, 2400, 1, 576, 2400, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_454 {{576, 363, 1, 576, 363, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_455 {{576, 400, 1, 576, 400, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_456 {{576, 4608, 1, 576, 4608, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_457 {{576, 576, 1, 576, 576, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_458 {{576, 75, 1, 576, 75, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_459 {{576, 800, 1, 576, 800, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_460 {{576, 864, 1, 576, 864, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_461 {{625, 1600, 1, 625, 1600, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_462 {{625, 2400, 1, 625, 2400, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_463 {{625, 4, 1, 625, 4, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_464 {{625, 576, 1, 625, 576, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_465 {{625, 864, 1, 625, 864, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_466 {{625, 9, 1, 625, 9, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_467 {{64, 128, 1, 64, 128, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_468 {{64, 147, 1, 64, 147, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_469 {{64, 1600, 1, 64, 1600, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_470 {{64, 192, 1, 64, 192, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_471 {{64, 2304, 1, 64, 2304, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_472 {{64, 2400, 1, 64, 2400, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_473 {{64, 256, 1, 64, 256, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_474 {{64, 400, 1, 64, 400, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_475 {{64, 4608, 1, 64, 4608, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_476 {{64, 480, 1, 64, 480, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_477 {{64, 4, 1, 64, 4, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_478 {{64, 512, 1, 64, 512, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_479 {{64, 528, 1, 64, 528, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_480 {{64, 576, 1, 64, 576, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_481 {{64, 600, 1, 64, 600, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_482 {{64, 608, 1, 64, 608, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_483 {{64, 64, 1, 64, 64, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_484 {{64, 800, 1, 64, 800, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_485 {{64, 9216, 1, 64, 9216, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_486 {{64, 9, 1, 64, 9, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_487 {{676, 1152, 1, 676, 1152, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_488 {{676, 147, 1, 676, 147, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_489 {{676, 1600, 1, 676, 1600, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_490 {{676, 1728, 1, 676, 1728, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_491 {{676, 2304, 1, 676, 2304, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_492 {{676, 2400, 1, 676, 2400, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_493 {{676, 363, 1, 676, 363, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_494 {{676, 400, 1, 676, 400, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_495 {{676, 4608, 1, 676, 4608, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_496 {{676, 4, 1, 676, 4, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_497 {{676, 576, 1, 676, 576, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_498 {{676, 800, 1, 676, 800, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_499 {{676, 864, 1, 676, 864, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_500 {{729, 1152, 1, 729, 1152, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_501 {{729, 1600, 1, 729, 1600, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_502 {{729, 2304, 1, 729, 2304, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_503 {{729, 2400, 1, 729, 2400, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_504 {{729, 4, 1, 729, 4, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_505 {{729, 576, 1, 729, 576, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_506 {{729, 864, 1, 729, 864, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_507 {{729, 9, 1, 729, 9, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_508 {{7440, 4608, 1, 7440, 4608, 7440}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_509 {{7812, 4608, 1, 7812, 4608, 7812}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_510 {{784, 1152, 1, 784, 1152, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_511 {{784, 128, 1, 784, 128, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_512 {{784, 147, 1, 784, 147, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_513 {{784, 1600, 1, 784, 1600, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_514 {{784, 1728, 1, 784, 1728, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_515 {{784, 2304, 1, 784, 2304, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_516 {{784, 2400, 1, 784, 2400, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_517 {{784, 256, 1, 784, 256, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_518 {{784, 27, 1, 784, 27, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_519 {{784, 400, 1, 784, 400, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_520 {{784, 4608, 1, 784, 4608, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_521 {{784, 4, 1, 784, 4, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_522 {{784, 576, 1, 784, 576, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_523 {{784, 64, 1, 784, 64, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_524 {{784, 75, 1, 784, 75, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_525 {{784, 800, 1, 784, 800, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_526 {{784, 864, 1, 784, 864, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_527 {{8192, 4608, 1, 8192, 4608, 8192}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_528 {{8192, 480, 1, 8192, 480, 8192}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_529 {{81, 1008, 1, 81, 1008, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_530 {{81, 1024, 1, 81, 1024, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_531 {{81, 1056, 1, 81, 1056, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_532 {{81, 1152, 1, 81, 1152, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_533 {{81, 1296, 1, 81, 1296, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_534 {{81, 1440, 1, 81, 1440, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_535 {{81, 1600, 1, 81, 1600, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_536 {{81, 1728, 1, 81, 1728, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_537 {{81, 192, 1, 81, 192, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_538 {{81, 2016, 1, 81, 2016, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_539 {{81, 2048, 1, 81, 2048, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_540 {{81, 2304, 1, 81, 2304, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_541 {{81, 2400, 1, 81, 2400, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_542 {{81, 256, 1, 81, 256, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_543 {{81, 3456, 1, 81, 3456, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_544 {{81, 400, 1, 81, 400, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_545 {{81, 4608, 1, 81, 4608, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_546 {{81, 4, 1, 81, 4, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_547 {{81, 512, 1, 81, 512, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_548 {{81, 576, 1, 81, 576, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_549 {{81, 800, 1, 81, 800, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_550 {{81, 832, 1, 81, 832, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_551 {{81, 864, 1, 81, 864, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_552 {{81, 9216, 1, 81, 9216, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_553 {{81, 9, 1, 81, 9, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_554 {{8385, 480, 1, 8385, 480, 8385}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_555 {{841, 128, 1, 841, 128, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_556 {{841, 1600, 1, 841, 1600, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_557 {{841, 256, 1, 841, 256, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_558 {{841, 576, 1, 841, 576, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_559 {{841, 64, 1, 841, 64, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_560 {{841, 864, 1, 841, 864, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_561 {{841, 9, 1, 841, 9, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_562 {{8580, 4608, 1, 8580, 4608, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_563 {{8580, 480, 1, 8580, 480, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_564 {{8580, 512, 1, 8580, 512, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_565 {{8580, 528, 1, 8580, 528, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_566 {{8580, 832, 1, 8580, 832, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_567 {{8777, 480, 1, 8777, 480, 8777}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_568 {{8976, 480, 1, 8976, 480, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_569 {{8976, 512, 1, 8976, 512, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_570 {{8976, 528, 1, 8976, 528, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_571 {{8976, 832, 1, 8976, 832, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_572 {{900, 1152, 1, 900, 1152, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_573 {{900, 128, 1, 900, 128, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_574 {{900, 147, 1, 900, 147, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_575 {{900, 1728, 1, 900, 1728, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_576 {{900, 192, 1, 900, 192, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_577 {{900, 2304, 1, 900, 2304, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_578 {{900, 256, 1, 900, 256, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_579 {{900, 27, 1, 900, 27, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_580 {{900, 320, 1, 900, 320, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_581 {{900, 4608, 1, 900, 4608, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_582 {{900, 4, 1, 900, 4, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_583 {{900, 512, 1, 900, 512, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_584 {{900, 576, 1, 900, 576, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_585 {{900, 64, 1, 900, 64, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_586 {{900, 75, 1, 900, 75, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_587 {{900, 864, 1, 900, 864, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_588 {{9025, 363, 1, 9025, 363, 9025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_589 {{9409, 363, 1, 9409, 363, 9409}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_590 {{9604, 363, 1, 9604, 363, 9604}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_591 {{961, 128, 1, 961, 128, 961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_592 {{961, 256, 1, 961, 256, 961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_593 {{961, 64, 1, 961, 64, 961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_594 {{9801, 363, 1, 9801, 363, 9801}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_595 {{9, 1200, 1, 9, 1200, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_596 {{9, 1440, 1, 9, 1440, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_597 {{9, 1728, 1, 9, 1728, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_598 {{9, 2016, 1, 9, 2016, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_599 {{9, 4608, 1, 9, 4608, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_600 {{9, 4, 1, 9, 4, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_601 {{9, 512, 1, 9, 512, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_602 {{9, 528, 1, 9, 528, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_603 {{9, 576, 1, 9, 576, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_604 {{9, 608, 1, 9, 608, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_605 {{9, 800, 1, 9, 800, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_606 {{9, 9216, 1, 9, 9216, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp32_607 {{9, 9, 1, 9, 9, 9}, {1, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_ctest_bwddata_fp32 = {
conv_ctest_bwddata_fp32_001, conv_ctest_bwddata_fp32_002, 
conv_ctest_bwddata_fp32_003, conv_ctest_bwddata_fp32_004, 
conv_ctest_bwddata_fp32_005, conv_ctest_bwddata_fp32_006, 
conv_ctest_bwddata_fp32_007, conv_ctest_bwddata_fp32_008, 
conv_ctest_bwddata_fp32_009, conv_ctest_bwddata_fp32_010, 
conv_ctest_bwddata_fp32_011, conv_ctest_bwddata_fp32_012, 
conv_ctest_bwddata_fp32_013, conv_ctest_bwddata_fp32_014, 
conv_ctest_bwddata_fp32_015, conv_ctest_bwddata_fp32_016, 
conv_ctest_bwddata_fp32_017, conv_ctest_bwddata_fp32_018, 
conv_ctest_bwddata_fp32_019, conv_ctest_bwddata_fp32_020, 
conv_ctest_bwddata_fp32_021, conv_ctest_bwddata_fp32_022, 
conv_ctest_bwddata_fp32_023, conv_ctest_bwddata_fp32_024, 
conv_ctest_bwddata_fp32_025, conv_ctest_bwddata_fp32_026, 
conv_ctest_bwddata_fp32_027, conv_ctest_bwddata_fp32_028, 
conv_ctest_bwddata_fp32_029, conv_ctest_bwddata_fp32_030, 
conv_ctest_bwddata_fp32_031, conv_ctest_bwddata_fp32_032, 
conv_ctest_bwddata_fp32_033, conv_ctest_bwddata_fp32_034, 
conv_ctest_bwddata_fp32_035, conv_ctest_bwddata_fp32_036, 
conv_ctest_bwddata_fp32_037, conv_ctest_bwddata_fp32_038, 
conv_ctest_bwddata_fp32_039, conv_ctest_bwddata_fp32_040, 
conv_ctest_bwddata_fp32_041, conv_ctest_bwddata_fp32_042, 
conv_ctest_bwddata_fp32_043, conv_ctest_bwddata_fp32_044, 
conv_ctest_bwddata_fp32_045, conv_ctest_bwddata_fp32_046, 
conv_ctest_bwddata_fp32_047, conv_ctest_bwddata_fp32_048, 
conv_ctest_bwddata_fp32_049, conv_ctest_bwddata_fp32_050, 
conv_ctest_bwddata_fp32_051, conv_ctest_bwddata_fp32_052, 
conv_ctest_bwddata_fp32_053, conv_ctest_bwddata_fp32_054, 
conv_ctest_bwddata_fp32_055, conv_ctest_bwddata_fp32_056, 
conv_ctest_bwddata_fp32_057, conv_ctest_bwddata_fp32_058, 
conv_ctest_bwddata_fp32_059, conv_ctest_bwddata_fp32_060, 
conv_ctest_bwddata_fp32_061, conv_ctest_bwddata_fp32_062, 
conv_ctest_bwddata_fp32_063, conv_ctest_bwddata_fp32_064, 
conv_ctest_bwddata_fp32_065, conv_ctest_bwddata_fp32_066, 
conv_ctest_bwddata_fp32_067, conv_ctest_bwddata_fp32_068, 
conv_ctest_bwddata_fp32_069, conv_ctest_bwddata_fp32_070, 
conv_ctest_bwddata_fp32_071, conv_ctest_bwddata_fp32_072, 
conv_ctest_bwddata_fp32_073, conv_ctest_bwddata_fp32_074, 
conv_ctest_bwddata_fp32_075, conv_ctest_bwddata_fp32_076, 
conv_ctest_bwddata_fp32_077, conv_ctest_bwddata_fp32_078, 
conv_ctest_bwddata_fp32_079, conv_ctest_bwddata_fp32_080, 
conv_ctest_bwddata_fp32_081, conv_ctest_bwddata_fp32_082, 
conv_ctest_bwddata_fp32_083, conv_ctest_bwddata_fp32_084, 
conv_ctest_bwddata_fp32_085, conv_ctest_bwddata_fp32_086, 
conv_ctest_bwddata_fp32_087, conv_ctest_bwddata_fp32_088, 
conv_ctest_bwddata_fp32_089, conv_ctest_bwddata_fp32_090, 
conv_ctest_bwddata_fp32_091, conv_ctest_bwddata_fp32_092, 
conv_ctest_bwddata_fp32_093, conv_ctest_bwddata_fp32_094, 
conv_ctest_bwddata_fp32_095, conv_ctest_bwddata_fp32_096, 
conv_ctest_bwddata_fp32_097, conv_ctest_bwddata_fp32_098, 
conv_ctest_bwddata_fp32_099, conv_ctest_bwddata_fp32_100, 
conv_ctest_bwddata_fp32_101, conv_ctest_bwddata_fp32_102, 
conv_ctest_bwddata_fp32_103, conv_ctest_bwddata_fp32_104, 
conv_ctest_bwddata_fp32_105, conv_ctest_bwddata_fp32_106, 
conv_ctest_bwddata_fp32_107, conv_ctest_bwddata_fp32_108, 
conv_ctest_bwddata_fp32_109, conv_ctest_bwddata_fp32_110, 
conv_ctest_bwddata_fp32_111, conv_ctest_bwddata_fp32_112, 
conv_ctest_bwddata_fp32_113, conv_ctest_bwddata_fp32_114, 
conv_ctest_bwddata_fp32_115, conv_ctest_bwddata_fp32_116, 
conv_ctest_bwddata_fp32_117, conv_ctest_bwddata_fp32_118, 
conv_ctest_bwddata_fp32_119, conv_ctest_bwddata_fp32_120, 
conv_ctest_bwddata_fp32_121, conv_ctest_bwddata_fp32_122, 
conv_ctest_bwddata_fp32_123, conv_ctest_bwddata_fp32_124, 
conv_ctest_bwddata_fp32_125, conv_ctest_bwddata_fp32_126, 
conv_ctest_bwddata_fp32_127, conv_ctest_bwddata_fp32_128, 
conv_ctest_bwddata_fp32_129, conv_ctest_bwddata_fp32_130, 
conv_ctest_bwddata_fp32_131, conv_ctest_bwddata_fp32_132, 
conv_ctest_bwddata_fp32_133, conv_ctest_bwddata_fp32_134, 
conv_ctest_bwddata_fp32_135, conv_ctest_bwddata_fp32_136, 
conv_ctest_bwddata_fp32_137, conv_ctest_bwddata_fp32_138, 
conv_ctest_bwddata_fp32_139, conv_ctest_bwddata_fp32_140, 
conv_ctest_bwddata_fp32_141, conv_ctest_bwddata_fp32_142, 
conv_ctest_bwddata_fp32_143, conv_ctest_bwddata_fp32_144, 
conv_ctest_bwddata_fp32_145, conv_ctest_bwddata_fp32_146, 
conv_ctest_bwddata_fp32_147, conv_ctest_bwddata_fp32_148, 
conv_ctest_bwddata_fp32_149, conv_ctest_bwddata_fp32_150, 
conv_ctest_bwddata_fp32_151, conv_ctest_bwddata_fp32_152, 
conv_ctest_bwddata_fp32_153, conv_ctest_bwddata_fp32_154, 
conv_ctest_bwddata_fp32_155, conv_ctest_bwddata_fp32_156, 
conv_ctest_bwddata_fp32_157, conv_ctest_bwddata_fp32_158, 
conv_ctest_bwddata_fp32_159, conv_ctest_bwddata_fp32_160, 
conv_ctest_bwddata_fp32_161, conv_ctest_bwddata_fp32_162, 
conv_ctest_bwddata_fp32_163, conv_ctest_bwddata_fp32_164, 
conv_ctest_bwddata_fp32_165, conv_ctest_bwddata_fp32_166, 
conv_ctest_bwddata_fp32_167, conv_ctest_bwddata_fp32_168, 
conv_ctest_bwddata_fp32_169, conv_ctest_bwddata_fp32_170, 
conv_ctest_bwddata_fp32_171, conv_ctest_bwddata_fp32_172, 
conv_ctest_bwddata_fp32_173, conv_ctest_bwddata_fp32_174, 
conv_ctest_bwddata_fp32_175, conv_ctest_bwddata_fp32_176, 
conv_ctest_bwddata_fp32_177, conv_ctest_bwddata_fp32_178, 
conv_ctest_bwddata_fp32_179, conv_ctest_bwddata_fp32_180, 
conv_ctest_bwddata_fp32_181, conv_ctest_bwddata_fp32_182, 
conv_ctest_bwddata_fp32_183, conv_ctest_bwddata_fp32_184, 
conv_ctest_bwddata_fp32_185, conv_ctest_bwddata_fp32_186, 
conv_ctest_bwddata_fp32_187, conv_ctest_bwddata_fp32_188, 
conv_ctest_bwddata_fp32_189, conv_ctest_bwddata_fp32_190, 
conv_ctest_bwddata_fp32_191, conv_ctest_bwddata_fp32_192, 
conv_ctest_bwddata_fp32_193, conv_ctest_bwddata_fp32_194, 
conv_ctest_bwddata_fp32_195, conv_ctest_bwddata_fp32_196, 
conv_ctest_bwddata_fp32_197, conv_ctest_bwddata_fp32_198, 
conv_ctest_bwddata_fp32_199, conv_ctest_bwddata_fp32_200, 
conv_ctest_bwddata_fp32_201, conv_ctest_bwddata_fp32_202, 
conv_ctest_bwddata_fp32_203, conv_ctest_bwddata_fp32_204, 
conv_ctest_bwddata_fp32_205, conv_ctest_bwddata_fp32_206, 
conv_ctest_bwddata_fp32_207, conv_ctest_bwddata_fp32_208, 
conv_ctest_bwddata_fp32_209, conv_ctest_bwddata_fp32_210, 
conv_ctest_bwddata_fp32_211, conv_ctest_bwddata_fp32_212, 
conv_ctest_bwddata_fp32_213, conv_ctest_bwddata_fp32_214, 
conv_ctest_bwddata_fp32_215, conv_ctest_bwddata_fp32_216, 
conv_ctest_bwddata_fp32_217, conv_ctest_bwddata_fp32_218, 
conv_ctest_bwddata_fp32_219, conv_ctest_bwddata_fp32_220, 
conv_ctest_bwddata_fp32_221, conv_ctest_bwddata_fp32_222, 
conv_ctest_bwddata_fp32_223, conv_ctest_bwddata_fp32_224, 
conv_ctest_bwddata_fp32_225, conv_ctest_bwddata_fp32_226, 
conv_ctest_bwddata_fp32_227, conv_ctest_bwddata_fp32_228, 
conv_ctest_bwddata_fp32_229, conv_ctest_bwddata_fp32_230, 
conv_ctest_bwddata_fp32_231, conv_ctest_bwddata_fp32_232, 
conv_ctest_bwddata_fp32_233, conv_ctest_bwddata_fp32_234, 
conv_ctest_bwddata_fp32_235, conv_ctest_bwddata_fp32_236, 
conv_ctest_bwddata_fp32_237, conv_ctest_bwddata_fp32_238, 
conv_ctest_bwddata_fp32_239, conv_ctest_bwddata_fp32_240, 
conv_ctest_bwddata_fp32_241, conv_ctest_bwddata_fp32_242, 
conv_ctest_bwddata_fp32_243, conv_ctest_bwddata_fp32_244, 
conv_ctest_bwddata_fp32_245, conv_ctest_bwddata_fp32_246, 
conv_ctest_bwddata_fp32_247, conv_ctest_bwddata_fp32_248, 
conv_ctest_bwddata_fp32_249, conv_ctest_bwddata_fp32_250, 
conv_ctest_bwddata_fp32_251, conv_ctest_bwddata_fp32_252, 
conv_ctest_bwddata_fp32_253, conv_ctest_bwddata_fp32_254, 
conv_ctest_bwddata_fp32_255, conv_ctest_bwddata_fp32_256, 
conv_ctest_bwddata_fp32_257, conv_ctest_bwddata_fp32_258, 
conv_ctest_bwddata_fp32_259, conv_ctest_bwddata_fp32_260, 
conv_ctest_bwddata_fp32_261, conv_ctest_bwddata_fp32_262, 
conv_ctest_bwddata_fp32_263, conv_ctest_bwddata_fp32_264, 
conv_ctest_bwddata_fp32_265, conv_ctest_bwddata_fp32_266, 
conv_ctest_bwddata_fp32_267, conv_ctest_bwddata_fp32_268, 
conv_ctest_bwddata_fp32_269, conv_ctest_bwddata_fp32_270, 
conv_ctest_bwddata_fp32_271, conv_ctest_bwddata_fp32_272, 
conv_ctest_bwddata_fp32_273, conv_ctest_bwddata_fp32_274, 
conv_ctest_bwddata_fp32_275, conv_ctest_bwddata_fp32_276, 
conv_ctest_bwddata_fp32_277, conv_ctest_bwddata_fp32_278, 
conv_ctest_bwddata_fp32_279, conv_ctest_bwddata_fp32_280, 
conv_ctest_bwddata_fp32_281, conv_ctest_bwddata_fp32_282, 
conv_ctest_bwddata_fp32_283, conv_ctest_bwddata_fp32_284, 
conv_ctest_bwddata_fp32_285, conv_ctest_bwddata_fp32_286, 
conv_ctest_bwddata_fp32_287, conv_ctest_bwddata_fp32_288, 
conv_ctest_bwddata_fp32_289, conv_ctest_bwddata_fp32_290, 
conv_ctest_bwddata_fp32_291, conv_ctest_bwddata_fp32_292, 
conv_ctest_bwddata_fp32_293, conv_ctest_bwddata_fp32_294, 
conv_ctest_bwddata_fp32_295, conv_ctest_bwddata_fp32_296, 
conv_ctest_bwddata_fp32_297, conv_ctest_bwddata_fp32_298, 
conv_ctest_bwddata_fp32_299, conv_ctest_bwddata_fp32_300, 
conv_ctest_bwddata_fp32_301, conv_ctest_bwddata_fp32_302, 
conv_ctest_bwddata_fp32_303, conv_ctest_bwddata_fp32_304, 
conv_ctest_bwddata_fp32_305, conv_ctest_bwddata_fp32_306, 
conv_ctest_bwddata_fp32_307, conv_ctest_bwddata_fp32_308, 
conv_ctest_bwddata_fp32_309, conv_ctest_bwddata_fp32_310, 
conv_ctest_bwddata_fp32_311, conv_ctest_bwddata_fp32_312, 
conv_ctest_bwddata_fp32_313, conv_ctest_bwddata_fp32_314, 
conv_ctest_bwddata_fp32_315, conv_ctest_bwddata_fp32_316, 
conv_ctest_bwddata_fp32_317, conv_ctest_bwddata_fp32_318, 
conv_ctest_bwddata_fp32_319, conv_ctest_bwddata_fp32_320, 
conv_ctest_bwddata_fp32_321, conv_ctest_bwddata_fp32_322, 
conv_ctest_bwddata_fp32_323, conv_ctest_bwddata_fp32_324, 
conv_ctest_bwddata_fp32_325, conv_ctest_bwddata_fp32_326, 
conv_ctest_bwddata_fp32_327, conv_ctest_bwddata_fp32_328, 
conv_ctest_bwddata_fp32_329, conv_ctest_bwddata_fp32_330, 
conv_ctest_bwddata_fp32_331, conv_ctest_bwddata_fp32_332, 
conv_ctest_bwddata_fp32_333, conv_ctest_bwddata_fp32_334, 
conv_ctest_bwddata_fp32_335, conv_ctest_bwddata_fp32_336, 
conv_ctest_bwddata_fp32_337, conv_ctest_bwddata_fp32_338, 
conv_ctest_bwddata_fp32_339, conv_ctest_bwddata_fp32_340, 
conv_ctest_bwddata_fp32_341, conv_ctest_bwddata_fp32_342, 
conv_ctest_bwddata_fp32_343, conv_ctest_bwddata_fp32_344, 
conv_ctest_bwddata_fp32_345, conv_ctest_bwddata_fp32_346, 
conv_ctest_bwddata_fp32_347, conv_ctest_bwddata_fp32_348, 
conv_ctest_bwddata_fp32_349, conv_ctest_bwddata_fp32_350, 
conv_ctest_bwddata_fp32_351, conv_ctest_bwddata_fp32_352, 
conv_ctest_bwddata_fp32_353, conv_ctest_bwddata_fp32_354, 
conv_ctest_bwddata_fp32_355, conv_ctest_bwddata_fp32_356, 
conv_ctest_bwddata_fp32_357, conv_ctest_bwddata_fp32_358, 
conv_ctest_bwddata_fp32_359, conv_ctest_bwddata_fp32_360, 
conv_ctest_bwddata_fp32_361, conv_ctest_bwddata_fp32_362, 
conv_ctest_bwddata_fp32_363, conv_ctest_bwddata_fp32_364, 
conv_ctest_bwddata_fp32_365, conv_ctest_bwddata_fp32_366, 
conv_ctest_bwddata_fp32_367, conv_ctest_bwddata_fp32_368, 
conv_ctest_bwddata_fp32_369, conv_ctest_bwddata_fp32_370, 
conv_ctest_bwddata_fp32_371, conv_ctest_bwddata_fp32_372, 
conv_ctest_bwddata_fp32_373, conv_ctest_bwddata_fp32_374, 
conv_ctest_bwddata_fp32_375, conv_ctest_bwddata_fp32_376, 
conv_ctest_bwddata_fp32_377, conv_ctest_bwddata_fp32_378, 
conv_ctest_bwddata_fp32_379, conv_ctest_bwddata_fp32_380, 
conv_ctest_bwddata_fp32_381, conv_ctest_bwddata_fp32_382, 
conv_ctest_bwddata_fp32_383, conv_ctest_bwddata_fp32_384, 
conv_ctest_bwddata_fp32_385, conv_ctest_bwddata_fp32_386, 
conv_ctest_bwddata_fp32_387, conv_ctest_bwddata_fp32_388, 
conv_ctest_bwddata_fp32_389, conv_ctest_bwddata_fp32_390, 
conv_ctest_bwddata_fp32_391, conv_ctest_bwddata_fp32_392, 
conv_ctest_bwddata_fp32_393, conv_ctest_bwddata_fp32_394, 
conv_ctest_bwddata_fp32_395, conv_ctest_bwddata_fp32_396, 
conv_ctest_bwddata_fp32_397, conv_ctest_bwddata_fp32_398, 
conv_ctest_bwddata_fp32_399, conv_ctest_bwddata_fp32_400, 
conv_ctest_bwddata_fp32_401, conv_ctest_bwddata_fp32_402, 
conv_ctest_bwddata_fp32_403, conv_ctest_bwddata_fp32_404, 
conv_ctest_bwddata_fp32_405, conv_ctest_bwddata_fp32_406, 
conv_ctest_bwddata_fp32_407, conv_ctest_bwddata_fp32_408, 
conv_ctest_bwddata_fp32_409, conv_ctest_bwddata_fp32_410, 
conv_ctest_bwddata_fp32_411, conv_ctest_bwddata_fp32_412, 
conv_ctest_bwddata_fp32_413, conv_ctest_bwddata_fp32_414, 
conv_ctest_bwddata_fp32_415, conv_ctest_bwddata_fp32_416, 
conv_ctest_bwddata_fp32_417, conv_ctest_bwddata_fp32_418, 
conv_ctest_bwddata_fp32_419, conv_ctest_bwddata_fp32_420, 
conv_ctest_bwddata_fp32_421, conv_ctest_bwddata_fp32_422, 
conv_ctest_bwddata_fp32_423, conv_ctest_bwddata_fp32_424, 
conv_ctest_bwddata_fp32_425, conv_ctest_bwddata_fp32_426, 
conv_ctest_bwddata_fp32_427, conv_ctest_bwddata_fp32_428, 
conv_ctest_bwddata_fp32_429, conv_ctest_bwddata_fp32_430, 
conv_ctest_bwddata_fp32_431, conv_ctest_bwddata_fp32_432, 
conv_ctest_bwddata_fp32_433, conv_ctest_bwddata_fp32_434, 
conv_ctest_bwddata_fp32_435, conv_ctest_bwddata_fp32_436, 
conv_ctest_bwddata_fp32_437, conv_ctest_bwddata_fp32_438, 
conv_ctest_bwddata_fp32_439, conv_ctest_bwddata_fp32_440, 
conv_ctest_bwddata_fp32_441, conv_ctest_bwddata_fp32_442, 
conv_ctest_bwddata_fp32_443, conv_ctest_bwddata_fp32_444, 
conv_ctest_bwddata_fp32_445, conv_ctest_bwddata_fp32_446, 
conv_ctest_bwddata_fp32_447, conv_ctest_bwddata_fp32_448, 
conv_ctest_bwddata_fp32_449, conv_ctest_bwddata_fp32_450, 
conv_ctest_bwddata_fp32_451, conv_ctest_bwddata_fp32_452, 
conv_ctest_bwddata_fp32_453, conv_ctest_bwddata_fp32_454, 
conv_ctest_bwddata_fp32_455, conv_ctest_bwddata_fp32_456, 
conv_ctest_bwddata_fp32_457, conv_ctest_bwddata_fp32_458, 
conv_ctest_bwddata_fp32_459, conv_ctest_bwddata_fp32_460, 
conv_ctest_bwddata_fp32_461, conv_ctest_bwddata_fp32_462, 
conv_ctest_bwddata_fp32_463, conv_ctest_bwddata_fp32_464, 
conv_ctest_bwddata_fp32_465, conv_ctest_bwddata_fp32_466, 
conv_ctest_bwddata_fp32_467, conv_ctest_bwddata_fp32_468, 
conv_ctest_bwddata_fp32_469, conv_ctest_bwddata_fp32_470, 
conv_ctest_bwddata_fp32_471, conv_ctest_bwddata_fp32_472, 
conv_ctest_bwddata_fp32_473, conv_ctest_bwddata_fp32_474, 
conv_ctest_bwddata_fp32_475, conv_ctest_bwddata_fp32_476, 
conv_ctest_bwddata_fp32_477, conv_ctest_bwddata_fp32_478, 
conv_ctest_bwddata_fp32_479, conv_ctest_bwddata_fp32_480, 
conv_ctest_bwddata_fp32_481, conv_ctest_bwddata_fp32_482, 
conv_ctest_bwddata_fp32_483, conv_ctest_bwddata_fp32_484, 
conv_ctest_bwddata_fp32_485, conv_ctest_bwddata_fp32_486, 
conv_ctest_bwddata_fp32_487, conv_ctest_bwddata_fp32_488, 
conv_ctest_bwddata_fp32_489, conv_ctest_bwddata_fp32_490, 
conv_ctest_bwddata_fp32_491, conv_ctest_bwddata_fp32_492, 
conv_ctest_bwddata_fp32_493, conv_ctest_bwddata_fp32_494, 
conv_ctest_bwddata_fp32_495, conv_ctest_bwddata_fp32_496, 
conv_ctest_bwddata_fp32_497, conv_ctest_bwddata_fp32_498, 
conv_ctest_bwddata_fp32_499, conv_ctest_bwddata_fp32_500, 
conv_ctest_bwddata_fp32_501, conv_ctest_bwddata_fp32_502, 
conv_ctest_bwddata_fp32_503, conv_ctest_bwddata_fp32_504, 
conv_ctest_bwddata_fp32_505, conv_ctest_bwddata_fp32_506, 
conv_ctest_bwddata_fp32_507, conv_ctest_bwddata_fp32_508, 
conv_ctest_bwddata_fp32_509, conv_ctest_bwddata_fp32_510, 
conv_ctest_bwddata_fp32_511, conv_ctest_bwddata_fp32_512, 
conv_ctest_bwddata_fp32_513, conv_ctest_bwddata_fp32_514, 
conv_ctest_bwddata_fp32_515, conv_ctest_bwddata_fp32_516, 
conv_ctest_bwddata_fp32_517, conv_ctest_bwddata_fp32_518, 
conv_ctest_bwddata_fp32_519, conv_ctest_bwddata_fp32_520, 
conv_ctest_bwddata_fp32_521, conv_ctest_bwddata_fp32_522, 
conv_ctest_bwddata_fp32_523, conv_ctest_bwddata_fp32_524, 
conv_ctest_bwddata_fp32_525, conv_ctest_bwddata_fp32_526, 
conv_ctest_bwddata_fp32_527, conv_ctest_bwddata_fp32_528, 
conv_ctest_bwddata_fp32_529, conv_ctest_bwddata_fp32_530, 
conv_ctest_bwddata_fp32_531, conv_ctest_bwddata_fp32_532, 
conv_ctest_bwddata_fp32_533, conv_ctest_bwddata_fp32_534, 
conv_ctest_bwddata_fp32_535, conv_ctest_bwddata_fp32_536, 
conv_ctest_bwddata_fp32_537, conv_ctest_bwddata_fp32_538, 
conv_ctest_bwddata_fp32_539, conv_ctest_bwddata_fp32_540, 
conv_ctest_bwddata_fp32_541, conv_ctest_bwddata_fp32_542, 
conv_ctest_bwddata_fp32_543, conv_ctest_bwddata_fp32_544, 
conv_ctest_bwddata_fp32_545, conv_ctest_bwddata_fp32_546, 
conv_ctest_bwddata_fp32_547, conv_ctest_bwddata_fp32_548, 
conv_ctest_bwddata_fp32_549, conv_ctest_bwddata_fp32_550, 
conv_ctest_bwddata_fp32_551, conv_ctest_bwddata_fp32_552, 
conv_ctest_bwddata_fp32_553, conv_ctest_bwddata_fp32_554, 
conv_ctest_bwddata_fp32_555, conv_ctest_bwddata_fp32_556, 
conv_ctest_bwddata_fp32_557, conv_ctest_bwddata_fp32_558, 
conv_ctest_bwddata_fp32_559, conv_ctest_bwddata_fp32_560, 
conv_ctest_bwddata_fp32_561, conv_ctest_bwddata_fp32_562, 
conv_ctest_bwddata_fp32_563, conv_ctest_bwddata_fp32_564, 
conv_ctest_bwddata_fp32_565, conv_ctest_bwddata_fp32_566, 
conv_ctest_bwddata_fp32_567, conv_ctest_bwddata_fp32_568, 
conv_ctest_bwddata_fp32_569, conv_ctest_bwddata_fp32_570, 
conv_ctest_bwddata_fp32_571, conv_ctest_bwddata_fp32_572, 
conv_ctest_bwddata_fp32_573, conv_ctest_bwddata_fp32_574, 
conv_ctest_bwddata_fp32_575, conv_ctest_bwddata_fp32_576, 
conv_ctest_bwddata_fp32_577, conv_ctest_bwddata_fp32_578, 
conv_ctest_bwddata_fp32_579, conv_ctest_bwddata_fp32_580, 
conv_ctest_bwddata_fp32_581, conv_ctest_bwddata_fp32_582, 
conv_ctest_bwddata_fp32_583, conv_ctest_bwddata_fp32_584, 
conv_ctest_bwddata_fp32_585, conv_ctest_bwddata_fp32_586, 
conv_ctest_bwddata_fp32_587, conv_ctest_bwddata_fp32_588, 
conv_ctest_bwddata_fp32_589, conv_ctest_bwddata_fp32_590, 
conv_ctest_bwddata_fp32_591, conv_ctest_bwddata_fp32_592, 
conv_ctest_bwddata_fp32_593, conv_ctest_bwddata_fp32_594, 
conv_ctest_bwddata_fp32_595, conv_ctest_bwddata_fp32_596, 
conv_ctest_bwddata_fp32_597, conv_ctest_bwddata_fp32_598, 
conv_ctest_bwddata_fp32_599, conv_ctest_bwddata_fp32_600, 
conv_ctest_bwddata_fp32_601, conv_ctest_bwddata_fp32_602, 
conv_ctest_bwddata_fp32_603, conv_ctest_bwddata_fp32_604, 
conv_ctest_bwddata_fp32_605, conv_ctest_bwddata_fp32_606, 
conv_ctest_bwddata_fp32_607, 
};
 
gemm_tuple conv_ctest_bwddata_fp16_001 {{10000, 363, 1, 10000, 363, 10000}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_002 {{100, 1008, 1, 100, 1008, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_003 {{100, 1152, 1, 100, 1152, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_004 {{100, 128, 1, 100, 128, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_005 {{100, 1296, 1, 100, 1296, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_006 {{100, 1440, 1, 100, 1440, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_007 {{100, 1600, 1, 100, 1600, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_008 {{100, 1728, 1, 100, 1728, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_009 {{100, 192, 1, 100, 192, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_010 {{100, 2304, 1, 100, 2304, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_011 {{100, 2400, 1, 100, 2400, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_012 {{100, 256, 1, 100, 256, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_013 {{100, 400, 1, 100, 400, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_014 {{100, 4608, 1, 100, 4608, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_015 {{100, 480, 1, 100, 480, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_016 {{100, 4, 1, 100, 4, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_017 {{100, 512, 1, 100, 512, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_018 {{100, 528, 1, 100, 528, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_019 {{100, 576, 1, 100, 576, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_020 {{100, 600, 1, 100, 600, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_021 {{100, 608, 1, 100, 608, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_022 {{100, 64, 1, 100, 64, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_023 {{100, 800, 1, 100, 800, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_024 {{100, 864, 1, 100, 864, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_025 {{100, 9216, 1, 100, 9216, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_026 {{100, 9, 1, 100, 9, 100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_027 {{1024, 128, 1, 1024, 128, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_028 {{1024, 147, 1, 1024, 147, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_029 {{1024, 192, 1, 1024, 192, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_030 {{1024, 256, 1, 1024, 256, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_031 {{1024, 27, 1, 1024, 27, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_032 {{1024, 320, 1, 1024, 320, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_033 {{1024, 363, 1, 1024, 363, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_034 {{1024, 512, 1, 1024, 512, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_035 {{1024, 64, 1, 1024, 64, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_036 {{1024, 75, 1, 1024, 75, 1024}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_037 {{10404, 363, 1, 10404, 363, 10404}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_038 {{10609, 147, 1, 10609, 147, 10609}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_039 {{10816, 147, 1, 10816, 147, 10816}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_040 {{10816, 1600, 1, 10816, 1600, 10816}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_041 {{11025, 147, 1, 11025, 147, 11025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_042 {{11236, 147, 1, 11236, 147, 11236}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_043 {{11449, 147, 1, 11449, 147, 11449}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_044 {{11449, 363, 1, 11449, 363, 11449}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_045 {{11449, 75, 1, 11449, 75, 11449}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_046 {{1156, 27, 1, 1156, 27, 1156}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_047 {{11664, 147, 1, 11664, 147, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_048 {{11664, 1600, 1, 11664, 1600, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_049 {{11664, 363, 1, 11664, 363, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_050 {{11664, 576, 1, 11664, 576, 11664}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_051 {{11881, 147, 1, 11881, 147, 11881}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_052 {{11881, 363, 1, 11881, 363, 11881}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_053 {{11881, 75, 1, 11881, 75, 11881}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_054 {{12100, 147, 1, 12100, 147, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_055 {{12100, 1600, 1, 12100, 1600, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_056 {{12100, 27, 1, 12100, 27, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_057 {{12100, 363, 1, 12100, 363, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_058 {{12100, 576, 1, 12100, 576, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_059 {{12100, 75, 1, 12100, 75, 12100}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_060 {{121, 1024, 1, 121, 1024, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_061 {{121, 1056, 1, 121, 1056, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_062 {{121, 192, 1, 121, 192, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_063 {{121, 2304, 1, 121, 2304, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_064 {{121, 3456, 1, 121, 3456, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_065 {{121, 363, 1, 121, 363, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_066 {{121, 4, 1, 121, 4, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_067 {{121, 512, 1, 121, 512, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_068 {{121, 75, 1, 121, 75, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_069 {{121, 832, 1, 121, 832, 121}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_070 {{12321, 147, 1, 12321, 147, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_071 {{12321, 27, 1, 12321, 27, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_072 {{12321, 363, 1, 12321, 363, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_073 {{12321, 75, 1, 12321, 75, 12321}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_074 {{12544, 147, 1, 12544, 147, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_075 {{12544, 1600, 1, 12544, 1600, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_076 {{12544, 27, 1, 12544, 27, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_077 {{12544, 363, 1, 12544, 363, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_078 {{12544, 576, 1, 12544, 576, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_079 {{12544, 75, 1, 12544, 75, 12544}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_080 {{12769, 147, 1, 12769, 147, 12769}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_081 {{12769, 27, 1, 12769, 27, 12769}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_082 {{12769, 75, 1, 12769, 75, 12769}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_083 {{12996, 147, 1, 12996, 147, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_084 {{12996, 27, 1, 12996, 27, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_085 {{12996, 363, 1, 12996, 363, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_086 {{12996, 576, 1, 12996, 576, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_087 {{12996, 64, 1, 12996, 64, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_088 {{12996, 75, 1, 12996, 75, 12996}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_089 {{13225, 27, 1, 13225, 27, 13225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_090 {{13225, 75, 1, 13225, 75, 13225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_091 {{13456, 147, 1, 13456, 147, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_092 {{13456, 27, 1, 13456, 27, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_093 {{13456, 363, 1, 13456, 363, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_094 {{13456, 64, 1, 13456, 64, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_095 {{13456, 75, 1, 13456, 75, 13456}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_096 {{13689, 75, 1, 13689, 75, 13689}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_097 {{13924, 27, 1, 13924, 27, 13924}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_098 {{144, 1008, 1, 144, 1008, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_099 {{144, 1152, 1, 144, 1152, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_100 {{144, 1296, 1, 144, 1296, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_101 {{144, 1440, 1, 144, 1440, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_102 {{144, 1600, 1, 144, 1600, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_103 {{144, 1728, 1, 144, 1728, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_104 {{144, 2304, 1, 144, 2304, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_105 {{144, 2400, 1, 144, 2400, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_106 {{144, 363, 1, 144, 363, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_107 {{144, 400, 1, 144, 400, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_108 {{144, 4608, 1, 144, 4608, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_109 {{144, 4, 1, 144, 4, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_110 {{144, 576, 1, 144, 576, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_111 {{144, 600, 1, 144, 600, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_112 {{144, 800, 1, 144, 800, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_113 {{144, 864, 1, 144, 864, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_114 {{144, 9216, 1, 144, 9216, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_115 {{144, 9, 1, 144, 9, 144}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_116 {{169, 1152, 1, 169, 1152, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_117 {{169, 147, 1, 169, 147, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_118 {{169, 1600, 1, 169, 1600, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_119 {{169, 1728, 1, 169, 1728, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_120 {{169, 2048, 1, 169, 2048, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_121 {{169, 2304, 1, 169, 2304, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_122 {{169, 2400, 1, 169, 2400, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_123 {{169, 3456, 1, 169, 3456, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_124 {{169, 400, 1, 169, 400, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_125 {{169, 4608, 1, 169, 4608, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_126 {{169, 4, 1, 169, 4, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_127 {{169, 576, 1, 169, 576, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_128 {{169, 800, 1, 169, 800, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_129 {{169, 864, 1, 169, 864, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_130 {{169, 9, 1, 169, 9, 169}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_131 {{16, 1024, 1, 16, 1024, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_132 {{16, 1056, 1, 16, 1056, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_133 {{16, 1200, 1, 16, 1200, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_134 {{16, 1440, 1, 16, 1440, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_135 {{16, 1728, 1, 16, 1728, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_136 {{16, 192, 1, 16, 192, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_137 {{16, 2016, 1, 16, 2016, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_138 {{16, 2304, 1, 16, 2304, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_139 {{16, 4608, 1, 16, 4608, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_140 {{16, 4, 1, 16, 4, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_141 {{16, 512, 1, 16, 512, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_142 {{16, 800, 1, 16, 800, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_143 {{16, 832, 1, 16, 832, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_144 {{16, 9216, 1, 16, 9216, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_145 {{16, 9, 1, 16, 9, 16}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_146 {{1860, 4608, 1, 1860, 4608, 1860}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_147 {{1953, 4608, 1, 1953, 4608, 1953}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_148 {{196, 1008, 1, 196, 1008, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_149 {{196, 1024, 1, 196, 1024, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_150 {{196, 1152, 1, 196, 1152, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_151 {{196, 128, 1, 196, 128, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_152 {{196, 1296, 1, 196, 1296, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_153 {{196, 1440, 1, 196, 1440, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_154 {{196, 147, 1, 196, 147, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_155 {{196, 1600, 1, 196, 1600, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_156 {{196, 1728, 1, 196, 1728, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_157 {{196, 192, 1, 196, 192, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_158 {{196, 2304, 1, 196, 2304, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_159 {{196, 2400, 1, 196, 2400, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_160 {{196, 256, 1, 196, 256, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_161 {{196, 27, 1, 196, 27, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_162 {{196, 320, 1, 196, 320, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_163 {{196, 363, 1, 196, 363, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_164 {{196, 400, 1, 196, 400, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_165 {{196, 4608, 1, 196, 4608, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_166 {{196, 4, 1, 196, 4, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_167 {{196, 512, 1, 196, 512, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_168 {{196, 576, 1, 196, 576, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_169 {{196, 600, 1, 196, 600, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_170 {{196, 64, 1, 196, 64, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_171 {{196, 75, 1, 196, 75, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_172 {{196, 800, 1, 196, 800, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_173 {{196, 864, 1, 196, 864, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_174 {{196, 9216, 1, 196, 9216, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_175 {{196, 9, 1, 196, 9, 196}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_176 {{1, 1200, 1, 1, 1200, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_177 {{1, 363, 1, 1, 363, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_178 {{1, 4608, 1, 1, 4608, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_179 {{1, 4, 1, 1, 4, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_180 {{1, 800, 1, 1, 800, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_181 {{1, 9, 1, 1, 9, 1}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_182 {{2048, 4608, 1, 2048, 4608, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_183 {{2048, 480, 1, 2048, 480, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_184 {{2048, 512, 1, 2048, 512, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_185 {{2048, 528, 1, 2048, 528, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_186 {{2048, 832, 1, 2048, 832, 2048}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_187 {{2145, 480, 1, 2145, 480, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_188 {{2145, 512, 1, 2145, 512, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_189 {{2145, 528, 1, 2145, 528, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_190 {{2145, 832, 1, 2145, 832, 2145}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_191 {{2244, 4608, 1, 2244, 4608, 2244}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_192 {{225, 128, 1, 225, 128, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_193 {{225, 1600, 1, 225, 1600, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_194 {{225, 192, 1, 225, 192, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_195 {{225, 2048, 1, 225, 2048, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_196 {{225, 2304, 1, 225, 2304, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_197 {{225, 2400, 1, 225, 2400, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_198 {{225, 256, 1, 225, 256, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_199 {{225, 27, 1, 225, 27, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_200 {{225, 320, 1, 225, 320, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_201 {{225, 3456, 1, 225, 3456, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_202 {{225, 400, 1, 225, 400, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_203 {{225, 4, 1, 225, 4, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_204 {{225, 512, 1, 225, 512, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_205 {{225, 64, 1, 225, 64, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_206 {{225, 75, 1, 225, 75, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_207 {{225, 800, 1, 225, 800, 225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_208 {{2304, 1600, 1, 2304, 1600, 2304}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_209 {{2345, 480, 1, 2345, 480, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_210 {{2345, 512, 1, 2345, 512, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_211 {{2345, 528, 1, 2345, 528, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_212 {{2345, 832, 1, 2345, 832, 2345}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_213 {{256, 1008, 1, 256, 1008, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_214 {{256, 1024, 1, 256, 1024, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_215 {{256, 1152, 1, 256, 1152, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_216 {{256, 128, 1, 256, 128, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_217 {{256, 1296, 1, 256, 1296, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_218 {{256, 1440, 1, 256, 1440, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_219 {{256, 147, 1, 256, 147, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_220 {{256, 1728, 1, 256, 1728, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_221 {{256, 192, 1, 256, 192, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_222 {{256, 2304, 1, 256, 2304, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_223 {{256, 256, 1, 256, 256, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_224 {{256, 27, 1, 256, 27, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_225 {{256, 363, 1, 256, 363, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_226 {{256, 4608, 1, 256, 4608, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_227 {{256, 480, 1, 256, 480, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_228 {{256, 4, 1, 256, 4, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_229 {{256, 512, 1, 256, 512, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_230 {{256, 528, 1, 256, 528, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_231 {{256, 576, 1, 256, 576, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_232 {{256, 608, 1, 256, 608, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_233 {{256, 64, 1, 256, 64, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_234 {{256, 75, 1, 256, 75, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_235 {{256, 800, 1, 256, 800, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_236 {{256, 864, 1, 256, 864, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_237 {{256, 9, 1, 256, 9, 256}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_238 {{25, 1008, 1, 25, 1008, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_239 {{25, 1024, 1, 25, 1024, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_240 {{25, 1056, 1, 25, 1056, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_241 {{25, 1152, 1, 25, 1152, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_242 {{25, 1200, 1, 25, 1200, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_243 {{25, 1296, 1, 25, 1296, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_244 {{25, 1440, 1, 25, 1440, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_245 {{25, 1600, 1, 25, 1600, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_246 {{25, 1728, 1, 25, 1728, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_247 {{25, 192, 1, 25, 192, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_248 {{25, 2016, 1, 25, 2016, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_249 {{25, 2304, 1, 25, 2304, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_250 {{25, 2400, 1, 25, 2400, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_251 {{25, 3456, 1, 25, 3456, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_252 {{25, 400, 1, 25, 400, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_253 {{25, 4608, 1, 25, 4608, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_254 {{25, 4, 1, 25, 4, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_255 {{25, 512, 1, 25, 512, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_256 {{25, 528, 1, 25, 528, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_257 {{25, 576, 1, 25, 576, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_258 {{25, 600, 1, 25, 600, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_259 {{25, 608, 1, 25, 608, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_260 {{25, 800, 1, 25, 800, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_261 {{25, 832, 1, 25, 832, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_262 {{25, 864, 1, 25, 864, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_263 {{25, 9216, 1, 25, 9216, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_264 {{25, 9, 1, 25, 9, 25}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_265 {{2601, 1600, 1, 2601, 1600, 2601}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_266 {{2704, 1152, 1, 2704, 1152, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_267 {{2704, 1600, 1, 2704, 1600, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_268 {{2704, 2304, 1, 2704, 2304, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_269 {{2704, 576, 1, 2704, 576, 2704}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_270 {{289, 128, 1, 289, 128, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_271 {{289, 192, 1, 289, 192, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_272 {{289, 256, 1, 289, 256, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_273 {{289, 320, 1, 289, 320, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_274 {{289, 4, 1, 289, 4, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_275 {{289, 512, 1, 289, 512, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_276 {{289, 64, 1, 289, 64, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_277 {{289, 75, 1, 289, 75, 289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_278 {{2916, 1152, 1, 2916, 1152, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_279 {{2916, 1600, 1, 2916, 1600, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_280 {{2916, 2304, 1, 2916, 2304, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_281 {{2916, 576, 1, 2916, 576, 2916}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_282 {{3025, 1600, 1, 3025, 1600, 3025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_283 {{3025, 576, 1, 3025, 576, 3025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_284 {{3136, 1152, 1, 3136, 1152, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_285 {{3136, 1600, 1, 3136, 1600, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_286 {{3136, 2304, 1, 3136, 2304, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_287 {{3136, 576, 1, 3136, 576, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_288 {{3136, 64, 1, 3136, 64, 3136}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_289 {{3249, 1600, 1, 3249, 1600, 3249}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_290 {{3249, 64, 1, 3249, 64, 3249}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_291 {{324, 128, 1, 324, 128, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_292 {{324, 192, 1, 324, 192, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_293 {{324, 256, 1, 324, 256, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_294 {{324, 27, 1, 324, 27, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_295 {{324, 480, 1, 324, 480, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_296 {{324, 512, 1, 324, 512, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_297 {{324, 528, 1, 324, 528, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_298 {{324, 576, 1, 324, 576, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_299 {{324, 608, 1, 324, 608, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_300 {{324, 64, 1, 324, 64, 324}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_301 {{33540, 480, 1, 33540, 480, 33540}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_302 {{3364, 1152, 1, 3364, 1152, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_303 {{3364, 128, 1, 3364, 128, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_304 {{3364, 2304, 1, 3364, 2304, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_305 {{3364, 256, 1, 3364, 256, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_306 {{3364, 576, 1, 3364, 576, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_307 {{3364, 64, 1, 3364, 64, 3364}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_308 {{34320, 480, 1, 34320, 480, 34320}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_309 {{3481, 64, 1, 3481, 64, 3481}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_310 {{3600, 128, 1, 3600, 128, 3600}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_311 {{3600, 256, 1, 3600, 256, 3600}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_312 {{3600, 64, 1, 3600, 64, 3600}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_313 {{361, 1600, 1, 361, 1600, 361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_314 {{361, 2400, 1, 361, 2400, 361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_315 {{36, 1008, 1, 36, 1008, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_316 {{36, 1024, 1, 36, 1024, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_317 {{36, 1152, 1, 36, 1152, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_318 {{36, 1296, 1, 36, 1296, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_319 {{36, 1440, 1, 36, 1440, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_320 {{36, 1600, 1, 36, 1600, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_321 {{36, 1728, 1, 36, 1728, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_322 {{36, 2016, 1, 36, 2016, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_323 {{36, 2048, 1, 36, 2048, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_324 {{36, 2304, 1, 36, 2304, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_325 {{36, 2400, 1, 36, 2400, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_326 {{36, 256, 1, 36, 256, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_327 {{36, 3456, 1, 36, 3456, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_328 {{36, 400, 1, 36, 400, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_329 {{36, 4608, 1, 36, 4608, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_330 {{36, 4, 1, 36, 4, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_331 {{36, 512, 1, 36, 512, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_332 {{36, 528, 1, 36, 528, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_333 {{36, 576, 1, 36, 576, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_334 {{36, 600, 1, 36, 600, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_335 {{36, 608, 1, 36, 608, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_336 {{36, 800, 1, 36, 800, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_337 {{36, 864, 1, 36, 864, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_338 {{36, 9216, 1, 36, 9216, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_339 {{36, 9, 1, 36, 9, 36}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_340 {{400, 147, 1, 400, 147, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_341 {{400, 1600, 1, 400, 1600, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_342 {{400, 2400, 1, 400, 2400, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_343 {{400, 400, 1, 400, 400, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_344 {{400, 800, 1, 400, 800, 400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_345 {{41616, 363, 1, 41616, 363, 41616}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_346 {{42849, 363, 1, 42849, 363, 42849}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_347 {{44521, 363, 1, 44521, 363, 44521}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_348 {{44944, 147, 1, 44944, 147, 44944}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_349 {{45796, 363, 1, 45796, 363, 45796}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_350 {{46225, 147, 1, 46225, 147, 46225}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_351 {{46656, 363, 1, 46656, 363, 46656}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_352 {{46656, 75, 1, 46656, 75, 46656}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_353 {{47089, 363, 1, 47089, 363, 47089}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_354 {{47524, 147, 1, 47524, 147, 47524}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_355 {{47524, 363, 1, 47524, 363, 47524}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_356 {{47961, 147, 1, 47961, 147, 47961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_357 {{47961, 363, 1, 47961, 363, 47961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_358 {{47961, 75, 1, 47961, 75, 47961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_359 {{48400, 147, 1, 48400, 147, 48400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_360 {{48400, 27, 1, 48400, 27, 48400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_361 {{48400, 75, 1, 48400, 75, 48400}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_362 {{484, 363, 1, 484, 363, 484}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_363 {{48841, 147, 1, 48841, 147, 48841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_364 {{48841, 363, 1, 48841, 363, 48841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_365 {{49284, 147, 1, 49284, 147, 49284}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_366 {{49284, 27, 1, 49284, 27, 49284}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_367 {{49284, 75, 1, 49284, 75, 49284}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_368 {{49729, 147, 1, 49729, 147, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_369 {{49729, 27, 1, 49729, 27, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_370 {{49729, 363, 1, 49729, 363, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_371 {{49729, 75, 1, 49729, 75, 49729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_372 {{49, 1008, 1, 49, 1008, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_373 {{49, 1024, 1, 49, 1024, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_374 {{49, 1056, 1, 49, 1056, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_375 {{49, 1152, 1, 49, 1152, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_376 {{49, 1200, 1, 49, 1200, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_377 {{49, 128, 1, 49, 128, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_378 {{49, 1296, 1, 49, 1296, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_379 {{49, 1440, 1, 49, 1440, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_380 {{49, 147, 1, 49, 147, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_381 {{49, 1600, 1, 49, 1600, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_382 {{49, 1728, 1, 49, 1728, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_383 {{49, 192, 1, 49, 192, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_384 {{49, 2016, 1, 49, 2016, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_385 {{49, 2048, 1, 49, 2048, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_386 {{49, 2304, 1, 49, 2304, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_387 {{49, 2400, 1, 49, 2400, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_388 {{49, 256, 1, 49, 256, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_389 {{49, 3456, 1, 49, 3456, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_390 {{49, 400, 1, 49, 400, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_391 {{49, 4608, 1, 49, 4608, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_392 {{49, 480, 1, 49, 480, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_393 {{49, 4, 1, 49, 4, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_394 {{49, 512, 1, 49, 512, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_395 {{49, 528, 1, 49, 528, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_396 {{49, 576, 1, 49, 576, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_397 {{49, 600, 1, 49, 600, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_398 {{49, 608, 1, 49, 608, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_399 {{49, 64, 1, 49, 64, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_400 {{49, 800, 1, 49, 800, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_401 {{49, 832, 1, 49, 832, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_402 {{49, 864, 1, 49, 864, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_403 {{49, 9216, 1, 49, 9216, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_404 {{49, 9, 1, 49, 9, 49}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_405 {{4, 1200, 1, 4, 1200, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_406 {{4, 1440, 1, 4, 1440, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_407 {{4, 1600, 1, 4, 1600, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_408 {{4, 1728, 1, 4, 1728, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_409 {{4, 2016, 1, 4, 2016, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_410 {{4, 2400, 1, 4, 2400, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_411 {{4, 363, 1, 4, 363, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_412 {{4, 400, 1, 4, 400, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_413 {{4, 4608, 1, 4, 4608, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_414 {{4, 4, 1, 4, 4, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_415 {{4, 512, 1, 4, 512, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_416 {{4, 528, 1, 4, 528, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_417 {{4, 576, 1, 4, 576, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_418 {{4, 600, 1, 4, 600, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_419 {{4, 608, 1, 4, 608, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_420 {{4, 800, 1, 4, 800, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_421 {{4, 9216, 1, 4, 9216, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_422 {{4, 9, 1, 4, 9, 4}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_423 {{50176, 147, 1, 50176, 147, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_424 {{50176, 27, 1, 50176, 27, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_425 {{50176, 363, 1, 50176, 363, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_426 {{50176, 75, 1, 50176, 75, 50176}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_427 {{50625, 147, 1, 50625, 147, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_428 {{50625, 27, 1, 50625, 27, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_429 {{50625, 363, 1, 50625, 363, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_430 {{50625, 75, 1, 50625, 75, 50625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_431 {{51076, 27, 1, 51076, 27, 51076}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_432 {{51529, 147, 1, 51529, 147, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_433 {{51529, 27, 1, 51529, 27, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_434 {{51529, 363, 1, 51529, 363, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_435 {{51529, 75, 1, 51529, 75, 51529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_436 {{52441, 147, 1, 52441, 147, 52441}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_437 {{52441, 27, 1, 52441, 27, 52441}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_438 {{52441, 75, 1, 52441, 75, 52441}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_439 {{529, 1600, 1, 529, 1600, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_440 {{529, 2400, 1, 529, 2400, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_441 {{529, 576, 1, 529, 576, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_442 {{529, 864, 1, 529, 864, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_443 {{529, 9, 1, 529, 9, 529}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_444 {{53361, 147, 1, 53361, 147, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_445 {{53361, 27, 1, 53361, 27, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_446 {{53361, 363, 1, 53361, 363, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_447 {{53361, 75, 1, 53361, 75, 53361}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_448 {{54289, 27, 1, 54289, 27, 54289}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_449 {{576, 1152, 1, 576, 1152, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_450 {{576, 1600, 1, 576, 1600, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_451 {{576, 1728, 1, 576, 1728, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_452 {{576, 2304, 1, 576, 2304, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_453 {{576, 2400, 1, 576, 2400, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_454 {{576, 363, 1, 576, 363, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_455 {{576, 400, 1, 576, 400, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_456 {{576, 4608, 1, 576, 4608, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_457 {{576, 576, 1, 576, 576, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_458 {{576, 75, 1, 576, 75, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_459 {{576, 800, 1, 576, 800, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_460 {{576, 864, 1, 576, 864, 576}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_461 {{625, 1600, 1, 625, 1600, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_462 {{625, 2400, 1, 625, 2400, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_463 {{625, 4, 1, 625, 4, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_464 {{625, 576, 1, 625, 576, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_465 {{625, 864, 1, 625, 864, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_466 {{625, 9, 1, 625, 9, 625}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_467 {{64, 128, 1, 64, 128, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_468 {{64, 147, 1, 64, 147, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_469 {{64, 1600, 1, 64, 1600, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_470 {{64, 192, 1, 64, 192, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_471 {{64, 2304, 1, 64, 2304, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_472 {{64, 2400, 1, 64, 2400, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_473 {{64, 256, 1, 64, 256, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_474 {{64, 400, 1, 64, 400, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_475 {{64, 4608, 1, 64, 4608, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_476 {{64, 480, 1, 64, 480, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_477 {{64, 4, 1, 64, 4, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_478 {{64, 512, 1, 64, 512, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_479 {{64, 528, 1, 64, 528, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_480 {{64, 576, 1, 64, 576, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_481 {{64, 600, 1, 64, 600, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_482 {{64, 608, 1, 64, 608, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_483 {{64, 64, 1, 64, 64, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_484 {{64, 800, 1, 64, 800, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_485 {{64, 9216, 1, 64, 9216, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_486 {{64, 9, 1, 64, 9, 64}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_487 {{676, 1152, 1, 676, 1152, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_488 {{676, 147, 1, 676, 147, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_489 {{676, 1600, 1, 676, 1600, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_490 {{676, 1728, 1, 676, 1728, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_491 {{676, 2304, 1, 676, 2304, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_492 {{676, 2400, 1, 676, 2400, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_493 {{676, 363, 1, 676, 363, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_494 {{676, 400, 1, 676, 400, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_495 {{676, 4608, 1, 676, 4608, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_496 {{676, 4, 1, 676, 4, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_497 {{676, 576, 1, 676, 576, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_498 {{676, 800, 1, 676, 800, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_499 {{676, 864, 1, 676, 864, 676}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_500 {{729, 1152, 1, 729, 1152, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_501 {{729, 1600, 1, 729, 1600, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_502 {{729, 2304, 1, 729, 2304, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_503 {{729, 2400, 1, 729, 2400, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_504 {{729, 4, 1, 729, 4, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_505 {{729, 576, 1, 729, 576, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_506 {{729, 864, 1, 729, 864, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_507 {{729, 9, 1, 729, 9, 729}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_508 {{7440, 4608, 1, 7440, 4608, 7440}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_509 {{7812, 4608, 1, 7812, 4608, 7812}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_510 {{784, 1152, 1, 784, 1152, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_511 {{784, 128, 1, 784, 128, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_512 {{784, 147, 1, 784, 147, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_513 {{784, 1600, 1, 784, 1600, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_514 {{784, 1728, 1, 784, 1728, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_515 {{784, 2304, 1, 784, 2304, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_516 {{784, 2400, 1, 784, 2400, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_517 {{784, 256, 1, 784, 256, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_518 {{784, 27, 1, 784, 27, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_519 {{784, 400, 1, 784, 400, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_520 {{784, 4608, 1, 784, 4608, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_521 {{784, 4, 1, 784, 4, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_522 {{784, 576, 1, 784, 576, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_523 {{784, 64, 1, 784, 64, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_524 {{784, 75, 1, 784, 75, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_525 {{784, 800, 1, 784, 800, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_526 {{784, 864, 1, 784, 864, 784}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_527 {{8192, 4608, 1, 8192, 4608, 8192}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_528 {{8192, 480, 1, 8192, 480, 8192}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_529 {{81, 1008, 1, 81, 1008, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_530 {{81, 1024, 1, 81, 1024, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_531 {{81, 1056, 1, 81, 1056, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_532 {{81, 1152, 1, 81, 1152, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_533 {{81, 1296, 1, 81, 1296, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_534 {{81, 1440, 1, 81, 1440, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_535 {{81, 1600, 1, 81, 1600, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_536 {{81, 1728, 1, 81, 1728, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_537 {{81, 192, 1, 81, 192, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_538 {{81, 2016, 1, 81, 2016, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_539 {{81, 2048, 1, 81, 2048, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_540 {{81, 2304, 1, 81, 2304, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_541 {{81, 2400, 1, 81, 2400, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_542 {{81, 256, 1, 81, 256, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_543 {{81, 3456, 1, 81, 3456, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_544 {{81, 400, 1, 81, 400, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_545 {{81, 4608, 1, 81, 4608, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_546 {{81, 4, 1, 81, 4, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_547 {{81, 512, 1, 81, 512, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_548 {{81, 576, 1, 81, 576, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_549 {{81, 800, 1, 81, 800, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_550 {{81, 832, 1, 81, 832, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_551 {{81, 864, 1, 81, 864, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_552 {{81, 9216, 1, 81, 9216, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_553 {{81, 9, 1, 81, 9, 81}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_554 {{8385, 480, 1, 8385, 480, 8385}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_555 {{841, 128, 1, 841, 128, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_556 {{841, 1600, 1, 841, 1600, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_557 {{841, 256, 1, 841, 256, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_558 {{841, 576, 1, 841, 576, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_559 {{841, 64, 1, 841, 64, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_560 {{841, 864, 1, 841, 864, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_561 {{841, 9, 1, 841, 9, 841}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_562 {{8580, 4608, 1, 8580, 4608, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_563 {{8580, 480, 1, 8580, 480, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_564 {{8580, 512, 1, 8580, 512, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_565 {{8580, 528, 1, 8580, 528, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_566 {{8580, 832, 1, 8580, 832, 8580}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_567 {{8777, 480, 1, 8777, 480, 8777}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_568 {{8976, 480, 1, 8976, 480, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_569 {{8976, 512, 1, 8976, 512, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_570 {{8976, 528, 1, 8976, 528, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_571 {{8976, 832, 1, 8976, 832, 8976}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_572 {{900, 1152, 1, 900, 1152, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_573 {{900, 128, 1, 900, 128, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_574 {{900, 147, 1, 900, 147, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_575 {{900, 1728, 1, 900, 1728, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_576 {{900, 192, 1, 900, 192, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_577 {{900, 2304, 1, 900, 2304, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_578 {{900, 256, 1, 900, 256, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_579 {{900, 27, 1, 900, 27, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_580 {{900, 320, 1, 900, 320, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_581 {{900, 4608, 1, 900, 4608, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_582 {{900, 4, 1, 900, 4, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_583 {{900, 512, 1, 900, 512, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_584 {{900, 576, 1, 900, 576, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_585 {{900, 64, 1, 900, 64, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_586 {{900, 75, 1, 900, 75, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_587 {{900, 864, 1, 900, 864, 900}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_588 {{9025, 363, 1, 9025, 363, 9025}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_589 {{9409, 363, 1, 9409, 363, 9409}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_590 {{9604, 363, 1, 9604, 363, 9604}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_591 {{961, 128, 1, 961, 128, 961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_592 {{961, 256, 1, 961, 256, 961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_593 {{961, 64, 1, 961, 64, 961}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_594 {{9801, 363, 1, 9801, 363, 9801}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_595 {{9, 1200, 1, 9, 1200, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_596 {{9, 1440, 1, 9, 1440, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_597 {{9, 1728, 1, 9, 1728, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_598 {{9, 2016, 1, 9, 2016, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_599 {{9, 4608, 1, 9, 4608, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_600 {{9, 4, 1, 9, 4, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_601 {{9, 512, 1, 9, 512, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_602 {{9, 528, 1, 9, 528, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_603 {{9, 576, 1, 9, 576, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_604 {{9, 608, 1, 9, 608, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_605 {{9, 800, 1, 9, 800, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_606 {{9, 9216, 1, 9, 9216, 9}, {1, 0}, {'N', 'T'}};
gemm_tuple conv_ctest_bwddata_fp16_607 {{9, 9, 1, 9, 9, 9}, {1, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_ctest_bwddata_fp16 = {
conv_ctest_bwddata_fp16_001, conv_ctest_bwddata_fp16_002, 
conv_ctest_bwddata_fp16_003, conv_ctest_bwddata_fp16_004, 
conv_ctest_bwddata_fp16_005, conv_ctest_bwddata_fp16_006, 
conv_ctest_bwddata_fp16_007, conv_ctest_bwddata_fp16_008, 
conv_ctest_bwddata_fp16_009, conv_ctest_bwddata_fp16_010, 
conv_ctest_bwddata_fp16_011, conv_ctest_bwddata_fp16_012, 
conv_ctest_bwddata_fp16_013, conv_ctest_bwddata_fp16_014, 
conv_ctest_bwddata_fp16_015, conv_ctest_bwddata_fp16_016, 
conv_ctest_bwddata_fp16_017, conv_ctest_bwddata_fp16_018, 
conv_ctest_bwddata_fp16_019, conv_ctest_bwddata_fp16_020, 
conv_ctest_bwddata_fp16_021, conv_ctest_bwddata_fp16_022, 
conv_ctest_bwddata_fp16_023, conv_ctest_bwddata_fp16_024, 
conv_ctest_bwddata_fp16_025, conv_ctest_bwddata_fp16_026, 
conv_ctest_bwddata_fp16_027, conv_ctest_bwddata_fp16_028, 
conv_ctest_bwddata_fp16_029, conv_ctest_bwddata_fp16_030, 
conv_ctest_bwddata_fp16_031, conv_ctest_bwddata_fp16_032, 
conv_ctest_bwddata_fp16_033, conv_ctest_bwddata_fp16_034, 
conv_ctest_bwddata_fp16_035, conv_ctest_bwddata_fp16_036, 
conv_ctest_bwddata_fp16_037, conv_ctest_bwddata_fp16_038, 
conv_ctest_bwddata_fp16_039, conv_ctest_bwddata_fp16_040, 
conv_ctest_bwddata_fp16_041, conv_ctest_bwddata_fp16_042, 
conv_ctest_bwddata_fp16_043, conv_ctest_bwddata_fp16_044, 
conv_ctest_bwddata_fp16_045, conv_ctest_bwddata_fp16_046, 
conv_ctest_bwddata_fp16_047, conv_ctest_bwddata_fp16_048, 
conv_ctest_bwddata_fp16_049, conv_ctest_bwddata_fp16_050, 
conv_ctest_bwddata_fp16_051, conv_ctest_bwddata_fp16_052, 
conv_ctest_bwddata_fp16_053, conv_ctest_bwddata_fp16_054, 
conv_ctest_bwddata_fp16_055, conv_ctest_bwddata_fp16_056, 
conv_ctest_bwddata_fp16_057, conv_ctest_bwddata_fp16_058, 
conv_ctest_bwddata_fp16_059, conv_ctest_bwddata_fp16_060, 
conv_ctest_bwddata_fp16_061, conv_ctest_bwddata_fp16_062, 
conv_ctest_bwddata_fp16_063, conv_ctest_bwddata_fp16_064, 
conv_ctest_bwddata_fp16_065, conv_ctest_bwddata_fp16_066, 
conv_ctest_bwddata_fp16_067, conv_ctest_bwddata_fp16_068, 
conv_ctest_bwddata_fp16_069, conv_ctest_bwddata_fp16_070, 
conv_ctest_bwddata_fp16_071, conv_ctest_bwddata_fp16_072, 
conv_ctest_bwddata_fp16_073, conv_ctest_bwddata_fp16_074, 
conv_ctest_bwddata_fp16_075, conv_ctest_bwddata_fp16_076, 
conv_ctest_bwddata_fp16_077, conv_ctest_bwddata_fp16_078, 
conv_ctest_bwddata_fp16_079, conv_ctest_bwddata_fp16_080, 
conv_ctest_bwddata_fp16_081, conv_ctest_bwddata_fp16_082, 
conv_ctest_bwddata_fp16_083, conv_ctest_bwddata_fp16_084, 
conv_ctest_bwddata_fp16_085, conv_ctest_bwddata_fp16_086, 
conv_ctest_bwddata_fp16_087, conv_ctest_bwddata_fp16_088, 
conv_ctest_bwddata_fp16_089, conv_ctest_bwddata_fp16_090, 
conv_ctest_bwddata_fp16_091, conv_ctest_bwddata_fp16_092, 
conv_ctest_bwddata_fp16_093, conv_ctest_bwddata_fp16_094, 
conv_ctest_bwddata_fp16_095, conv_ctest_bwddata_fp16_096, 
conv_ctest_bwddata_fp16_097, conv_ctest_bwddata_fp16_098, 
conv_ctest_bwddata_fp16_099, conv_ctest_bwddata_fp16_100, 
conv_ctest_bwddata_fp16_101, conv_ctest_bwddata_fp16_102, 
conv_ctest_bwddata_fp16_103, conv_ctest_bwddata_fp16_104, 
conv_ctest_bwddata_fp16_105, conv_ctest_bwddata_fp16_106, 
conv_ctest_bwddata_fp16_107, conv_ctest_bwddata_fp16_108, 
conv_ctest_bwddata_fp16_109, conv_ctest_bwddata_fp16_110, 
conv_ctest_bwddata_fp16_111, conv_ctest_bwddata_fp16_112, 
conv_ctest_bwddata_fp16_113, conv_ctest_bwddata_fp16_114, 
conv_ctest_bwddata_fp16_115, conv_ctest_bwddata_fp16_116, 
conv_ctest_bwddata_fp16_117, conv_ctest_bwddata_fp16_118, 
conv_ctest_bwddata_fp16_119, conv_ctest_bwddata_fp16_120, 
conv_ctest_bwddata_fp16_121, conv_ctest_bwddata_fp16_122, 
conv_ctest_bwddata_fp16_123, conv_ctest_bwddata_fp16_124, 
conv_ctest_bwddata_fp16_125, conv_ctest_bwddata_fp16_126, 
conv_ctest_bwddata_fp16_127, conv_ctest_bwddata_fp16_128, 
conv_ctest_bwddata_fp16_129, conv_ctest_bwddata_fp16_130, 
conv_ctest_bwddata_fp16_131, conv_ctest_bwddata_fp16_132, 
conv_ctest_bwddata_fp16_133, conv_ctest_bwddata_fp16_134, 
conv_ctest_bwddata_fp16_135, conv_ctest_bwddata_fp16_136, 
conv_ctest_bwddata_fp16_137, conv_ctest_bwddata_fp16_138, 
conv_ctest_bwddata_fp16_139, conv_ctest_bwddata_fp16_140, 
conv_ctest_bwddata_fp16_141, conv_ctest_bwddata_fp16_142, 
conv_ctest_bwddata_fp16_143, conv_ctest_bwddata_fp16_144, 
conv_ctest_bwddata_fp16_145, conv_ctest_bwddata_fp16_146, 
conv_ctest_bwddata_fp16_147, conv_ctest_bwddata_fp16_148, 
conv_ctest_bwddata_fp16_149, conv_ctest_bwddata_fp16_150, 
conv_ctest_bwddata_fp16_151, conv_ctest_bwddata_fp16_152, 
conv_ctest_bwddata_fp16_153, conv_ctest_bwddata_fp16_154, 
conv_ctest_bwddata_fp16_155, conv_ctest_bwddata_fp16_156, 
conv_ctest_bwddata_fp16_157, conv_ctest_bwddata_fp16_158, 
conv_ctest_bwddata_fp16_159, conv_ctest_bwddata_fp16_160, 
conv_ctest_bwddata_fp16_161, conv_ctest_bwddata_fp16_162, 
conv_ctest_bwddata_fp16_163, conv_ctest_bwddata_fp16_164, 
conv_ctest_bwddata_fp16_165, conv_ctest_bwddata_fp16_166, 
conv_ctest_bwddata_fp16_167, conv_ctest_bwddata_fp16_168, 
conv_ctest_bwddata_fp16_169, conv_ctest_bwddata_fp16_170, 
conv_ctest_bwddata_fp16_171, conv_ctest_bwddata_fp16_172, 
conv_ctest_bwddata_fp16_173, conv_ctest_bwddata_fp16_174, 
conv_ctest_bwddata_fp16_175, conv_ctest_bwddata_fp16_176, 
conv_ctest_bwddata_fp16_177, conv_ctest_bwddata_fp16_178, 
conv_ctest_bwddata_fp16_179, conv_ctest_bwddata_fp16_180, 
conv_ctest_bwddata_fp16_181, conv_ctest_bwddata_fp16_182, 
conv_ctest_bwddata_fp16_183, conv_ctest_bwddata_fp16_184, 
conv_ctest_bwddata_fp16_185, conv_ctest_bwddata_fp16_186, 
conv_ctest_bwddata_fp16_187, conv_ctest_bwddata_fp16_188, 
conv_ctest_bwddata_fp16_189, conv_ctest_bwddata_fp16_190, 
conv_ctest_bwddata_fp16_191, conv_ctest_bwddata_fp16_192, 
conv_ctest_bwddata_fp16_193, conv_ctest_bwddata_fp16_194, 
conv_ctest_bwddata_fp16_195, conv_ctest_bwddata_fp16_196, 
conv_ctest_bwddata_fp16_197, conv_ctest_bwddata_fp16_198, 
conv_ctest_bwddata_fp16_199, conv_ctest_bwddata_fp16_200, 
conv_ctest_bwddata_fp16_201, conv_ctest_bwddata_fp16_202, 
conv_ctest_bwddata_fp16_203, conv_ctest_bwddata_fp16_204, 
conv_ctest_bwddata_fp16_205, conv_ctest_bwddata_fp16_206, 
conv_ctest_bwddata_fp16_207, conv_ctest_bwddata_fp16_208, 
conv_ctest_bwddata_fp16_209, conv_ctest_bwddata_fp16_210, 
conv_ctest_bwddata_fp16_211, conv_ctest_bwddata_fp16_212, 
conv_ctest_bwddata_fp16_213, conv_ctest_bwddata_fp16_214, 
conv_ctest_bwddata_fp16_215, conv_ctest_bwddata_fp16_216, 
conv_ctest_bwddata_fp16_217, conv_ctest_bwddata_fp16_218, 
conv_ctest_bwddata_fp16_219, conv_ctest_bwddata_fp16_220, 
conv_ctest_bwddata_fp16_221, conv_ctest_bwddata_fp16_222, 
conv_ctest_bwddata_fp16_223, conv_ctest_bwddata_fp16_224, 
conv_ctest_bwddata_fp16_225, conv_ctest_bwddata_fp16_226, 
conv_ctest_bwddata_fp16_227, conv_ctest_bwddata_fp16_228, 
conv_ctest_bwddata_fp16_229, conv_ctest_bwddata_fp16_230, 
conv_ctest_bwddata_fp16_231, conv_ctest_bwddata_fp16_232, 
conv_ctest_bwddata_fp16_233, conv_ctest_bwddata_fp16_234, 
conv_ctest_bwddata_fp16_235, conv_ctest_bwddata_fp16_236, 
conv_ctest_bwddata_fp16_237, conv_ctest_bwddata_fp16_238, 
conv_ctest_bwddata_fp16_239, conv_ctest_bwddata_fp16_240, 
conv_ctest_bwddata_fp16_241, conv_ctest_bwddata_fp16_242, 
conv_ctest_bwddata_fp16_243, conv_ctest_bwddata_fp16_244, 
conv_ctest_bwddata_fp16_245, conv_ctest_bwddata_fp16_246, 
conv_ctest_bwddata_fp16_247, conv_ctest_bwddata_fp16_248, 
conv_ctest_bwddata_fp16_249, conv_ctest_bwddata_fp16_250, 
conv_ctest_bwddata_fp16_251, conv_ctest_bwddata_fp16_252, 
conv_ctest_bwddata_fp16_253, conv_ctest_bwddata_fp16_254, 
conv_ctest_bwddata_fp16_255, conv_ctest_bwddata_fp16_256, 
conv_ctest_bwddata_fp16_257, conv_ctest_bwddata_fp16_258, 
conv_ctest_bwddata_fp16_259, conv_ctest_bwddata_fp16_260, 
conv_ctest_bwddata_fp16_261, conv_ctest_bwddata_fp16_262, 
conv_ctest_bwddata_fp16_263, conv_ctest_bwddata_fp16_264, 
conv_ctest_bwddata_fp16_265, conv_ctest_bwddata_fp16_266, 
conv_ctest_bwddata_fp16_267, conv_ctest_bwddata_fp16_268, 
conv_ctest_bwddata_fp16_269, conv_ctest_bwddata_fp16_270, 
conv_ctest_bwddata_fp16_271, conv_ctest_bwddata_fp16_272, 
conv_ctest_bwddata_fp16_273, conv_ctest_bwddata_fp16_274, 
conv_ctest_bwddata_fp16_275, conv_ctest_bwddata_fp16_276, 
conv_ctest_bwddata_fp16_277, conv_ctest_bwddata_fp16_278, 
conv_ctest_bwddata_fp16_279, conv_ctest_bwddata_fp16_280, 
conv_ctest_bwddata_fp16_281, conv_ctest_bwddata_fp16_282, 
conv_ctest_bwddata_fp16_283, conv_ctest_bwddata_fp16_284, 
conv_ctest_bwddata_fp16_285, conv_ctest_bwddata_fp16_286, 
conv_ctest_bwddata_fp16_287, conv_ctest_bwddata_fp16_288, 
conv_ctest_bwddata_fp16_289, conv_ctest_bwddata_fp16_290, 
conv_ctest_bwddata_fp16_291, conv_ctest_bwddata_fp16_292, 
conv_ctest_bwddata_fp16_293, conv_ctest_bwddata_fp16_294, 
conv_ctest_bwddata_fp16_295, conv_ctest_bwddata_fp16_296, 
conv_ctest_bwddata_fp16_297, conv_ctest_bwddata_fp16_298, 
conv_ctest_bwddata_fp16_299, conv_ctest_bwddata_fp16_300, 
conv_ctest_bwddata_fp16_301, conv_ctest_bwddata_fp16_302, 
conv_ctest_bwddata_fp16_303, conv_ctest_bwddata_fp16_304, 
conv_ctest_bwddata_fp16_305, conv_ctest_bwddata_fp16_306, 
conv_ctest_bwddata_fp16_307, conv_ctest_bwddata_fp16_308, 
conv_ctest_bwddata_fp16_309, conv_ctest_bwddata_fp16_310, 
conv_ctest_bwddata_fp16_311, conv_ctest_bwddata_fp16_312, 
conv_ctest_bwddata_fp16_313, conv_ctest_bwddata_fp16_314, 
conv_ctest_bwddata_fp16_315, conv_ctest_bwddata_fp16_316, 
conv_ctest_bwddata_fp16_317, conv_ctest_bwddata_fp16_318, 
conv_ctest_bwddata_fp16_319, conv_ctest_bwddata_fp16_320, 
conv_ctest_bwddata_fp16_321, conv_ctest_bwddata_fp16_322, 
conv_ctest_bwddata_fp16_323, conv_ctest_bwddata_fp16_324, 
conv_ctest_bwddata_fp16_325, conv_ctest_bwddata_fp16_326, 
conv_ctest_bwddata_fp16_327, conv_ctest_bwddata_fp16_328, 
conv_ctest_bwddata_fp16_329, conv_ctest_bwddata_fp16_330, 
conv_ctest_bwddata_fp16_331, conv_ctest_bwddata_fp16_332, 
conv_ctest_bwddata_fp16_333, conv_ctest_bwddata_fp16_334, 
conv_ctest_bwddata_fp16_335, conv_ctest_bwddata_fp16_336, 
conv_ctest_bwddata_fp16_337, conv_ctest_bwddata_fp16_338, 
conv_ctest_bwddata_fp16_339, conv_ctest_bwddata_fp16_340, 
conv_ctest_bwddata_fp16_341, conv_ctest_bwddata_fp16_342, 
conv_ctest_bwddata_fp16_343, conv_ctest_bwddata_fp16_344, 
conv_ctest_bwddata_fp16_345, conv_ctest_bwddata_fp16_346, 
conv_ctest_bwddata_fp16_347, conv_ctest_bwddata_fp16_348, 
conv_ctest_bwddata_fp16_349, conv_ctest_bwddata_fp16_350, 
conv_ctest_bwddata_fp16_351, conv_ctest_bwddata_fp16_352, 
conv_ctest_bwddata_fp16_353, conv_ctest_bwddata_fp16_354, 
conv_ctest_bwddata_fp16_355, conv_ctest_bwddata_fp16_356, 
conv_ctest_bwddata_fp16_357, conv_ctest_bwddata_fp16_358, 
conv_ctest_bwddata_fp16_359, conv_ctest_bwddata_fp16_360, 
conv_ctest_bwddata_fp16_361, conv_ctest_bwddata_fp16_362, 
conv_ctest_bwddata_fp16_363, conv_ctest_bwddata_fp16_364, 
conv_ctest_bwddata_fp16_365, conv_ctest_bwddata_fp16_366, 
conv_ctest_bwddata_fp16_367, conv_ctest_bwddata_fp16_368, 
conv_ctest_bwddata_fp16_369, conv_ctest_bwddata_fp16_370, 
conv_ctest_bwddata_fp16_371, conv_ctest_bwddata_fp16_372, 
conv_ctest_bwddata_fp16_373, conv_ctest_bwddata_fp16_374, 
conv_ctest_bwddata_fp16_375, conv_ctest_bwddata_fp16_376, 
conv_ctest_bwddata_fp16_377, conv_ctest_bwddata_fp16_378, 
conv_ctest_bwddata_fp16_379, conv_ctest_bwddata_fp16_380, 
conv_ctest_bwddata_fp16_381, conv_ctest_bwddata_fp16_382, 
conv_ctest_bwddata_fp16_383, conv_ctest_bwddata_fp16_384, 
conv_ctest_bwddata_fp16_385, conv_ctest_bwddata_fp16_386, 
conv_ctest_bwddata_fp16_387, conv_ctest_bwddata_fp16_388, 
conv_ctest_bwddata_fp16_389, conv_ctest_bwddata_fp16_390, 
conv_ctest_bwddata_fp16_391, conv_ctest_bwddata_fp16_392, 
conv_ctest_bwddata_fp16_393, conv_ctest_bwddata_fp16_394, 
conv_ctest_bwddata_fp16_395, conv_ctest_bwddata_fp16_396, 
conv_ctest_bwddata_fp16_397, conv_ctest_bwddata_fp16_398, 
conv_ctest_bwddata_fp16_399, conv_ctest_bwddata_fp16_400, 
conv_ctest_bwddata_fp16_401, conv_ctest_bwddata_fp16_402, 
conv_ctest_bwddata_fp16_403, conv_ctest_bwddata_fp16_404, 
conv_ctest_bwddata_fp16_405, conv_ctest_bwddata_fp16_406, 
conv_ctest_bwddata_fp16_407, conv_ctest_bwddata_fp16_408, 
conv_ctest_bwddata_fp16_409, conv_ctest_bwddata_fp16_410, 
conv_ctest_bwddata_fp16_411, conv_ctest_bwddata_fp16_412, 
conv_ctest_bwddata_fp16_413, conv_ctest_bwddata_fp16_414, 
conv_ctest_bwddata_fp16_415, conv_ctest_bwddata_fp16_416, 
conv_ctest_bwddata_fp16_417, conv_ctest_bwddata_fp16_418, 
conv_ctest_bwddata_fp16_419, conv_ctest_bwddata_fp16_420, 
conv_ctest_bwddata_fp16_421, conv_ctest_bwddata_fp16_422, 
conv_ctest_bwddata_fp16_423, conv_ctest_bwddata_fp16_424, 
conv_ctest_bwddata_fp16_425, conv_ctest_bwddata_fp16_426, 
conv_ctest_bwddata_fp16_427, conv_ctest_bwddata_fp16_428, 
conv_ctest_bwddata_fp16_429, conv_ctest_bwddata_fp16_430, 
conv_ctest_bwddata_fp16_431, conv_ctest_bwddata_fp16_432, 
conv_ctest_bwddata_fp16_433, conv_ctest_bwddata_fp16_434, 
conv_ctest_bwddata_fp16_435, conv_ctest_bwddata_fp16_436, 
conv_ctest_bwddata_fp16_437, conv_ctest_bwddata_fp16_438, 
conv_ctest_bwddata_fp16_439, conv_ctest_bwddata_fp16_440, 
conv_ctest_bwddata_fp16_441, conv_ctest_bwddata_fp16_442, 
conv_ctest_bwddata_fp16_443, conv_ctest_bwddata_fp16_444, 
conv_ctest_bwddata_fp16_445, conv_ctest_bwddata_fp16_446, 
conv_ctest_bwddata_fp16_447, conv_ctest_bwddata_fp16_448, 
conv_ctest_bwddata_fp16_449, conv_ctest_bwddata_fp16_450, 
conv_ctest_bwddata_fp16_451, conv_ctest_bwddata_fp16_452, 
conv_ctest_bwddata_fp16_453, conv_ctest_bwddata_fp16_454, 
conv_ctest_bwddata_fp16_455, conv_ctest_bwddata_fp16_456, 
conv_ctest_bwddata_fp16_457, conv_ctest_bwddata_fp16_458, 
conv_ctest_bwddata_fp16_459, conv_ctest_bwddata_fp16_460, 
conv_ctest_bwddata_fp16_461, conv_ctest_bwddata_fp16_462, 
conv_ctest_bwddata_fp16_463, conv_ctest_bwddata_fp16_464, 
conv_ctest_bwddata_fp16_465, conv_ctest_bwddata_fp16_466, 
conv_ctest_bwddata_fp16_467, conv_ctest_bwddata_fp16_468, 
conv_ctest_bwddata_fp16_469, conv_ctest_bwddata_fp16_470, 
conv_ctest_bwddata_fp16_471, conv_ctest_bwddata_fp16_472, 
conv_ctest_bwddata_fp16_473, conv_ctest_bwddata_fp16_474, 
conv_ctest_bwddata_fp16_475, conv_ctest_bwddata_fp16_476, 
conv_ctest_bwddata_fp16_477, conv_ctest_bwddata_fp16_478, 
conv_ctest_bwddata_fp16_479, conv_ctest_bwddata_fp16_480, 
conv_ctest_bwddata_fp16_481, conv_ctest_bwddata_fp16_482, 
conv_ctest_bwddata_fp16_483, conv_ctest_bwddata_fp16_484, 
conv_ctest_bwddata_fp16_485, conv_ctest_bwddata_fp16_486, 
conv_ctest_bwddata_fp16_487, conv_ctest_bwddata_fp16_488, 
conv_ctest_bwddata_fp16_489, conv_ctest_bwddata_fp16_490, 
conv_ctest_bwddata_fp16_491, conv_ctest_bwddata_fp16_492, 
conv_ctest_bwddata_fp16_493, conv_ctest_bwddata_fp16_494, 
conv_ctest_bwddata_fp16_495, conv_ctest_bwddata_fp16_496, 
conv_ctest_bwddata_fp16_497, conv_ctest_bwddata_fp16_498, 
conv_ctest_bwddata_fp16_499, conv_ctest_bwddata_fp16_500, 
conv_ctest_bwddata_fp16_501, conv_ctest_bwddata_fp16_502, 
conv_ctest_bwddata_fp16_503, conv_ctest_bwddata_fp16_504, 
conv_ctest_bwddata_fp16_505, conv_ctest_bwddata_fp16_506, 
conv_ctest_bwddata_fp16_507, conv_ctest_bwddata_fp16_508, 
conv_ctest_bwddata_fp16_509, conv_ctest_bwddata_fp16_510, 
conv_ctest_bwddata_fp16_511, conv_ctest_bwddata_fp16_512, 
conv_ctest_bwddata_fp16_513, conv_ctest_bwddata_fp16_514, 
conv_ctest_bwddata_fp16_515, conv_ctest_bwddata_fp16_516, 
conv_ctest_bwddata_fp16_517, conv_ctest_bwddata_fp16_518, 
conv_ctest_bwddata_fp16_519, conv_ctest_bwddata_fp16_520, 
conv_ctest_bwddata_fp16_521, conv_ctest_bwddata_fp16_522, 
conv_ctest_bwddata_fp16_523, conv_ctest_bwddata_fp16_524, 
conv_ctest_bwddata_fp16_525, conv_ctest_bwddata_fp16_526, 
conv_ctest_bwddata_fp16_527, conv_ctest_bwddata_fp16_528, 
conv_ctest_bwddata_fp16_529, conv_ctest_bwddata_fp16_530, 
conv_ctest_bwddata_fp16_531, conv_ctest_bwddata_fp16_532, 
conv_ctest_bwddata_fp16_533, conv_ctest_bwddata_fp16_534, 
conv_ctest_bwddata_fp16_535, conv_ctest_bwddata_fp16_536, 
conv_ctest_bwddata_fp16_537, conv_ctest_bwddata_fp16_538, 
conv_ctest_bwddata_fp16_539, conv_ctest_bwddata_fp16_540, 
conv_ctest_bwddata_fp16_541, conv_ctest_bwddata_fp16_542, 
conv_ctest_bwddata_fp16_543, conv_ctest_bwddata_fp16_544, 
conv_ctest_bwddata_fp16_545, conv_ctest_bwddata_fp16_546, 
conv_ctest_bwddata_fp16_547, conv_ctest_bwddata_fp16_548, 
conv_ctest_bwddata_fp16_549, conv_ctest_bwddata_fp16_550, 
conv_ctest_bwddata_fp16_551, conv_ctest_bwddata_fp16_552, 
conv_ctest_bwddata_fp16_553, conv_ctest_bwddata_fp16_554, 
conv_ctest_bwddata_fp16_555, conv_ctest_bwddata_fp16_556, 
conv_ctest_bwddata_fp16_557, conv_ctest_bwddata_fp16_558, 
conv_ctest_bwddata_fp16_559, conv_ctest_bwddata_fp16_560, 
conv_ctest_bwddata_fp16_561, conv_ctest_bwddata_fp16_562, 
conv_ctest_bwddata_fp16_563, conv_ctest_bwddata_fp16_564, 
conv_ctest_bwddata_fp16_565, conv_ctest_bwddata_fp16_566, 
conv_ctest_bwddata_fp16_567, conv_ctest_bwddata_fp16_568, 
conv_ctest_bwddata_fp16_569, conv_ctest_bwddata_fp16_570, 
conv_ctest_bwddata_fp16_571, conv_ctest_bwddata_fp16_572, 
conv_ctest_bwddata_fp16_573, conv_ctest_bwddata_fp16_574, 
conv_ctest_bwddata_fp16_575, conv_ctest_bwddata_fp16_576, 
conv_ctest_bwddata_fp16_577, conv_ctest_bwddata_fp16_578, 
conv_ctest_bwddata_fp16_579, conv_ctest_bwddata_fp16_580, 
conv_ctest_bwddata_fp16_581, conv_ctest_bwddata_fp16_582, 
conv_ctest_bwddata_fp16_583, conv_ctest_bwddata_fp16_584, 
conv_ctest_bwddata_fp16_585, conv_ctest_bwddata_fp16_586, 
conv_ctest_bwddata_fp16_587, conv_ctest_bwddata_fp16_588, 
conv_ctest_bwddata_fp16_589, conv_ctest_bwddata_fp16_590, 
conv_ctest_bwddata_fp16_591, conv_ctest_bwddata_fp16_592, 
conv_ctest_bwddata_fp16_593, conv_ctest_bwddata_fp16_594, 
conv_ctest_bwddata_fp16_595, conv_ctest_bwddata_fp16_596, 
conv_ctest_bwddata_fp16_597, conv_ctest_bwddata_fp16_598, 
conv_ctest_bwddata_fp16_599, conv_ctest_bwddata_fp16_600, 
conv_ctest_bwddata_fp16_601, conv_ctest_bwddata_fp16_602, 
conv_ctest_bwddata_fp16_603, conv_ctest_bwddata_fp16_604, 
conv_ctest_bwddata_fp16_605, conv_ctest_bwddata_fp16_606, 
conv_ctest_bwddata_fp16_607, 
};

gemm_tuple conv_ctest_bwdwrw_fp32_001 {{1008, 1, 100, 100, 100, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_002 {{1008, 1, 144, 144, 144, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_003 {{1008, 1, 196, 196, 196, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_004 {{1008, 1, 256, 256, 256, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_005 {{1008, 1, 25, 25, 25, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_006 {{1008, 1, 36, 36, 36, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_007 {{1008, 1, 49, 49, 49, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_008 {{1008, 1, 81, 81, 81, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_009 {{1024, 1, 121, 121, 121, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_010 {{1024, 1, 144, 144, 144, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_011 {{1024, 1, 16, 16, 16, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_012 {{1024, 1, 196, 196, 196, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_013 {{1024, 1, 256, 256, 256, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_014 {{1024, 1, 25, 25, 25, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_015 {{1024, 1, 36, 36, 36, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_016 {{1024, 1, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_017 {{1024, 1, 81, 81, 81, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_018 {{1056, 1, 121, 121, 121, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_019 {{1056, 1, 16, 16, 16, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_020 {{1056, 1, 25, 25, 25, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_021 {{1056, 1, 49, 49, 49, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_022 {{1056, 1, 81, 81, 81, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_023 {{1152, 1, 100, 100, 100, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_024 {{1152, 1, 144, 144, 144, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_025 {{1152, 1, 169, 169, 169, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_026 {{1152, 1, 196, 196, 196, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_027 {{1152, 1, 256, 256, 256, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_028 {{1152, 1, 25, 25, 25, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_029 {{1152, 1, 2704, 2704, 2704, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_030 {{1152, 1, 2916, 2916, 2916, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_031 {{1152, 1, 3136, 3136, 3136, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_032 {{1152, 1, 3364, 3364, 3364, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_033 {{1152, 1, 36, 36, 36, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_034 {{1152, 1, 49, 49, 49, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_035 {{1152, 1, 576, 576, 576, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_036 {{1152, 1, 676, 676, 676, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_037 {{1152, 1, 729, 729, 729, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_038 {{1152, 1, 784, 784, 784, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_039 {{1152, 1, 81, 81, 81, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_040 {{1152, 1, 900, 900, 900, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_041 {{1200, 1, 16, 16, 16, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_042 {{1200, 1, 1, 1, 1, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_043 {{1200, 1, 25, 25, 25, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_044 {{1200, 1, 49, 49, 49, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_045 {{1200, 1, 4, 4, 4, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_046 {{1200, 1, 9, 9, 9, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_047 {{128, 1, 100, 100, 100, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_048 {{128, 1, 1024, 1024, 1024, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_049 {{128, 1, 196, 196, 196, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_050 {{128, 1, 225, 225, 225, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_051 {{128, 1, 256, 256, 256, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_052 {{128, 1, 289, 289, 289, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_053 {{128, 1, 3136, 3136, 3136, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_054 {{128, 1, 324, 324, 324, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_055 {{128, 1, 3364, 3364, 3364, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_056 {{128, 1, 3600, 3600, 3600, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_057 {{128, 1, 49, 49, 49, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_058 {{128, 1, 64, 64, 64, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_059 {{128, 1, 784, 784, 784, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_060 {{128, 1, 841, 841, 841, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_061 {{128, 1, 900, 900, 900, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_062 {{128, 1, 961, 961, 961, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_063 {{1296, 1, 100, 100, 100, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_064 {{1296, 1, 144, 144, 144, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_065 {{1296, 1, 196, 196, 196, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_066 {{1296, 1, 256, 256, 256, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_067 {{1296, 1, 25, 25, 25, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_068 {{1296, 1, 36, 36, 36, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_069 {{1296, 1, 49, 49, 49, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_070 {{1296, 1, 81, 81, 81, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_071 {{1440, 1, 100, 100, 100, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_072 {{1440, 1, 144, 144, 144, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_073 {{1440, 1, 16, 16, 16, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_074 {{1440, 1, 196, 196, 196, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_075 {{1440, 1, 256, 256, 256, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_076 {{1440, 1, 25, 25, 25, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_077 {{1440, 1, 36, 36, 36, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_078 {{1440, 1, 49, 49, 49, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_079 {{1440, 1, 4, 4, 4, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_080 {{1440, 1, 81, 81, 81, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_081 {{1440, 1, 9, 9, 9, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_082 {{147, 1, 1024, 1024, 1024, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_083 {{147, 1, 10609, 10609, 10609, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_084 {{147, 1, 10816, 10816, 10816, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_085 {{147, 1, 11025, 11025, 11025, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_086 {{147, 1, 11236, 11236, 11236, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_087 {{147, 1, 11449, 11449, 11449, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_088 {{147, 1, 11664, 11664, 11664, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_089 {{147, 1, 11881, 11881, 11881, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_090 {{147, 1, 12100, 12100, 12100, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_091 {{147, 1, 12321, 12321, 12321, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_092 {{147, 1, 12544, 12544, 12544, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_093 {{147, 1, 12769, 12769, 12769, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_094 {{147, 1, 12996, 12996, 12996, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_095 {{147, 1, 13456, 13456, 13456, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_096 {{147, 1, 169, 169, 169, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_097 {{147, 1, 196, 196, 196, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_098 {{147, 1, 256, 256, 256, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_099 {{147, 1, 400, 400, 400, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_100 {{147, 1, 44944, 44944, 44944, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_101 {{147, 1, 46225, 46225, 46225, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_102 {{147, 1, 47524, 47524, 47524, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_103 {{147, 1, 47961, 47961, 47961, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_104 {{147, 1, 48400, 48400, 48400, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_105 {{147, 1, 48841, 48841, 48841, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_106 {{147, 1, 49284, 49284, 49284, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_107 {{147, 1, 49729, 49729, 49729, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_108 {{147, 1, 49, 49, 49, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_109 {{147, 1, 50176, 50176, 50176, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_110 {{147, 1, 50625, 50625, 50625, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_111 {{147, 1, 51529, 51529, 51529, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_112 {{147, 1, 52441, 52441, 52441, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_113 {{147, 1, 53361, 53361, 53361, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_114 {{147, 1, 64, 64, 64, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_115 {{147, 1, 676, 676, 676, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_116 {{147, 1, 784, 784, 784, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_117 {{147, 1, 900, 900, 900, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_118 {{1600, 1, 100, 100, 100, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_119 {{1600, 1, 10816, 10816, 10816, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_120 {{1600, 1, 11664, 11664, 11664, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_121 {{1600, 1, 12100, 12100, 12100, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_122 {{1600, 1, 12544, 12544, 12544, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_123 {{1600, 1, 144, 144, 144, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_124 {{1600, 1, 169, 169, 169, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_125 {{1600, 1, 196, 196, 196, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_126 {{1600, 1, 225, 225, 225, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_127 {{1600, 1, 2304, 2304, 2304, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_128 {{1600, 1, 25, 25, 25, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_129 {{1600, 1, 2601, 2601, 2601, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_130 {{1600, 1, 2704, 2704, 2704, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_131 {{1600, 1, 2916, 2916, 2916, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_132 {{1600, 1, 3025, 3025, 3025, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_133 {{1600, 1, 3136, 3136, 3136, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_134 {{1600, 1, 3249, 3249, 3249, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_135 {{1600, 1, 361, 361, 361, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_136 {{1600, 1, 36, 36, 36, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_137 {{1600, 1, 400, 400, 400, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_138 {{1600, 1, 49, 49, 49, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_139 {{1600, 1, 4, 4, 4, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_140 {{1600, 1, 529, 529, 529, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_141 {{1600, 1, 576, 576, 576, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_142 {{1600, 1, 625, 625, 625, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_143 {{1600, 1, 64, 64, 64, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_144 {{1600, 1, 676, 676, 676, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_145 {{1600, 1, 729, 729, 729, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_146 {{1600, 1, 784, 784, 784, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_147 {{1600, 1, 81, 81, 81, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_148 {{1600, 1, 841, 841, 841, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_149 {{1728, 1, 100, 100, 100, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_150 {{1728, 1, 144, 144, 144, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_151 {{1728, 1, 169, 169, 169, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_152 {{1728, 1, 16, 16, 16, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_153 {{1728, 1, 196, 196, 196, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_154 {{1728, 1, 256, 256, 256, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_155 {{1728, 1, 25, 25, 25, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_156 {{1728, 1, 36, 36, 36, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_157 {{1728, 1, 49, 49, 49, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_158 {{1728, 1, 4, 4, 4, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_159 {{1728, 1, 576, 576, 576, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_160 {{1728, 1, 676, 676, 676, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_161 {{1728, 1, 784, 784, 784, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_162 {{1728, 1, 81, 81, 81, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_163 {{1728, 1, 900, 900, 900, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_164 {{1728, 1, 9, 9, 9, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_165 {{192, 1, 100, 100, 100, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_166 {{192, 1, 1024, 1024, 1024, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_167 {{192, 1, 121, 121, 121, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_168 {{192, 1, 16, 16, 16, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_169 {{192, 1, 196, 196, 196, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_170 {{192, 1, 225, 225, 225, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_171 {{192, 1, 256, 256, 256, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_172 {{192, 1, 25, 25, 25, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_173 {{192, 1, 289, 289, 289, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_174 {{192, 1, 324, 324, 324, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_175 {{192, 1, 49, 49, 49, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_176 {{192, 1, 64, 64, 64, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_177 {{192, 1, 784, 784, 784, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_178 {{192, 1, 81, 81, 81, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_179 {{192, 1, 900, 900, 900, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_180 {{2016, 1, 16, 16, 16, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_181 {{2016, 1, 25, 25, 25, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_182 {{2016, 1, 36, 36, 36, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_183 {{2016, 1, 49, 49, 49, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_184 {{2016, 1, 4, 4, 4, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_185 {{2016, 1, 81, 81, 81, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_186 {{2016, 1, 9, 9, 9, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_187 {{2048, 1, 121, 121, 121, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_188 {{2048, 1, 169, 169, 169, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_189 {{2048, 1, 225, 225, 225, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_190 {{2048, 1, 36, 36, 36, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_191 {{2048, 1, 49, 49, 49, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_192 {{2048, 1, 81, 81, 81, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_193 {{2304, 1, 100, 100, 100, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_194 {{2304, 1, 121, 121, 121, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_195 {{2304, 1, 144, 144, 144, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_196 {{2304, 1, 169, 169, 169, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_197 {{2304, 1, 16, 16, 16, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_198 {{2304, 1, 196, 196, 196, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_199 {{2304, 1, 225, 225, 225, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_200 {{2304, 1, 256, 256, 256, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_201 {{2304, 1, 25, 25, 25, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_202 {{2304, 1, 2704, 2704, 2704, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_203 {{2304, 1, 2916, 2916, 2916, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_204 {{2304, 1, 3136, 3136, 3136, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_205 {{2304, 1, 3364, 3364, 3364, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_206 {{2304, 1, 36, 36, 36, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_207 {{2304, 1, 49, 49, 49, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_208 {{2304, 1, 576, 576, 576, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_209 {{2304, 1, 64, 64, 64, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_210 {{2304, 1, 676, 676, 676, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_211 {{2304, 1, 729, 729, 729, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_212 {{2304, 1, 784, 784, 784, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_213 {{2304, 1, 81, 81, 81, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_214 {{2304, 1, 900, 900, 900, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_215 {{2400, 1, 100, 100, 100, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_216 {{2400, 1, 144, 144, 144, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_217 {{2400, 1, 169, 169, 169, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_218 {{2400, 1, 196, 196, 196, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_219 {{2400, 1, 225, 225, 225, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_220 {{2400, 1, 25, 25, 25, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_221 {{2400, 1, 361, 361, 361, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_222 {{2400, 1, 36, 36, 36, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_223 {{2400, 1, 400, 400, 400, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_224 {{2400, 1, 49, 49, 49, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_225 {{2400, 1, 4, 4, 4, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_226 {{2400, 1, 529, 529, 529, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_227 {{2400, 1, 576, 576, 576, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_228 {{2400, 1, 625, 625, 625, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_229 {{2400, 1, 64, 64, 64, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_230 {{2400, 1, 676, 676, 676, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_231 {{2400, 1, 729, 729, 729, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_232 {{2400, 1, 784, 784, 784, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_233 {{2400, 1, 81, 81, 81, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_234 {{256, 1, 100, 100, 100, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_235 {{256, 1, 1024, 1024, 1024, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_236 {{256, 1, 144, 144, 144, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_237 {{256, 1, 169, 169, 169, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_238 {{256, 1, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_239 {{256, 1, 225, 225, 225, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_240 {{256, 1, 256, 256, 256, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_241 {{256, 1, 289, 289, 289, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_242 {{256, 1, 3136, 3136, 3136, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_243 {{256, 1, 324, 324, 324, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_244 {{256, 1, 3364, 3364, 3364, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_245 {{256, 1, 3600, 3600, 3600, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_246 {{256, 1, 36, 36, 36, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_247 {{256, 1, 49, 49, 49, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_248 {{256, 1, 64, 64, 64, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_249 {{256, 1, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_250 {{256, 1, 81, 81, 81, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_251 {{256, 1, 841, 841, 841, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_252 {{256, 1, 900, 900, 900, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_253 {{256, 1, 961, 961, 961, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_254 {{27, 1, 1024, 1024, 1024, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_255 {{27, 1, 1156, 1156, 1156, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_256 {{27, 1, 12100, 12100, 12100, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_257 {{27, 1, 12321, 12321, 12321, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_258 {{27, 1, 12544, 12544, 12544, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_259 {{27, 1, 12769, 12769, 12769, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_260 {{27, 1, 12996, 12996, 12996, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_261 {{27, 1, 13225, 13225, 13225, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_262 {{27, 1, 13456, 13456, 13456, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_263 {{27, 1, 13924, 13924, 13924, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_264 {{27, 1, 196, 196, 196, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_265 {{27, 1, 225, 225, 225, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_266 {{27, 1, 256, 256, 256, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_267 {{27, 1, 324, 324, 324, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_268 {{27, 1, 48400, 48400, 48400, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_269 {{27, 1, 49284, 49284, 49284, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_270 {{27, 1, 49729, 49729, 49729, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_271 {{27, 1, 50176, 50176, 50176, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_272 {{27, 1, 50625, 50625, 50625, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_273 {{27, 1, 51076, 51076, 51076, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_274 {{27, 1, 51529, 51529, 51529, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_275 {{27, 1, 52441, 52441, 52441, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_276 {{27, 1, 53361, 53361, 53361, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_277 {{27, 1, 54289, 54289, 54289, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_278 {{27, 1, 784, 784, 784, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_279 {{27, 1, 900, 900, 900, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_280 {{320, 1, 1024, 1024, 1024, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_281 {{320, 1, 196, 196, 196, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_282 {{320, 1, 225, 225, 225, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_283 {{320, 1, 289, 289, 289, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_284 {{320, 1, 784, 784, 784, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_285 {{320, 1, 900, 900, 900, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_286 {{3456, 1, 121, 121, 121, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_287 {{3456, 1, 169, 169, 169, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_288 {{3456, 1, 225, 225, 225, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_289 {{3456, 1, 25, 25, 25, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_290 {{3456, 1, 36, 36, 36, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_291 {{3456, 1, 49, 49, 49, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_292 {{3456, 1, 81, 81, 81, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_293 {{363, 1, 10000, 10000, 10000, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_294 {{363, 1, 1024, 1024, 1024, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_295 {{363, 1, 10404, 10404, 10404, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_296 {{363, 1, 11449, 11449, 11449, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_297 {{363, 1, 11664, 11664, 11664, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_298 {{363, 1, 11881, 11881, 11881, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_299 {{363, 1, 12100, 12100, 12100, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_300 {{363, 1, 121, 121, 121, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_301 {{363, 1, 12321, 12321, 12321, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_302 {{363, 1, 12544, 12544, 12544, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_303 {{363, 1, 12996, 12996, 12996, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_304 {{363, 1, 13456, 13456, 13456, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_305 {{363, 1, 144, 144, 144, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_306 {{363, 1, 196, 196, 196, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_307 {{363, 1, 1, 1, 1, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_308 {{363, 1, 256, 256, 256, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_309 {{363, 1, 41616, 41616, 41616, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_310 {{363, 1, 42849, 42849, 42849, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_311 {{363, 1, 44521, 44521, 44521, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_312 {{363, 1, 45796, 45796, 45796, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_313 {{363, 1, 46656, 46656, 46656, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_314 {{363, 1, 47089, 47089, 47089, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_315 {{363, 1, 47524, 47524, 47524, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_316 {{363, 1, 47961, 47961, 47961, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_317 {{363, 1, 484, 484, 484, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_318 {{363, 1, 48841, 48841, 48841, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_319 {{363, 1, 49729, 49729, 49729, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_320 {{363, 1, 4, 4, 4, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_321 {{363, 1, 50176, 50176, 50176, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_322 {{363, 1, 50625, 50625, 50625, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_323 {{363, 1, 51529, 51529, 51529, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_324 {{363, 1, 53361, 53361, 53361, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_325 {{363, 1, 576, 576, 576, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_326 {{363, 1, 676, 676, 676, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_327 {{363, 1, 9025, 9025, 9025, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_328 {{363, 1, 9409, 9409, 9409, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_329 {{363, 1, 9604, 9604, 9604, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_330 {{363, 1, 9801, 9801, 9801, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_331 {{400, 1, 100, 100, 100, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_332 {{400, 1, 144, 144, 144, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_333 {{400, 1, 169, 169, 169, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_334 {{400, 1, 196, 196, 196, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_335 {{400, 1, 225, 225, 225, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_336 {{400, 1, 25, 25, 25, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_337 {{400, 1, 36, 36, 36, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_338 {{400, 1, 400, 400, 400, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_339 {{400, 1, 49, 49, 49, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_340 {{400, 1, 4, 4, 4, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_341 {{400, 1, 576, 576, 576, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_342 {{400, 1, 64, 64, 64, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_343 {{400, 1, 676, 676, 676, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_344 {{400, 1, 784, 784, 784, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_345 {{400, 1, 81, 81, 81, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_346 {{4608, 1, 100, 100, 100, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_347 {{4608, 1, 144, 144, 144, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_348 {{4608, 1, 169, 169, 169, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_349 {{4608, 1, 16, 16, 16, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_350 {{4608, 1, 1860, 1860, 1860, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_351 {{4608, 1, 1953, 1953, 1953, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_352 {{4608, 1, 196, 196, 196, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_353 {{4608, 1, 1, 1, 1, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_354 {{4608, 1, 2048, 2048, 2048, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_355 {{4608, 1, 2244, 2244, 2244, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_356 {{4608, 1, 256, 256, 256, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_357 {{4608, 1, 25, 25, 25, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_358 {{4608, 1, 36, 36, 36, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_359 {{4608, 1, 49, 49, 49, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_360 {{4608, 1, 4, 4, 4, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_361 {{4608, 1, 576, 576, 576, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_362 {{4608, 1, 64, 64, 64, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_363 {{4608, 1, 676, 676, 676, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_364 {{4608, 1, 7440, 7440, 7440, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_365 {{4608, 1, 7812, 7812, 7812, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_366 {{4608, 1, 784, 784, 784, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_367 {{4608, 1, 8192, 8192, 8192, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_368 {{4608, 1, 81, 81, 81, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_369 {{4608, 1, 8580, 8580, 8580, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_370 {{4608, 1, 900, 900, 900, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_371 {{4608, 1, 9, 9, 9, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_372 {{480, 1, 100, 100, 100, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_373 {{480, 1, 196, 196, 196, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_374 {{480, 1, 2048, 2048, 2048, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_375 {{480, 1, 2145, 2145, 2145, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_376 {{480, 1, 2345, 2345, 2345, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_377 {{480, 1, 256, 256, 256, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_378 {{480, 1, 324, 324, 324, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_379 {{480, 1, 32768, 32768, 32768, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_380 {{480, 1, 33540, 33540, 33540, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_381 {{480, 1, 34320, 34320, 34320, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_382 {{480, 1, 49, 49, 49, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_383 {{480, 1, 64, 64, 64, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_384 {{480, 1, 8192, 8192, 8192, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_385 {{480, 1, 8385, 8385, 8385, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_386 {{480, 1, 8580, 8580, 8580, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_387 {{480, 1, 8777, 8777, 8777, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_388 {{480, 1, 8976, 8976, 8976, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_389 {{4, 1, 100, 100, 100, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_390 {{4, 1, 121, 121, 121, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_391 {{4, 1, 144, 144, 144, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_392 {{4, 1, 169, 169, 169, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_393 {{4, 1, 16, 16, 16, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_394 {{4, 1, 196, 196, 196, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_395 {{4, 1, 1, 1, 1, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_396 {{4, 1, 225, 225, 225, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_397 {{4, 1, 256, 256, 256, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_398 {{4, 1, 25, 25, 25, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_399 {{4, 1, 289, 289, 289, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_400 {{4, 1, 36, 36, 36, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_401 {{4, 1, 49, 49, 49, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_402 {{4, 1, 4, 4, 4, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_403 {{4, 1, 625, 625, 625, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_404 {{4, 1, 64, 64, 64, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_405 {{4, 1, 676, 676, 676, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_406 {{4, 1, 729, 729, 729, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_407 {{4, 1, 784, 784, 784, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_408 {{4, 1, 81, 81, 81, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_409 {{4, 1, 900, 900, 900, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_410 {{4, 1, 9, 9, 9, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_411 {{512, 1, 100, 100, 100, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_412 {{512, 1, 1024, 1024, 1024, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_413 {{512, 1, 121, 121, 121, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_414 {{512, 1, 144, 144, 144, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_415 {{512, 1, 16, 16, 16, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_416 {{512, 1, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_417 {{512, 1, 2048, 2048, 2048, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_418 {{512, 1, 2145, 2145, 2145, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_419 {{512, 1, 225, 225, 225, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_420 {{512, 1, 2345, 2345, 2345, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_421 {{512, 1, 256, 256, 256, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_422 {{512, 1, 25, 25, 25, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_423 {{512, 1, 289, 289, 289, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_424 {{512, 1, 324, 324, 324, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_425 {{512, 1, 36, 36, 36, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_426 {{512, 1, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_427 {{512, 1, 4, 4, 4, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_428 {{512, 1, 64, 64, 64, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_429 {{512, 1, 784, 784, 784, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_430 {{512, 1, 8192, 8192, 8192, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_431 {{512, 1, 81, 81, 81, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_432 {{512, 1, 8580, 8580, 8580, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_433 {{512, 1, 8976, 8976, 8976, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_434 {{512, 1, 900, 900, 900, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_435 {{512, 1, 9, 9, 9, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_436 {{528, 1, 100, 100, 100, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_437 {{528, 1, 16, 16, 16, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_438 {{528, 1, 196, 196, 196, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_439 {{528, 1, 2048, 2048, 2048, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_440 {{528, 1, 2145, 2145, 2145, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_441 {{528, 1, 2345, 2345, 2345, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_442 {{528, 1, 256, 256, 256, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_443 {{528, 1, 25, 25, 25, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_444 {{528, 1, 324, 324, 324, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_445 {{528, 1, 36, 36, 36, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_446 {{528, 1, 49, 49, 49, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_447 {{528, 1, 4, 4, 4, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_448 {{528, 1, 64, 64, 64, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_449 {{528, 1, 8192, 8192, 8192, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_450 {{528, 1, 8580, 8580, 8580, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_451 {{528, 1, 8976, 8976, 8976, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_452 {{528, 1, 9, 9, 9, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_453 {{576, 1, 100, 100, 100, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_454 {{576, 1, 11664, 11664, 11664, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_455 {{576, 1, 12100, 12100, 12100, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_456 {{576, 1, 12544, 12544, 12544, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_457 {{576, 1, 12996, 12996, 12996, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_458 {{576, 1, 144, 144, 144, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_459 {{576, 1, 169, 169, 169, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_460 {{576, 1, 16, 16, 16, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_461 {{576, 1, 196, 196, 196, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_462 {{576, 1, 256, 256, 256, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_463 {{576, 1, 25, 25, 25, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_464 {{576, 1, 2704, 2704, 2704, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_465 {{576, 1, 2916, 2916, 2916, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_466 {{576, 1, 3025, 3025, 3025, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_467 {{576, 1, 3136, 3136, 3136, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_468 {{576, 1, 324, 324, 324, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_469 {{576, 1, 3364, 3364, 3364, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_470 {{576, 1, 36, 36, 36, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_471 {{576, 1, 49, 49, 49, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_472 {{576, 1, 4, 4, 4, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_473 {{576, 1, 529, 529, 529, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_474 {{576, 1, 576, 576, 576, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_475 {{576, 1, 625, 625, 625, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_476 {{576, 1, 64, 64, 64, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_477 {{576, 1, 676, 676, 676, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_478 {{576, 1, 729, 729, 729, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_479 {{576, 1, 784, 784, 784, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_480 {{576, 1, 81, 81, 81, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_481 {{576, 1, 841, 841, 841, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_482 {{576, 1, 900, 900, 900, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_483 {{576, 1, 9, 9, 9, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_484 {{600, 1, 100, 100, 100, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_485 {{600, 1, 144, 144, 144, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_486 {{600, 1, 196, 196, 196, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_487 {{600, 1, 25, 25, 25, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_488 {{600, 1, 36, 36, 36, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_489 {{600, 1, 49, 49, 49, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_490 {{600, 1, 4, 4, 4, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_491 {{600, 1, 64, 64, 64, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_492 {{608, 1, 100, 100, 100, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_493 {{608, 1, 16, 16, 16, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_494 {{608, 1, 196, 196, 196, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_495 {{608, 1, 256, 256, 256, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_496 {{608, 1, 25, 25, 25, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_497 {{608, 1, 324, 324, 324, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_498 {{608, 1, 36, 36, 36, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_499 {{608, 1, 49, 49, 49, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_500 {{608, 1, 4, 4, 4, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_501 {{608, 1, 64, 64, 64, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_502 {{608, 1, 9, 9, 9, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_503 {{64, 1, 100, 100, 100, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_504 {{64, 1, 1024, 1024, 1024, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_505 {{64, 1, 12544, 12544, 12544, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_506 {{64, 1, 12996, 12996, 12996, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_507 {{64, 1, 13456, 13456, 13456, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_508 {{64, 1, 196, 196, 196, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_509 {{64, 1, 225, 225, 225, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_510 {{64, 1, 256, 256, 256, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_511 {{64, 1, 289, 289, 289, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_512 {{64, 1, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_513 {{64, 1, 3249, 3249, 3249, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_514 {{64, 1, 324, 324, 324, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_515 {{64, 1, 3364, 3364, 3364, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_516 {{64, 1, 3481, 3481, 3481, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_517 {{64, 1, 3600, 3600, 3600, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_518 {{64, 1, 49, 49, 49, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_519 {{64, 1, 64, 64, 64, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_520 {{64, 1, 729, 729, 729, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_521 {{64, 1, 784, 784, 784, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_522 {{64, 1, 841, 841, 841, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_523 {{64, 1, 900, 900, 900, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_524 {{64, 1, 961, 961, 961, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_525 {{75, 1, 1024, 1024, 1024, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_526 {{75, 1, 11449, 11449, 11449, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_527 {{75, 1, 11881, 11881, 11881, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_528 {{75, 1, 12100, 12100, 12100, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_529 {{75, 1, 121, 121, 121, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_530 {{75, 1, 12321, 12321, 12321, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_531 {{75, 1, 12544, 12544, 12544, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_532 {{75, 1, 12769, 12769, 12769, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_533 {{75, 1, 12996, 12996, 12996, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_534 {{75, 1, 13225, 13225, 13225, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_535 {{75, 1, 13456, 13456, 13456, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_536 {{75, 1, 13689, 13689, 13689, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_537 {{75, 1, 196, 196, 196, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_538 {{75, 1, 225, 225, 225, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_539 {{75, 1, 256, 256, 256, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_540 {{75, 1, 289, 289, 289, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_541 {{75, 1, 46656, 46656, 46656, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_542 {{75, 1, 47961, 47961, 47961, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_543 {{75, 1, 48400, 48400, 48400, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_544 {{75, 1, 49284, 49284, 49284, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_545 {{75, 1, 49729, 49729, 49729, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_546 {{75, 1, 50176, 50176, 50176, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_547 {{75, 1, 50625, 50625, 50625, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_548 {{75, 1, 51529, 51529, 51529, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_549 {{75, 1, 52441, 52441, 52441, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_550 {{75, 1, 53361, 53361, 53361, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_551 {{75, 1, 576, 576, 576, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_552 {{75, 1, 784, 784, 784, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_553 {{75, 1, 900, 900, 900, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_554 {{800, 1, 100, 100, 100, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_555 {{800, 1, 144, 144, 144, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_556 {{800, 1, 169, 169, 169, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_557 {{800, 1, 16, 16, 16, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_558 {{800, 1, 196, 196, 196, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_559 {{800, 1, 1, 1, 1, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_560 {{800, 1, 225, 225, 225, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_561 {{800, 1, 256, 256, 256, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_562 {{800, 1, 25, 25, 25, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_563 {{800, 1, 36, 36, 36, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_564 {{800, 1, 400, 400, 400, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_565 {{800, 1, 49, 49, 49, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_566 {{800, 1, 4, 4, 4, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_567 {{800, 1, 576, 576, 576, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_568 {{800, 1, 64, 64, 64, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_569 {{800, 1, 676, 676, 676, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_570 {{800, 1, 784, 784, 784, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_571 {{800, 1, 81, 81, 81, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_572 {{800, 1, 9, 9, 9, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_573 {{832, 1, 121, 121, 121, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_574 {{832, 1, 16, 16, 16, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_575 {{832, 1, 2048, 2048, 2048, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_576 {{832, 1, 2145, 2145, 2145, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_577 {{832, 1, 2345, 2345, 2345, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_578 {{832, 1, 25, 25, 25, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_579 {{832, 1, 49, 49, 49, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_580 {{832, 1, 8192, 8192, 8192, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_581 {{832, 1, 81, 81, 81, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_582 {{832, 1, 8580, 8580, 8580, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_583 {{832, 1, 8976, 8976, 8976, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_584 {{864, 1, 100, 100, 100, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_585 {{864, 1, 144, 144, 144, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_586 {{864, 1, 169, 169, 169, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_587 {{864, 1, 196, 196, 196, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_588 {{864, 1, 256, 256, 256, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_589 {{864, 1, 25, 25, 25, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_590 {{864, 1, 36, 36, 36, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_591 {{864, 1, 49, 49, 49, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_592 {{864, 1, 529, 529, 529, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_593 {{864, 1, 576, 576, 576, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_594 {{864, 1, 625, 625, 625, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_595 {{864, 1, 676, 676, 676, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_596 {{864, 1, 729, 729, 729, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_597 {{864, 1, 784, 784, 784, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_598 {{864, 1, 81, 81, 81, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_599 {{864, 1, 841, 841, 841, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_600 {{864, 1, 900, 900, 900, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_601 {{9216, 1, 100, 100, 100, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_602 {{9216, 1, 144, 144, 144, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_603 {{9216, 1, 16, 16, 16, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_604 {{9216, 1, 196, 196, 196, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_605 {{9216, 1, 25, 25, 25, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_606 {{9216, 1, 36, 36, 36, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_607 {{9216, 1, 49, 49, 49, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_608 {{9216, 1, 4, 4, 4, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_609 {{9216, 1, 64, 64, 64, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_610 {{9216, 1, 81, 81, 81, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_611 {{9216, 1, 9, 9, 9, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_612 {{9, 1, 100, 100, 100, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_613 {{9, 1, 144, 144, 144, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_614 {{9, 1, 169, 169, 169, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_615 {{9, 1, 16, 16, 16, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_616 {{9, 1, 196, 196, 196, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_617 {{9, 1, 1, 1, 1, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_618 {{9, 1, 256, 256, 256, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_619 {{9, 1, 25, 25, 25, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_620 {{9, 1, 36, 36, 36, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_621 {{9, 1, 49, 49, 49, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_622 {{9, 1, 4, 4, 4, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_623 {{9, 1, 529, 529, 529, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_624 {{9, 1, 625, 625, 625, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_625 {{9, 1, 64, 64, 64, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_626 {{9, 1, 729, 729, 729, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_627 {{9, 1, 81, 81, 81, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_628 {{9, 1, 841, 841, 841, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp32_629 {{9, 1, 9, 9, 9, 9}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> conv_ctest_bwdwrw_fp32 = {
conv_ctest_bwdwrw_fp32_001, conv_ctest_bwdwrw_fp32_002, 
conv_ctest_bwdwrw_fp32_003, conv_ctest_bwdwrw_fp32_004, 
conv_ctest_bwdwrw_fp32_005, conv_ctest_bwdwrw_fp32_006, 
conv_ctest_bwdwrw_fp32_007, conv_ctest_bwdwrw_fp32_008, 
conv_ctest_bwdwrw_fp32_009, conv_ctest_bwdwrw_fp32_010, 
conv_ctest_bwdwrw_fp32_011, conv_ctest_bwdwrw_fp32_012, 
conv_ctest_bwdwrw_fp32_013, conv_ctest_bwdwrw_fp32_014, 
conv_ctest_bwdwrw_fp32_015, conv_ctest_bwdwrw_fp32_016, 
conv_ctest_bwdwrw_fp32_017, conv_ctest_bwdwrw_fp32_018, 
conv_ctest_bwdwrw_fp32_019, conv_ctest_bwdwrw_fp32_020, 
conv_ctest_bwdwrw_fp32_021, conv_ctest_bwdwrw_fp32_022, 
conv_ctest_bwdwrw_fp32_023, conv_ctest_bwdwrw_fp32_024, 
conv_ctest_bwdwrw_fp32_025, conv_ctest_bwdwrw_fp32_026, 
conv_ctest_bwdwrw_fp32_027, conv_ctest_bwdwrw_fp32_028, 
conv_ctest_bwdwrw_fp32_029, conv_ctest_bwdwrw_fp32_030, 
conv_ctest_bwdwrw_fp32_031, conv_ctest_bwdwrw_fp32_032, 
conv_ctest_bwdwrw_fp32_033, conv_ctest_bwdwrw_fp32_034, 
conv_ctest_bwdwrw_fp32_035, conv_ctest_bwdwrw_fp32_036, 
conv_ctest_bwdwrw_fp32_037, conv_ctest_bwdwrw_fp32_038, 
conv_ctest_bwdwrw_fp32_039, conv_ctest_bwdwrw_fp32_040, 
conv_ctest_bwdwrw_fp32_041, conv_ctest_bwdwrw_fp32_042, 
conv_ctest_bwdwrw_fp32_043, conv_ctest_bwdwrw_fp32_044, 
conv_ctest_bwdwrw_fp32_045, conv_ctest_bwdwrw_fp32_046, 
conv_ctest_bwdwrw_fp32_047, conv_ctest_bwdwrw_fp32_048, 
conv_ctest_bwdwrw_fp32_049, conv_ctest_bwdwrw_fp32_050, 
conv_ctest_bwdwrw_fp32_051, conv_ctest_bwdwrw_fp32_052, 
conv_ctest_bwdwrw_fp32_053, conv_ctest_bwdwrw_fp32_054, 
conv_ctest_bwdwrw_fp32_055, conv_ctest_bwdwrw_fp32_056, 
conv_ctest_bwdwrw_fp32_057, conv_ctest_bwdwrw_fp32_058, 
conv_ctest_bwdwrw_fp32_059, conv_ctest_bwdwrw_fp32_060, 
conv_ctest_bwdwrw_fp32_061, conv_ctest_bwdwrw_fp32_062, 
conv_ctest_bwdwrw_fp32_063, conv_ctest_bwdwrw_fp32_064, 
conv_ctest_bwdwrw_fp32_065, conv_ctest_bwdwrw_fp32_066, 
conv_ctest_bwdwrw_fp32_067, conv_ctest_bwdwrw_fp32_068, 
conv_ctest_bwdwrw_fp32_069, conv_ctest_bwdwrw_fp32_070, 
conv_ctest_bwdwrw_fp32_071, conv_ctest_bwdwrw_fp32_072, 
conv_ctest_bwdwrw_fp32_073, conv_ctest_bwdwrw_fp32_074, 
conv_ctest_bwdwrw_fp32_075, conv_ctest_bwdwrw_fp32_076, 
conv_ctest_bwdwrw_fp32_077, conv_ctest_bwdwrw_fp32_078, 
conv_ctest_bwdwrw_fp32_079, conv_ctest_bwdwrw_fp32_080, 
conv_ctest_bwdwrw_fp32_081, conv_ctest_bwdwrw_fp32_082, 
conv_ctest_bwdwrw_fp32_083, conv_ctest_bwdwrw_fp32_084, 
conv_ctest_bwdwrw_fp32_085, conv_ctest_bwdwrw_fp32_086, 
conv_ctest_bwdwrw_fp32_087, conv_ctest_bwdwrw_fp32_088, 
conv_ctest_bwdwrw_fp32_089, conv_ctest_bwdwrw_fp32_090, 
conv_ctest_bwdwrw_fp32_091, conv_ctest_bwdwrw_fp32_092, 
conv_ctest_bwdwrw_fp32_093, conv_ctest_bwdwrw_fp32_094, 
conv_ctest_bwdwrw_fp32_095, conv_ctest_bwdwrw_fp32_096, 
conv_ctest_bwdwrw_fp32_097, conv_ctest_bwdwrw_fp32_098, 
conv_ctest_bwdwrw_fp32_099, conv_ctest_bwdwrw_fp32_100, 
conv_ctest_bwdwrw_fp32_101, conv_ctest_bwdwrw_fp32_102, 
conv_ctest_bwdwrw_fp32_103, conv_ctest_bwdwrw_fp32_104, 
conv_ctest_bwdwrw_fp32_105, conv_ctest_bwdwrw_fp32_106, 
conv_ctest_bwdwrw_fp32_107, conv_ctest_bwdwrw_fp32_108, 
conv_ctest_bwdwrw_fp32_109, conv_ctest_bwdwrw_fp32_110, 
conv_ctest_bwdwrw_fp32_111, conv_ctest_bwdwrw_fp32_112, 
conv_ctest_bwdwrw_fp32_113, conv_ctest_bwdwrw_fp32_114, 
conv_ctest_bwdwrw_fp32_115, conv_ctest_bwdwrw_fp32_116, 
conv_ctest_bwdwrw_fp32_117, conv_ctest_bwdwrw_fp32_118, 
conv_ctest_bwdwrw_fp32_119, conv_ctest_bwdwrw_fp32_120, 
conv_ctest_bwdwrw_fp32_121, conv_ctest_bwdwrw_fp32_122, 
conv_ctest_bwdwrw_fp32_123, conv_ctest_bwdwrw_fp32_124, 
conv_ctest_bwdwrw_fp32_125, conv_ctest_bwdwrw_fp32_126, 
conv_ctest_bwdwrw_fp32_127, conv_ctest_bwdwrw_fp32_128, 
conv_ctest_bwdwrw_fp32_129, conv_ctest_bwdwrw_fp32_130, 
conv_ctest_bwdwrw_fp32_131, conv_ctest_bwdwrw_fp32_132, 
conv_ctest_bwdwrw_fp32_133, conv_ctest_bwdwrw_fp32_134, 
conv_ctest_bwdwrw_fp32_135, conv_ctest_bwdwrw_fp32_136, 
conv_ctest_bwdwrw_fp32_137, conv_ctest_bwdwrw_fp32_138, 
conv_ctest_bwdwrw_fp32_139, conv_ctest_bwdwrw_fp32_140, 
conv_ctest_bwdwrw_fp32_141, conv_ctest_bwdwrw_fp32_142, 
conv_ctest_bwdwrw_fp32_143, conv_ctest_bwdwrw_fp32_144, 
conv_ctest_bwdwrw_fp32_145, conv_ctest_bwdwrw_fp32_146, 
conv_ctest_bwdwrw_fp32_147, conv_ctest_bwdwrw_fp32_148, 
conv_ctest_bwdwrw_fp32_149, conv_ctest_bwdwrw_fp32_150, 
conv_ctest_bwdwrw_fp32_151, conv_ctest_bwdwrw_fp32_152, 
conv_ctest_bwdwrw_fp32_153, conv_ctest_bwdwrw_fp32_154, 
conv_ctest_bwdwrw_fp32_155, conv_ctest_bwdwrw_fp32_156, 
conv_ctest_bwdwrw_fp32_157, conv_ctest_bwdwrw_fp32_158, 
conv_ctest_bwdwrw_fp32_159, conv_ctest_bwdwrw_fp32_160, 
conv_ctest_bwdwrw_fp32_161, conv_ctest_bwdwrw_fp32_162, 
conv_ctest_bwdwrw_fp32_163, conv_ctest_bwdwrw_fp32_164, 
conv_ctest_bwdwrw_fp32_165, conv_ctest_bwdwrw_fp32_166, 
conv_ctest_bwdwrw_fp32_167, conv_ctest_bwdwrw_fp32_168, 
conv_ctest_bwdwrw_fp32_169, conv_ctest_bwdwrw_fp32_170, 
conv_ctest_bwdwrw_fp32_171, conv_ctest_bwdwrw_fp32_172, 
conv_ctest_bwdwrw_fp32_173, conv_ctest_bwdwrw_fp32_174, 
conv_ctest_bwdwrw_fp32_175, conv_ctest_bwdwrw_fp32_176, 
conv_ctest_bwdwrw_fp32_177, conv_ctest_bwdwrw_fp32_178, 
conv_ctest_bwdwrw_fp32_179, conv_ctest_bwdwrw_fp32_180, 
conv_ctest_bwdwrw_fp32_181, conv_ctest_bwdwrw_fp32_182, 
conv_ctest_bwdwrw_fp32_183, conv_ctest_bwdwrw_fp32_184, 
conv_ctest_bwdwrw_fp32_185, conv_ctest_bwdwrw_fp32_186, 
conv_ctest_bwdwrw_fp32_187, conv_ctest_bwdwrw_fp32_188, 
conv_ctest_bwdwrw_fp32_189, conv_ctest_bwdwrw_fp32_190, 
conv_ctest_bwdwrw_fp32_191, conv_ctest_bwdwrw_fp32_192, 
conv_ctest_bwdwrw_fp32_193, conv_ctest_bwdwrw_fp32_194, 
conv_ctest_bwdwrw_fp32_195, conv_ctest_bwdwrw_fp32_196, 
conv_ctest_bwdwrw_fp32_197, conv_ctest_bwdwrw_fp32_198, 
conv_ctest_bwdwrw_fp32_199, conv_ctest_bwdwrw_fp32_200, 
conv_ctest_bwdwrw_fp32_201, conv_ctest_bwdwrw_fp32_202, 
conv_ctest_bwdwrw_fp32_203, conv_ctest_bwdwrw_fp32_204, 
conv_ctest_bwdwrw_fp32_205, conv_ctest_bwdwrw_fp32_206, 
conv_ctest_bwdwrw_fp32_207, conv_ctest_bwdwrw_fp32_208, 
conv_ctest_bwdwrw_fp32_209, conv_ctest_bwdwrw_fp32_210, 
conv_ctest_bwdwrw_fp32_211, conv_ctest_bwdwrw_fp32_212, 
conv_ctest_bwdwrw_fp32_213, conv_ctest_bwdwrw_fp32_214, 
conv_ctest_bwdwrw_fp32_215, conv_ctest_bwdwrw_fp32_216, 
conv_ctest_bwdwrw_fp32_217, conv_ctest_bwdwrw_fp32_218, 
conv_ctest_bwdwrw_fp32_219, conv_ctest_bwdwrw_fp32_220, 
conv_ctest_bwdwrw_fp32_221, conv_ctest_bwdwrw_fp32_222, 
conv_ctest_bwdwrw_fp32_223, conv_ctest_bwdwrw_fp32_224, 
conv_ctest_bwdwrw_fp32_225, conv_ctest_bwdwrw_fp32_226, 
conv_ctest_bwdwrw_fp32_227, conv_ctest_bwdwrw_fp32_228, 
conv_ctest_bwdwrw_fp32_229, conv_ctest_bwdwrw_fp32_230, 
conv_ctest_bwdwrw_fp32_231, conv_ctest_bwdwrw_fp32_232, 
conv_ctest_bwdwrw_fp32_233, conv_ctest_bwdwrw_fp32_234, 
conv_ctest_bwdwrw_fp32_235, conv_ctest_bwdwrw_fp32_236, 
conv_ctest_bwdwrw_fp32_237, conv_ctest_bwdwrw_fp32_238, 
conv_ctest_bwdwrw_fp32_239, conv_ctest_bwdwrw_fp32_240, 
conv_ctest_bwdwrw_fp32_241, conv_ctest_bwdwrw_fp32_242, 
conv_ctest_bwdwrw_fp32_243, conv_ctest_bwdwrw_fp32_244, 
conv_ctest_bwdwrw_fp32_245, conv_ctest_bwdwrw_fp32_246, 
conv_ctest_bwdwrw_fp32_247, conv_ctest_bwdwrw_fp32_248, 
conv_ctest_bwdwrw_fp32_249, conv_ctest_bwdwrw_fp32_250, 
conv_ctest_bwdwrw_fp32_251, conv_ctest_bwdwrw_fp32_252, 
conv_ctest_bwdwrw_fp32_253, conv_ctest_bwdwrw_fp32_254, 
conv_ctest_bwdwrw_fp32_255, conv_ctest_bwdwrw_fp32_256, 
conv_ctest_bwdwrw_fp32_257, conv_ctest_bwdwrw_fp32_258, 
conv_ctest_bwdwrw_fp32_259, conv_ctest_bwdwrw_fp32_260, 
conv_ctest_bwdwrw_fp32_261, conv_ctest_bwdwrw_fp32_262, 
conv_ctest_bwdwrw_fp32_263, conv_ctest_bwdwrw_fp32_264, 
conv_ctest_bwdwrw_fp32_265, conv_ctest_bwdwrw_fp32_266, 
conv_ctest_bwdwrw_fp32_267, conv_ctest_bwdwrw_fp32_268, 
conv_ctest_bwdwrw_fp32_269, conv_ctest_bwdwrw_fp32_270, 
conv_ctest_bwdwrw_fp32_271, conv_ctest_bwdwrw_fp32_272, 
conv_ctest_bwdwrw_fp32_273, conv_ctest_bwdwrw_fp32_274, 
conv_ctest_bwdwrw_fp32_275, conv_ctest_bwdwrw_fp32_276, 
conv_ctest_bwdwrw_fp32_277, conv_ctest_bwdwrw_fp32_278, 
conv_ctest_bwdwrw_fp32_279, conv_ctest_bwdwrw_fp32_280, 
conv_ctest_bwdwrw_fp32_281, conv_ctest_bwdwrw_fp32_282, 
conv_ctest_bwdwrw_fp32_283, conv_ctest_bwdwrw_fp32_284, 
conv_ctest_bwdwrw_fp32_285, conv_ctest_bwdwrw_fp32_286, 
conv_ctest_bwdwrw_fp32_287, conv_ctest_bwdwrw_fp32_288, 
conv_ctest_bwdwrw_fp32_289, conv_ctest_bwdwrw_fp32_290, 
conv_ctest_bwdwrw_fp32_291, conv_ctest_bwdwrw_fp32_292, 
conv_ctest_bwdwrw_fp32_293, conv_ctest_bwdwrw_fp32_294, 
conv_ctest_bwdwrw_fp32_295, conv_ctest_bwdwrw_fp32_296, 
conv_ctest_bwdwrw_fp32_297, conv_ctest_bwdwrw_fp32_298, 
conv_ctest_bwdwrw_fp32_299, conv_ctest_bwdwrw_fp32_300, 
conv_ctest_bwdwrw_fp32_301, conv_ctest_bwdwrw_fp32_302, 
conv_ctest_bwdwrw_fp32_303, conv_ctest_bwdwrw_fp32_304, 
conv_ctest_bwdwrw_fp32_305, conv_ctest_bwdwrw_fp32_306, 
conv_ctest_bwdwrw_fp32_307, conv_ctest_bwdwrw_fp32_308, 
conv_ctest_bwdwrw_fp32_309, conv_ctest_bwdwrw_fp32_310, 
conv_ctest_bwdwrw_fp32_311, conv_ctest_bwdwrw_fp32_312, 
conv_ctest_bwdwrw_fp32_313, conv_ctest_bwdwrw_fp32_314, 
conv_ctest_bwdwrw_fp32_315, conv_ctest_bwdwrw_fp32_316, 
conv_ctest_bwdwrw_fp32_317, conv_ctest_bwdwrw_fp32_318, 
conv_ctest_bwdwrw_fp32_319, conv_ctest_bwdwrw_fp32_320, 
conv_ctest_bwdwrw_fp32_321, conv_ctest_bwdwrw_fp32_322, 
conv_ctest_bwdwrw_fp32_323, conv_ctest_bwdwrw_fp32_324, 
conv_ctest_bwdwrw_fp32_325, conv_ctest_bwdwrw_fp32_326, 
conv_ctest_bwdwrw_fp32_327, conv_ctest_bwdwrw_fp32_328, 
conv_ctest_bwdwrw_fp32_329, conv_ctest_bwdwrw_fp32_330, 
conv_ctest_bwdwrw_fp32_331, conv_ctest_bwdwrw_fp32_332, 
conv_ctest_bwdwrw_fp32_333, conv_ctest_bwdwrw_fp32_334, 
conv_ctest_bwdwrw_fp32_335, conv_ctest_bwdwrw_fp32_336, 
conv_ctest_bwdwrw_fp32_337, conv_ctest_bwdwrw_fp32_338, 
conv_ctest_bwdwrw_fp32_339, conv_ctest_bwdwrw_fp32_340, 
conv_ctest_bwdwrw_fp32_341, conv_ctest_bwdwrw_fp32_342, 
conv_ctest_bwdwrw_fp32_343, conv_ctest_bwdwrw_fp32_344, 
conv_ctest_bwdwrw_fp32_345, conv_ctest_bwdwrw_fp32_346, 
conv_ctest_bwdwrw_fp32_347, conv_ctest_bwdwrw_fp32_348, 
conv_ctest_bwdwrw_fp32_349, conv_ctest_bwdwrw_fp32_350, 
conv_ctest_bwdwrw_fp32_351, conv_ctest_bwdwrw_fp32_352, 
conv_ctest_bwdwrw_fp32_353, conv_ctest_bwdwrw_fp32_354, 
conv_ctest_bwdwrw_fp32_355, conv_ctest_bwdwrw_fp32_356, 
conv_ctest_bwdwrw_fp32_357, conv_ctest_bwdwrw_fp32_358, 
conv_ctest_bwdwrw_fp32_359, conv_ctest_bwdwrw_fp32_360, 
conv_ctest_bwdwrw_fp32_361, conv_ctest_bwdwrw_fp32_362, 
conv_ctest_bwdwrw_fp32_363, conv_ctest_bwdwrw_fp32_364, 
conv_ctest_bwdwrw_fp32_365, conv_ctest_bwdwrw_fp32_366, 
conv_ctest_bwdwrw_fp32_367, conv_ctest_bwdwrw_fp32_368, 
conv_ctest_bwdwrw_fp32_369, conv_ctest_bwdwrw_fp32_370, 
conv_ctest_bwdwrw_fp32_371, conv_ctest_bwdwrw_fp32_372, 
conv_ctest_bwdwrw_fp32_373, conv_ctest_bwdwrw_fp32_374, 
conv_ctest_bwdwrw_fp32_375, conv_ctest_bwdwrw_fp32_376, 
conv_ctest_bwdwrw_fp32_377, conv_ctest_bwdwrw_fp32_378, 
conv_ctest_bwdwrw_fp32_379, conv_ctest_bwdwrw_fp32_380, 
conv_ctest_bwdwrw_fp32_381, conv_ctest_bwdwrw_fp32_382, 
conv_ctest_bwdwrw_fp32_383, conv_ctest_bwdwrw_fp32_384, 
conv_ctest_bwdwrw_fp32_385, conv_ctest_bwdwrw_fp32_386, 
conv_ctest_bwdwrw_fp32_387, conv_ctest_bwdwrw_fp32_388, 
conv_ctest_bwdwrw_fp32_389, conv_ctest_bwdwrw_fp32_390, 
conv_ctest_bwdwrw_fp32_391, conv_ctest_bwdwrw_fp32_392, 
conv_ctest_bwdwrw_fp32_393, conv_ctest_bwdwrw_fp32_394, 
conv_ctest_bwdwrw_fp32_395, conv_ctest_bwdwrw_fp32_396, 
conv_ctest_bwdwrw_fp32_397, conv_ctest_bwdwrw_fp32_398, 
conv_ctest_bwdwrw_fp32_399, conv_ctest_bwdwrw_fp32_400, 
conv_ctest_bwdwrw_fp32_401, conv_ctest_bwdwrw_fp32_402, 
conv_ctest_bwdwrw_fp32_403, conv_ctest_bwdwrw_fp32_404, 
conv_ctest_bwdwrw_fp32_405, conv_ctest_bwdwrw_fp32_406, 
conv_ctest_bwdwrw_fp32_407, conv_ctest_bwdwrw_fp32_408, 
conv_ctest_bwdwrw_fp32_409, conv_ctest_bwdwrw_fp32_410, 
conv_ctest_bwdwrw_fp32_411, conv_ctest_bwdwrw_fp32_412, 
conv_ctest_bwdwrw_fp32_413, conv_ctest_bwdwrw_fp32_414, 
conv_ctest_bwdwrw_fp32_415, conv_ctest_bwdwrw_fp32_416, 
conv_ctest_bwdwrw_fp32_417, conv_ctest_bwdwrw_fp32_418, 
conv_ctest_bwdwrw_fp32_419, conv_ctest_bwdwrw_fp32_420, 
conv_ctest_bwdwrw_fp32_421, conv_ctest_bwdwrw_fp32_422, 
conv_ctest_bwdwrw_fp32_423, conv_ctest_bwdwrw_fp32_424, 
conv_ctest_bwdwrw_fp32_425, conv_ctest_bwdwrw_fp32_426, 
conv_ctest_bwdwrw_fp32_427, conv_ctest_bwdwrw_fp32_428, 
conv_ctest_bwdwrw_fp32_429, conv_ctest_bwdwrw_fp32_430, 
conv_ctest_bwdwrw_fp32_431, conv_ctest_bwdwrw_fp32_432, 
conv_ctest_bwdwrw_fp32_433, conv_ctest_bwdwrw_fp32_434, 
conv_ctest_bwdwrw_fp32_435, conv_ctest_bwdwrw_fp32_436, 
conv_ctest_bwdwrw_fp32_437, conv_ctest_bwdwrw_fp32_438, 
conv_ctest_bwdwrw_fp32_439, conv_ctest_bwdwrw_fp32_440, 
conv_ctest_bwdwrw_fp32_441, conv_ctest_bwdwrw_fp32_442, 
conv_ctest_bwdwrw_fp32_443, conv_ctest_bwdwrw_fp32_444, 
conv_ctest_bwdwrw_fp32_445, conv_ctest_bwdwrw_fp32_446, 
conv_ctest_bwdwrw_fp32_447, conv_ctest_bwdwrw_fp32_448, 
conv_ctest_bwdwrw_fp32_449, conv_ctest_bwdwrw_fp32_450, 
conv_ctest_bwdwrw_fp32_451, conv_ctest_bwdwrw_fp32_452, 
conv_ctest_bwdwrw_fp32_453, conv_ctest_bwdwrw_fp32_454, 
conv_ctest_bwdwrw_fp32_455, conv_ctest_bwdwrw_fp32_456, 
conv_ctest_bwdwrw_fp32_457, conv_ctest_bwdwrw_fp32_458, 
conv_ctest_bwdwrw_fp32_459, conv_ctest_bwdwrw_fp32_460, 
conv_ctest_bwdwrw_fp32_461, conv_ctest_bwdwrw_fp32_462, 
conv_ctest_bwdwrw_fp32_463, conv_ctest_bwdwrw_fp32_464, 
conv_ctest_bwdwrw_fp32_465, conv_ctest_bwdwrw_fp32_466, 
conv_ctest_bwdwrw_fp32_467, conv_ctest_bwdwrw_fp32_468, 
conv_ctest_bwdwrw_fp32_469, conv_ctest_bwdwrw_fp32_470, 
conv_ctest_bwdwrw_fp32_471, conv_ctest_bwdwrw_fp32_472, 
conv_ctest_bwdwrw_fp32_473, conv_ctest_bwdwrw_fp32_474, 
conv_ctest_bwdwrw_fp32_475, conv_ctest_bwdwrw_fp32_476, 
conv_ctest_bwdwrw_fp32_477, conv_ctest_bwdwrw_fp32_478, 
conv_ctest_bwdwrw_fp32_479, conv_ctest_bwdwrw_fp32_480, 
conv_ctest_bwdwrw_fp32_481, conv_ctest_bwdwrw_fp32_482, 
conv_ctest_bwdwrw_fp32_483, conv_ctest_bwdwrw_fp32_484, 
conv_ctest_bwdwrw_fp32_485, conv_ctest_bwdwrw_fp32_486, 
conv_ctest_bwdwrw_fp32_487, conv_ctest_bwdwrw_fp32_488, 
conv_ctest_bwdwrw_fp32_489, conv_ctest_bwdwrw_fp32_490, 
conv_ctest_bwdwrw_fp32_491, conv_ctest_bwdwrw_fp32_492, 
conv_ctest_bwdwrw_fp32_493, conv_ctest_bwdwrw_fp32_494, 
conv_ctest_bwdwrw_fp32_495, conv_ctest_bwdwrw_fp32_496, 
conv_ctest_bwdwrw_fp32_497, conv_ctest_bwdwrw_fp32_498, 
conv_ctest_bwdwrw_fp32_499, conv_ctest_bwdwrw_fp32_500, 
conv_ctest_bwdwrw_fp32_501, conv_ctest_bwdwrw_fp32_502, 
conv_ctest_bwdwrw_fp32_503, conv_ctest_bwdwrw_fp32_504, 
conv_ctest_bwdwrw_fp32_505, conv_ctest_bwdwrw_fp32_506, 
conv_ctest_bwdwrw_fp32_507, conv_ctest_bwdwrw_fp32_508, 
conv_ctest_bwdwrw_fp32_509, conv_ctest_bwdwrw_fp32_510, 
conv_ctest_bwdwrw_fp32_511, conv_ctest_bwdwrw_fp32_512, 
conv_ctest_bwdwrw_fp32_513, conv_ctest_bwdwrw_fp32_514, 
conv_ctest_bwdwrw_fp32_515, conv_ctest_bwdwrw_fp32_516, 
conv_ctest_bwdwrw_fp32_517, conv_ctest_bwdwrw_fp32_518, 
conv_ctest_bwdwrw_fp32_519, conv_ctest_bwdwrw_fp32_520, 
conv_ctest_bwdwrw_fp32_521, conv_ctest_bwdwrw_fp32_522, 
conv_ctest_bwdwrw_fp32_523, conv_ctest_bwdwrw_fp32_524, 
conv_ctest_bwdwrw_fp32_525, conv_ctest_bwdwrw_fp32_526, 
conv_ctest_bwdwrw_fp32_527, conv_ctest_bwdwrw_fp32_528, 
conv_ctest_bwdwrw_fp32_529, conv_ctest_bwdwrw_fp32_530, 
conv_ctest_bwdwrw_fp32_531, conv_ctest_bwdwrw_fp32_532, 
conv_ctest_bwdwrw_fp32_533, conv_ctest_bwdwrw_fp32_534, 
conv_ctest_bwdwrw_fp32_535, conv_ctest_bwdwrw_fp32_536, 
conv_ctest_bwdwrw_fp32_537, conv_ctest_bwdwrw_fp32_538, 
conv_ctest_bwdwrw_fp32_539, conv_ctest_bwdwrw_fp32_540, 
conv_ctest_bwdwrw_fp32_541, conv_ctest_bwdwrw_fp32_542, 
conv_ctest_bwdwrw_fp32_543, conv_ctest_bwdwrw_fp32_544, 
conv_ctest_bwdwrw_fp32_545, conv_ctest_bwdwrw_fp32_546, 
conv_ctest_bwdwrw_fp32_547, conv_ctest_bwdwrw_fp32_548, 
conv_ctest_bwdwrw_fp32_549, conv_ctest_bwdwrw_fp32_550, 
conv_ctest_bwdwrw_fp32_551, conv_ctest_bwdwrw_fp32_552, 
conv_ctest_bwdwrw_fp32_553, conv_ctest_bwdwrw_fp32_554, 
conv_ctest_bwdwrw_fp32_555, conv_ctest_bwdwrw_fp32_556, 
conv_ctest_bwdwrw_fp32_557, conv_ctest_bwdwrw_fp32_558, 
conv_ctest_bwdwrw_fp32_559, conv_ctest_bwdwrw_fp32_560, 
conv_ctest_bwdwrw_fp32_561, conv_ctest_bwdwrw_fp32_562, 
conv_ctest_bwdwrw_fp32_563, conv_ctest_bwdwrw_fp32_564, 
conv_ctest_bwdwrw_fp32_565, conv_ctest_bwdwrw_fp32_566, 
conv_ctest_bwdwrw_fp32_567, conv_ctest_bwdwrw_fp32_568, 
conv_ctest_bwdwrw_fp32_569, conv_ctest_bwdwrw_fp32_570, 
conv_ctest_bwdwrw_fp32_571, conv_ctest_bwdwrw_fp32_572, 
conv_ctest_bwdwrw_fp32_573, conv_ctest_bwdwrw_fp32_574, 
conv_ctest_bwdwrw_fp32_575, conv_ctest_bwdwrw_fp32_576, 
conv_ctest_bwdwrw_fp32_577, conv_ctest_bwdwrw_fp32_578, 
conv_ctest_bwdwrw_fp32_579, conv_ctest_bwdwrw_fp32_580, 
conv_ctest_bwdwrw_fp32_581, conv_ctest_bwdwrw_fp32_582, 
conv_ctest_bwdwrw_fp32_583, conv_ctest_bwdwrw_fp32_584, 
conv_ctest_bwdwrw_fp32_585, conv_ctest_bwdwrw_fp32_586, 
conv_ctest_bwdwrw_fp32_587, conv_ctest_bwdwrw_fp32_588, 
conv_ctest_bwdwrw_fp32_589, conv_ctest_bwdwrw_fp32_590, 
conv_ctest_bwdwrw_fp32_591, conv_ctest_bwdwrw_fp32_592, 
conv_ctest_bwdwrw_fp32_593, conv_ctest_bwdwrw_fp32_594, 
conv_ctest_bwdwrw_fp32_595, conv_ctest_bwdwrw_fp32_596, 
conv_ctest_bwdwrw_fp32_597, conv_ctest_bwdwrw_fp32_598, 
conv_ctest_bwdwrw_fp32_599, conv_ctest_bwdwrw_fp32_600, 
conv_ctest_bwdwrw_fp32_601, conv_ctest_bwdwrw_fp32_602, 
conv_ctest_bwdwrw_fp32_603, conv_ctest_bwdwrw_fp32_604, 
conv_ctest_bwdwrw_fp32_605, conv_ctest_bwdwrw_fp32_606, 
conv_ctest_bwdwrw_fp32_607, conv_ctest_bwdwrw_fp32_608, 
conv_ctest_bwdwrw_fp32_609, conv_ctest_bwdwrw_fp32_610, 
conv_ctest_bwdwrw_fp32_611, conv_ctest_bwdwrw_fp32_612, 
conv_ctest_bwdwrw_fp32_613, conv_ctest_bwdwrw_fp32_614, 
conv_ctest_bwdwrw_fp32_615, conv_ctest_bwdwrw_fp32_616, 
conv_ctest_bwdwrw_fp32_617, conv_ctest_bwdwrw_fp32_618, 
conv_ctest_bwdwrw_fp32_619, conv_ctest_bwdwrw_fp32_620, 
conv_ctest_bwdwrw_fp32_621, conv_ctest_bwdwrw_fp32_622, 
conv_ctest_bwdwrw_fp32_623, conv_ctest_bwdwrw_fp32_624, 
conv_ctest_bwdwrw_fp32_625, conv_ctest_bwdwrw_fp32_626, 
conv_ctest_bwdwrw_fp32_627, conv_ctest_bwdwrw_fp32_628, 
conv_ctest_bwdwrw_fp32_629, 
};


gemm_tuple conv_ctest_bwdwrw_fp16_001 {{1008, 1, 100, 100, 100, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_002 {{1008, 1, 144, 144, 144, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_003 {{1008, 1, 196, 196, 196, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_004 {{1008, 1, 256, 256, 256, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_005 {{1008, 1, 25, 25, 25, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_006 {{1008, 1, 36, 36, 36, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_007 {{1008, 1, 49, 49, 49, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_008 {{1008, 1, 81, 81, 81, 1008}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_009 {{1024, 1, 121, 121, 121, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_010 {{1024, 1, 144, 144, 144, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_011 {{1024, 1, 16, 16, 16, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_012 {{1024, 1, 196, 196, 196, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_013 {{1024, 1, 256, 256, 256, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_014 {{1024, 1, 25, 25, 25, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_015 {{1024, 1, 36, 36, 36, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_016 {{1024, 1, 49, 49, 49, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_017 {{1024, 1, 81, 81, 81, 1024}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_018 {{1056, 1, 121, 121, 121, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_019 {{1056, 1, 16, 16, 16, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_020 {{1056, 1, 25, 25, 25, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_021 {{1056, 1, 49, 49, 49, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_022 {{1056, 1, 81, 81, 81, 1056}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_023 {{1152, 1, 100, 100, 100, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_024 {{1152, 1, 144, 144, 144, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_025 {{1152, 1, 169, 169, 169, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_026 {{1152, 1, 196, 196, 196, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_027 {{1152, 1, 256, 256, 256, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_028 {{1152, 1, 25, 25, 25, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_029 {{1152, 1, 2704, 2704, 2704, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_030 {{1152, 1, 2916, 2916, 2916, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_031 {{1152, 1, 3136, 3136, 3136, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_032 {{1152, 1, 3364, 3364, 3364, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_033 {{1152, 1, 36, 36, 36, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_034 {{1152, 1, 49, 49, 49, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_035 {{1152, 1, 576, 576, 576, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_036 {{1152, 1, 676, 676, 676, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_037 {{1152, 1, 729, 729, 729, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_038 {{1152, 1, 784, 784, 784, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_039 {{1152, 1, 81, 81, 81, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_040 {{1152, 1, 900, 900, 900, 1152}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_041 {{1200, 1, 16, 16, 16, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_042 {{1200, 1, 1, 1, 1, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_043 {{1200, 1, 25, 25, 25, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_044 {{1200, 1, 49, 49, 49, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_045 {{1200, 1, 4, 4, 4, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_046 {{1200, 1, 9, 9, 9, 1200}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_047 {{128, 1, 100, 100, 100, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_048 {{128, 1, 1024, 1024, 1024, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_049 {{128, 1, 196, 196, 196, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_050 {{128, 1, 225, 225, 225, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_051 {{128, 1, 256, 256, 256, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_052 {{128, 1, 289, 289, 289, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_053 {{128, 1, 3136, 3136, 3136, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_054 {{128, 1, 324, 324, 324, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_055 {{128, 1, 3364, 3364, 3364, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_056 {{128, 1, 3600, 3600, 3600, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_057 {{128, 1, 49, 49, 49, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_058 {{128, 1, 64, 64, 64, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_059 {{128, 1, 784, 784, 784, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_060 {{128, 1, 841, 841, 841, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_061 {{128, 1, 900, 900, 900, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_062 {{128, 1, 961, 961, 961, 128}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_063 {{1296, 1, 100, 100, 100, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_064 {{1296, 1, 144, 144, 144, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_065 {{1296, 1, 196, 196, 196, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_066 {{1296, 1, 256, 256, 256, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_067 {{1296, 1, 25, 25, 25, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_068 {{1296, 1, 36, 36, 36, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_069 {{1296, 1, 49, 49, 49, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_070 {{1296, 1, 81, 81, 81, 1296}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_071 {{1440, 1, 100, 100, 100, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_072 {{1440, 1, 144, 144, 144, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_073 {{1440, 1, 16, 16, 16, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_074 {{1440, 1, 196, 196, 196, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_075 {{1440, 1, 256, 256, 256, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_076 {{1440, 1, 25, 25, 25, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_077 {{1440, 1, 36, 36, 36, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_078 {{1440, 1, 49, 49, 49, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_079 {{1440, 1, 4, 4, 4, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_080 {{1440, 1, 81, 81, 81, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_081 {{1440, 1, 9, 9, 9, 1440}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_082 {{147, 1, 1024, 1024, 1024, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_083 {{147, 1, 10609, 10609, 10609, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_084 {{147, 1, 10816, 10816, 10816, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_085 {{147, 1, 11025, 11025, 11025, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_086 {{147, 1, 11236, 11236, 11236, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_087 {{147, 1, 11449, 11449, 11449, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_088 {{147, 1, 11664, 11664, 11664, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_089 {{147, 1, 11881, 11881, 11881, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_090 {{147, 1, 12100, 12100, 12100, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_091 {{147, 1, 12321, 12321, 12321, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_092 {{147, 1, 12544, 12544, 12544, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_093 {{147, 1, 12769, 12769, 12769, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_094 {{147, 1, 12996, 12996, 12996, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_095 {{147, 1, 13456, 13456, 13456, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_096 {{147, 1, 169, 169, 169, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_097 {{147, 1, 196, 196, 196, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_098 {{147, 1, 256, 256, 256, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_099 {{147, 1, 400, 400, 400, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_100 {{147, 1, 44944, 44944, 44944, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_101 {{147, 1, 46225, 46225, 46225, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_102 {{147, 1, 47524, 47524, 47524, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_103 {{147, 1, 47961, 47961, 47961, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_104 {{147, 1, 48400, 48400, 48400, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_105 {{147, 1, 48841, 48841, 48841, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_106 {{147, 1, 49284, 49284, 49284, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_107 {{147, 1, 49729, 49729, 49729, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_108 {{147, 1, 49, 49, 49, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_109 {{147, 1, 50176, 50176, 50176, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_110 {{147, 1, 50625, 50625, 50625, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_111 {{147, 1, 51529, 51529, 51529, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_112 {{147, 1, 52441, 52441, 52441, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_113 {{147, 1, 53361, 53361, 53361, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_114 {{147, 1, 64, 64, 64, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_115 {{147, 1, 676, 676, 676, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_116 {{147, 1, 784, 784, 784, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_117 {{147, 1, 900, 900, 900, 147}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_118 {{1600, 1, 100, 100, 100, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_119 {{1600, 1, 10816, 10816, 10816, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_120 {{1600, 1, 11664, 11664, 11664, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_121 {{1600, 1, 12100, 12100, 12100, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_122 {{1600, 1, 12544, 12544, 12544, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_123 {{1600, 1, 144, 144, 144, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_124 {{1600, 1, 169, 169, 169, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_125 {{1600, 1, 196, 196, 196, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_126 {{1600, 1, 225, 225, 225, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_127 {{1600, 1, 2304, 2304, 2304, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_128 {{1600, 1, 25, 25, 25, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_129 {{1600, 1, 2601, 2601, 2601, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_130 {{1600, 1, 2704, 2704, 2704, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_131 {{1600, 1, 2916, 2916, 2916, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_132 {{1600, 1, 3025, 3025, 3025, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_133 {{1600, 1, 3136, 3136, 3136, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_134 {{1600, 1, 3249, 3249, 3249, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_135 {{1600, 1, 361, 361, 361, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_136 {{1600, 1, 36, 36, 36, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_137 {{1600, 1, 400, 400, 400, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_138 {{1600, 1, 49, 49, 49, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_139 {{1600, 1, 4, 4, 4, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_140 {{1600, 1, 529, 529, 529, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_141 {{1600, 1, 576, 576, 576, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_142 {{1600, 1, 625, 625, 625, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_143 {{1600, 1, 64, 64, 64, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_144 {{1600, 1, 676, 676, 676, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_145 {{1600, 1, 729, 729, 729, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_146 {{1600, 1, 784, 784, 784, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_147 {{1600, 1, 81, 81, 81, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_148 {{1600, 1, 841, 841, 841, 1600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_149 {{1728, 1, 100, 100, 100, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_150 {{1728, 1, 144, 144, 144, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_151 {{1728, 1, 169, 169, 169, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_152 {{1728, 1, 16, 16, 16, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_153 {{1728, 1, 196, 196, 196, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_154 {{1728, 1, 256, 256, 256, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_155 {{1728, 1, 25, 25, 25, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_156 {{1728, 1, 36, 36, 36, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_157 {{1728, 1, 49, 49, 49, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_158 {{1728, 1, 4, 4, 4, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_159 {{1728, 1, 576, 576, 576, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_160 {{1728, 1, 676, 676, 676, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_161 {{1728, 1, 784, 784, 784, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_162 {{1728, 1, 81, 81, 81, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_163 {{1728, 1, 900, 900, 900, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_164 {{1728, 1, 9, 9, 9, 1728}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_165 {{192, 1, 100, 100, 100, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_166 {{192, 1, 1024, 1024, 1024, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_167 {{192, 1, 121, 121, 121, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_168 {{192, 1, 16, 16, 16, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_169 {{192, 1, 196, 196, 196, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_170 {{192, 1, 225, 225, 225, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_171 {{192, 1, 256, 256, 256, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_172 {{192, 1, 25, 25, 25, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_173 {{192, 1, 289, 289, 289, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_174 {{192, 1, 324, 324, 324, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_175 {{192, 1, 49, 49, 49, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_176 {{192, 1, 64, 64, 64, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_177 {{192, 1, 784, 784, 784, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_178 {{192, 1, 81, 81, 81, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_179 {{192, 1, 900, 900, 900, 192}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_180 {{2016, 1, 16, 16, 16, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_181 {{2016, 1, 25, 25, 25, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_182 {{2016, 1, 36, 36, 36, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_183 {{2016, 1, 49, 49, 49, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_184 {{2016, 1, 4, 4, 4, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_185 {{2016, 1, 81, 81, 81, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_186 {{2016, 1, 9, 9, 9, 2016}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_187 {{2048, 1, 121, 121, 121, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_188 {{2048, 1, 169, 169, 169, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_189 {{2048, 1, 225, 225, 225, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_190 {{2048, 1, 36, 36, 36, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_191 {{2048, 1, 49, 49, 49, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_192 {{2048, 1, 81, 81, 81, 2048}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_193 {{2304, 1, 100, 100, 100, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_194 {{2304, 1, 121, 121, 121, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_195 {{2304, 1, 144, 144, 144, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_196 {{2304, 1, 169, 169, 169, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_197 {{2304, 1, 16, 16, 16, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_198 {{2304, 1, 196, 196, 196, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_199 {{2304, 1, 225, 225, 225, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_200 {{2304, 1, 256, 256, 256, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_201 {{2304, 1, 25, 25, 25, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_202 {{2304, 1, 2704, 2704, 2704, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_203 {{2304, 1, 2916, 2916, 2916, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_204 {{2304, 1, 3136, 3136, 3136, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_205 {{2304, 1, 3364, 3364, 3364, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_206 {{2304, 1, 36, 36, 36, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_207 {{2304, 1, 49, 49, 49, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_208 {{2304, 1, 576, 576, 576, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_209 {{2304, 1, 64, 64, 64, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_210 {{2304, 1, 676, 676, 676, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_211 {{2304, 1, 729, 729, 729, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_212 {{2304, 1, 784, 784, 784, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_213 {{2304, 1, 81, 81, 81, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_214 {{2304, 1, 900, 900, 900, 2304}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_215 {{2400, 1, 100, 100, 100, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_216 {{2400, 1, 144, 144, 144, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_217 {{2400, 1, 169, 169, 169, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_218 {{2400, 1, 196, 196, 196, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_219 {{2400, 1, 225, 225, 225, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_220 {{2400, 1, 25, 25, 25, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_221 {{2400, 1, 361, 361, 361, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_222 {{2400, 1, 36, 36, 36, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_223 {{2400, 1, 400, 400, 400, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_224 {{2400, 1, 49, 49, 49, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_225 {{2400, 1, 4, 4, 4, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_226 {{2400, 1, 529, 529, 529, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_227 {{2400, 1, 576, 576, 576, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_228 {{2400, 1, 625, 625, 625, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_229 {{2400, 1, 64, 64, 64, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_230 {{2400, 1, 676, 676, 676, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_231 {{2400, 1, 729, 729, 729, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_232 {{2400, 1, 784, 784, 784, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_233 {{2400, 1, 81, 81, 81, 2400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_234 {{256, 1, 100, 100, 100, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_235 {{256, 1, 1024, 1024, 1024, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_236 {{256, 1, 144, 144, 144, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_237 {{256, 1, 169, 169, 169, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_238 {{256, 1, 196, 196, 196, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_239 {{256, 1, 225, 225, 225, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_240 {{256, 1, 256, 256, 256, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_241 {{256, 1, 289, 289, 289, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_242 {{256, 1, 3136, 3136, 3136, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_243 {{256, 1, 324, 324, 324, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_244 {{256, 1, 3364, 3364, 3364, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_245 {{256, 1, 3600, 3600, 3600, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_246 {{256, 1, 36, 36, 36, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_247 {{256, 1, 49, 49, 49, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_248 {{256, 1, 64, 64, 64, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_249 {{256, 1, 784, 784, 784, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_250 {{256, 1, 81, 81, 81, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_251 {{256, 1, 841, 841, 841, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_252 {{256, 1, 900, 900, 900, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_253 {{256, 1, 961, 961, 961, 256}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_254 {{27, 1, 1024, 1024, 1024, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_255 {{27, 1, 1156, 1156, 1156, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_256 {{27, 1, 12100, 12100, 12100, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_257 {{27, 1, 12321, 12321, 12321, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_258 {{27, 1, 12544, 12544, 12544, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_259 {{27, 1, 12769, 12769, 12769, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_260 {{27, 1, 12996, 12996, 12996, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_261 {{27, 1, 13225, 13225, 13225, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_262 {{27, 1, 13456, 13456, 13456, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_263 {{27, 1, 13924, 13924, 13924, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_264 {{27, 1, 196, 196, 196, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_265 {{27, 1, 225, 225, 225, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_266 {{27, 1, 256, 256, 256, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_267 {{27, 1, 324, 324, 324, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_268 {{27, 1, 48400, 48400, 48400, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_269 {{27, 1, 49284, 49284, 49284, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_270 {{27, 1, 49729, 49729, 49729, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_271 {{27, 1, 50176, 50176, 50176, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_272 {{27, 1, 50625, 50625, 50625, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_273 {{27, 1, 51076, 51076, 51076, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_274 {{27, 1, 51529, 51529, 51529, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_275 {{27, 1, 52441, 52441, 52441, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_276 {{27, 1, 53361, 53361, 53361, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_277 {{27, 1, 54289, 54289, 54289, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_278 {{27, 1, 784, 784, 784, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_279 {{27, 1, 900, 900, 900, 27}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_280 {{320, 1, 1024, 1024, 1024, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_281 {{320, 1, 196, 196, 196, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_282 {{320, 1, 225, 225, 225, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_283 {{320, 1, 289, 289, 289, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_284 {{320, 1, 784, 784, 784, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_285 {{320, 1, 900, 900, 900, 320}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_286 {{3456, 1, 121, 121, 121, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_287 {{3456, 1, 169, 169, 169, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_288 {{3456, 1, 225, 225, 225, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_289 {{3456, 1, 25, 25, 25, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_290 {{3456, 1, 36, 36, 36, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_291 {{3456, 1, 49, 49, 49, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_292 {{3456, 1, 81, 81, 81, 3456}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_293 {{363, 1, 10000, 10000, 10000, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_294 {{363, 1, 1024, 1024, 1024, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_295 {{363, 1, 10404, 10404, 10404, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_296 {{363, 1, 11449, 11449, 11449, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_297 {{363, 1, 11664, 11664, 11664, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_298 {{363, 1, 11881, 11881, 11881, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_299 {{363, 1, 12100, 12100, 12100, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_300 {{363, 1, 121, 121, 121, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_301 {{363, 1, 12321, 12321, 12321, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_302 {{363, 1, 12544, 12544, 12544, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_303 {{363, 1, 12996, 12996, 12996, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_304 {{363, 1, 13456, 13456, 13456, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_305 {{363, 1, 144, 144, 144, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_306 {{363, 1, 196, 196, 196, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_307 {{363, 1, 1, 1, 1, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_308 {{363, 1, 256, 256, 256, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_309 {{363, 1, 41616, 41616, 41616, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_310 {{363, 1, 42849, 42849, 42849, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_311 {{363, 1, 44521, 44521, 44521, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_312 {{363, 1, 45796, 45796, 45796, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_313 {{363, 1, 46656, 46656, 46656, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_314 {{363, 1, 47089, 47089, 47089, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_315 {{363, 1, 47524, 47524, 47524, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_316 {{363, 1, 47961, 47961, 47961, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_317 {{363, 1, 484, 484, 484, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_318 {{363, 1, 48841, 48841, 48841, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_319 {{363, 1, 49729, 49729, 49729, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_320 {{363, 1, 4, 4, 4, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_321 {{363, 1, 50176, 50176, 50176, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_322 {{363, 1, 50625, 50625, 50625, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_323 {{363, 1, 51529, 51529, 51529, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_324 {{363, 1, 53361, 53361, 53361, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_325 {{363, 1, 576, 576, 576, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_326 {{363, 1, 676, 676, 676, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_327 {{363, 1, 9025, 9025, 9025, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_328 {{363, 1, 9409, 9409, 9409, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_329 {{363, 1, 9604, 9604, 9604, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_330 {{363, 1, 9801, 9801, 9801, 363}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_331 {{400, 1, 100, 100, 100, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_332 {{400, 1, 144, 144, 144, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_333 {{400, 1, 169, 169, 169, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_334 {{400, 1, 196, 196, 196, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_335 {{400, 1, 225, 225, 225, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_336 {{400, 1, 25, 25, 25, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_337 {{400, 1, 36, 36, 36, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_338 {{400, 1, 400, 400, 400, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_339 {{400, 1, 49, 49, 49, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_340 {{400, 1, 4, 4, 4, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_341 {{400, 1, 576, 576, 576, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_342 {{400, 1, 64, 64, 64, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_343 {{400, 1, 676, 676, 676, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_344 {{400, 1, 784, 784, 784, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_345 {{400, 1, 81, 81, 81, 400}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_346 {{4608, 1, 100, 100, 100, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_347 {{4608, 1, 144, 144, 144, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_348 {{4608, 1, 169, 169, 169, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_349 {{4608, 1, 16, 16, 16, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_350 {{4608, 1, 1860, 1860, 1860, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_351 {{4608, 1, 1953, 1953, 1953, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_352 {{4608, 1, 196, 196, 196, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_353 {{4608, 1, 1, 1, 1, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_354 {{4608, 1, 2048, 2048, 2048, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_355 {{4608, 1, 2244, 2244, 2244, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_356 {{4608, 1, 256, 256, 256, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_357 {{4608, 1, 25, 25, 25, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_358 {{4608, 1, 36, 36, 36, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_359 {{4608, 1, 49, 49, 49, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_360 {{4608, 1, 4, 4, 4, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_361 {{4608, 1, 576, 576, 576, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_362 {{4608, 1, 64, 64, 64, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_363 {{4608, 1, 676, 676, 676, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_364 {{4608, 1, 7440, 7440, 7440, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_365 {{4608, 1, 7812, 7812, 7812, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_366 {{4608, 1, 784, 784, 784, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_367 {{4608, 1, 8192, 8192, 8192, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_368 {{4608, 1, 81, 81, 81, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_369 {{4608, 1, 8580, 8580, 8580, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_370 {{4608, 1, 900, 900, 900, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_371 {{4608, 1, 9, 9, 9, 4608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_372 {{480, 1, 100, 100, 100, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_373 {{480, 1, 196, 196, 196, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_374 {{480, 1, 2048, 2048, 2048, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_375 {{480, 1, 2145, 2145, 2145, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_376 {{480, 1, 2345, 2345, 2345, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_377 {{480, 1, 256, 256, 256, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_378 {{480, 1, 324, 324, 324, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_379 {{480, 1, 32768, 32768, 32768, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_380 {{480, 1, 33540, 33540, 33540, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_381 {{480, 1, 34320, 34320, 34320, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_382 {{480, 1, 49, 49, 49, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_383 {{480, 1, 64, 64, 64, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_384 {{480, 1, 8192, 8192, 8192, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_385 {{480, 1, 8385, 8385, 8385, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_386 {{480, 1, 8580, 8580, 8580, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_387 {{480, 1, 8777, 8777, 8777, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_388 {{480, 1, 8976, 8976, 8976, 480}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_389 {{4, 1, 100, 100, 100, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_390 {{4, 1, 121, 121, 121, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_391 {{4, 1, 144, 144, 144, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_392 {{4, 1, 169, 169, 169, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_393 {{4, 1, 16, 16, 16, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_394 {{4, 1, 196, 196, 196, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_395 {{4, 1, 1, 1, 1, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_396 {{4, 1, 225, 225, 225, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_397 {{4, 1, 256, 256, 256, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_398 {{4, 1, 25, 25, 25, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_399 {{4, 1, 289, 289, 289, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_400 {{4, 1, 36, 36, 36, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_401 {{4, 1, 49, 49, 49, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_402 {{4, 1, 4, 4, 4, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_403 {{4, 1, 625, 625, 625, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_404 {{4, 1, 64, 64, 64, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_405 {{4, 1, 676, 676, 676, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_406 {{4, 1, 729, 729, 729, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_407 {{4, 1, 784, 784, 784, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_408 {{4, 1, 81, 81, 81, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_409 {{4, 1, 900, 900, 900, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_410 {{4, 1, 9, 9, 9, 4}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_411 {{512, 1, 100, 100, 100, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_412 {{512, 1, 1024, 1024, 1024, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_413 {{512, 1, 121, 121, 121, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_414 {{512, 1, 144, 144, 144, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_415 {{512, 1, 16, 16, 16, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_416 {{512, 1, 196, 196, 196, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_417 {{512, 1, 2048, 2048, 2048, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_418 {{512, 1, 2145, 2145, 2145, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_419 {{512, 1, 225, 225, 225, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_420 {{512, 1, 2345, 2345, 2345, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_421 {{512, 1, 256, 256, 256, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_422 {{512, 1, 25, 25, 25, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_423 {{512, 1, 289, 289, 289, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_424 {{512, 1, 324, 324, 324, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_425 {{512, 1, 36, 36, 36, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_426 {{512, 1, 49, 49, 49, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_427 {{512, 1, 4, 4, 4, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_428 {{512, 1, 64, 64, 64, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_429 {{512, 1, 784, 784, 784, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_430 {{512, 1, 8192, 8192, 8192, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_431 {{512, 1, 81, 81, 81, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_432 {{512, 1, 8580, 8580, 8580, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_433 {{512, 1, 8976, 8976, 8976, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_434 {{512, 1, 900, 900, 900, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_435 {{512, 1, 9, 9, 9, 512}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_436 {{528, 1, 100, 100, 100, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_437 {{528, 1, 16, 16, 16, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_438 {{528, 1, 196, 196, 196, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_439 {{528, 1, 2048, 2048, 2048, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_440 {{528, 1, 2145, 2145, 2145, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_441 {{528, 1, 2345, 2345, 2345, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_442 {{528, 1, 256, 256, 256, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_443 {{528, 1, 25, 25, 25, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_444 {{528, 1, 324, 324, 324, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_445 {{528, 1, 36, 36, 36, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_446 {{528, 1, 49, 49, 49, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_447 {{528, 1, 4, 4, 4, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_448 {{528, 1, 64, 64, 64, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_449 {{528, 1, 8192, 8192, 8192, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_450 {{528, 1, 8580, 8580, 8580, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_451 {{528, 1, 8976, 8976, 8976, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_452 {{528, 1, 9, 9, 9, 528}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_453 {{576, 1, 100, 100, 100, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_454 {{576, 1, 11664, 11664, 11664, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_455 {{576, 1, 12100, 12100, 12100, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_456 {{576, 1, 12544, 12544, 12544, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_457 {{576, 1, 12996, 12996, 12996, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_458 {{576, 1, 144, 144, 144, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_459 {{576, 1, 169, 169, 169, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_460 {{576, 1, 16, 16, 16, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_461 {{576, 1, 196, 196, 196, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_462 {{576, 1, 256, 256, 256, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_463 {{576, 1, 25, 25, 25, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_464 {{576, 1, 2704, 2704, 2704, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_465 {{576, 1, 2916, 2916, 2916, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_466 {{576, 1, 3025, 3025, 3025, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_467 {{576, 1, 3136, 3136, 3136, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_468 {{576, 1, 324, 324, 324, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_469 {{576, 1, 3364, 3364, 3364, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_470 {{576, 1, 36, 36, 36, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_471 {{576, 1, 49, 49, 49, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_472 {{576, 1, 4, 4, 4, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_473 {{576, 1, 529, 529, 529, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_474 {{576, 1, 576, 576, 576, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_475 {{576, 1, 625, 625, 625, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_476 {{576, 1, 64, 64, 64, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_477 {{576, 1, 676, 676, 676, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_478 {{576, 1, 729, 729, 729, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_479 {{576, 1, 784, 784, 784, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_480 {{576, 1, 81, 81, 81, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_481 {{576, 1, 841, 841, 841, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_482 {{576, 1, 900, 900, 900, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_483 {{576, 1, 9, 9, 9, 576}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_484 {{600, 1, 100, 100, 100, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_485 {{600, 1, 144, 144, 144, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_486 {{600, 1, 196, 196, 196, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_487 {{600, 1, 25, 25, 25, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_488 {{600, 1, 36, 36, 36, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_489 {{600, 1, 49, 49, 49, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_490 {{600, 1, 4, 4, 4, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_491 {{600, 1, 64, 64, 64, 600}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_492 {{608, 1, 100, 100, 100, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_493 {{608, 1, 16, 16, 16, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_494 {{608, 1, 196, 196, 196, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_495 {{608, 1, 256, 256, 256, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_496 {{608, 1, 25, 25, 25, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_497 {{608, 1, 324, 324, 324, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_498 {{608, 1, 36, 36, 36, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_499 {{608, 1, 49, 49, 49, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_500 {{608, 1, 4, 4, 4, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_501 {{608, 1, 64, 64, 64, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_502 {{608, 1, 9, 9, 9, 608}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_503 {{64, 1, 100, 100, 100, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_504 {{64, 1, 1024, 1024, 1024, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_505 {{64, 1, 12544, 12544, 12544, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_506 {{64, 1, 12996, 12996, 12996, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_507 {{64, 1, 13456, 13456, 13456, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_508 {{64, 1, 196, 196, 196, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_509 {{64, 1, 225, 225, 225, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_510 {{64, 1, 256, 256, 256, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_511 {{64, 1, 289, 289, 289, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_512 {{64, 1, 3136, 3136, 3136, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_513 {{64, 1, 3249, 3249, 3249, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_514 {{64, 1, 324, 324, 324, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_515 {{64, 1, 3364, 3364, 3364, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_516 {{64, 1, 3481, 3481, 3481, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_517 {{64, 1, 3600, 3600, 3600, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_518 {{64, 1, 49, 49, 49, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_519 {{64, 1, 64, 64, 64, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_520 {{64, 1, 729, 729, 729, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_521 {{64, 1, 784, 784, 784, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_522 {{64, 1, 841, 841, 841, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_523 {{64, 1, 900, 900, 900, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_524 {{64, 1, 961, 961, 961, 64}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_525 {{75, 1, 1024, 1024, 1024, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_526 {{75, 1, 11449, 11449, 11449, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_527 {{75, 1, 11881, 11881, 11881, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_528 {{75, 1, 12100, 12100, 12100, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_529 {{75, 1, 121, 121, 121, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_530 {{75, 1, 12321, 12321, 12321, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_531 {{75, 1, 12544, 12544, 12544, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_532 {{75, 1, 12769, 12769, 12769, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_533 {{75, 1, 12996, 12996, 12996, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_534 {{75, 1, 13225, 13225, 13225, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_535 {{75, 1, 13456, 13456, 13456, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_536 {{75, 1, 13689, 13689, 13689, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_537 {{75, 1, 196, 196, 196, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_538 {{75, 1, 225, 225, 225, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_539 {{75, 1, 256, 256, 256, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_540 {{75, 1, 289, 289, 289, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_541 {{75, 1, 46656, 46656, 46656, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_542 {{75, 1, 47961, 47961, 47961, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_543 {{75, 1, 48400, 48400, 48400, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_544 {{75, 1, 49284, 49284, 49284, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_545 {{75, 1, 49729, 49729, 49729, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_546 {{75, 1, 50176, 50176, 50176, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_547 {{75, 1, 50625, 50625, 50625, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_548 {{75, 1, 51529, 51529, 51529, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_549 {{75, 1, 52441, 52441, 52441, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_550 {{75, 1, 53361, 53361, 53361, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_551 {{75, 1, 576, 576, 576, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_552 {{75, 1, 784, 784, 784, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_553 {{75, 1, 900, 900, 900, 75}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_554 {{800, 1, 100, 100, 100, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_555 {{800, 1, 144, 144, 144, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_556 {{800, 1, 169, 169, 169, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_557 {{800, 1, 16, 16, 16, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_558 {{800, 1, 196, 196, 196, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_559 {{800, 1, 1, 1, 1, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_560 {{800, 1, 225, 225, 225, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_561 {{800, 1, 256, 256, 256, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_562 {{800, 1, 25, 25, 25, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_563 {{800, 1, 36, 36, 36, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_564 {{800, 1, 400, 400, 400, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_565 {{800, 1, 49, 49, 49, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_566 {{800, 1, 4, 4, 4, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_567 {{800, 1, 576, 576, 576, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_568 {{800, 1, 64, 64, 64, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_569 {{800, 1, 676, 676, 676, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_570 {{800, 1, 784, 784, 784, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_571 {{800, 1, 81, 81, 81, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_572 {{800, 1, 9, 9, 9, 800}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_573 {{832, 1, 121, 121, 121, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_574 {{832, 1, 16, 16, 16, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_575 {{832, 1, 2048, 2048, 2048, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_576 {{832, 1, 2145, 2145, 2145, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_577 {{832, 1, 2345, 2345, 2345, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_578 {{832, 1, 25, 25, 25, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_579 {{832, 1, 49, 49, 49, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_580 {{832, 1, 8192, 8192, 8192, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_581 {{832, 1, 81, 81, 81, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_582 {{832, 1, 8580, 8580, 8580, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_583 {{832, 1, 8976, 8976, 8976, 832}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_584 {{864, 1, 100, 100, 100, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_585 {{864, 1, 144, 144, 144, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_586 {{864, 1, 169, 169, 169, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_587 {{864, 1, 196, 196, 196, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_588 {{864, 1, 256, 256, 256, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_589 {{864, 1, 25, 25, 25, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_590 {{864, 1, 36, 36, 36, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_591 {{864, 1, 49, 49, 49, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_592 {{864, 1, 529, 529, 529, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_593 {{864, 1, 576, 576, 576, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_594 {{864, 1, 625, 625, 625, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_595 {{864, 1, 676, 676, 676, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_596 {{864, 1, 729, 729, 729, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_597 {{864, 1, 784, 784, 784, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_598 {{864, 1, 81, 81, 81, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_599 {{864, 1, 841, 841, 841, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_600 {{864, 1, 900, 900, 900, 864}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_601 {{9216, 1, 100, 100, 100, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_602 {{9216, 1, 144, 144, 144, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_603 {{9216, 1, 16, 16, 16, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_604 {{9216, 1, 196, 196, 196, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_605 {{9216, 1, 25, 25, 25, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_606 {{9216, 1, 36, 36, 36, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_607 {{9216, 1, 49, 49, 49, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_608 {{9216, 1, 4, 4, 4, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_609 {{9216, 1, 64, 64, 64, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_610 {{9216, 1, 81, 81, 81, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_611 {{9216, 1, 9, 9, 9, 9216}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_612 {{9, 1, 100, 100, 100, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_613 {{9, 1, 144, 144, 144, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_614 {{9, 1, 169, 169, 169, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_615 {{9, 1, 16, 16, 16, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_616 {{9, 1, 196, 196, 196, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_617 {{9, 1, 1, 1, 1, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_618 {{9, 1, 256, 256, 256, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_619 {{9, 1, 25, 25, 25, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_620 {{9, 1, 36, 36, 36, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_621 {{9, 1, 49, 49, 49, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_622 {{9, 1, 4, 4, 4, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_623 {{9, 1, 529, 529, 529, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_624 {{9, 1, 625, 625, 625, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_625 {{9, 1, 64, 64, 64, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_626 {{9, 1, 729, 729, 729, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_627 {{9, 1, 81, 81, 81, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_628 {{9, 1, 841, 841, 841, 9}, {1, 1}, {'T', 'N'}};
gemm_tuple conv_ctest_bwdwrw_fp16_629 {{9, 1, 9, 9, 9, 9}, {1, 1}, {'T', 'N'}};

const vector<gemm_tuple> conv_ctest_bwdwrw_fp16 = {
conv_ctest_bwdwrw_fp16_001, conv_ctest_bwdwrw_fp16_002, 
conv_ctest_bwdwrw_fp16_003, conv_ctest_bwdwrw_fp16_004, 
conv_ctest_bwdwrw_fp16_005, conv_ctest_bwdwrw_fp16_006, 
conv_ctest_bwdwrw_fp16_007, conv_ctest_bwdwrw_fp16_008, 
conv_ctest_bwdwrw_fp16_009, conv_ctest_bwdwrw_fp16_010, 
conv_ctest_bwdwrw_fp16_011, conv_ctest_bwdwrw_fp16_012, 
conv_ctest_bwdwrw_fp16_013, conv_ctest_bwdwrw_fp16_014, 
conv_ctest_bwdwrw_fp16_015, conv_ctest_bwdwrw_fp16_016, 
conv_ctest_bwdwrw_fp16_017, conv_ctest_bwdwrw_fp16_018, 
conv_ctest_bwdwrw_fp16_019, conv_ctest_bwdwrw_fp16_020, 
conv_ctest_bwdwrw_fp16_021, conv_ctest_bwdwrw_fp16_022, 
conv_ctest_bwdwrw_fp16_023, conv_ctest_bwdwrw_fp16_024, 
conv_ctest_bwdwrw_fp16_025, conv_ctest_bwdwrw_fp16_026, 
conv_ctest_bwdwrw_fp16_027, conv_ctest_bwdwrw_fp16_028, 
conv_ctest_bwdwrw_fp16_029, conv_ctest_bwdwrw_fp16_030, 
conv_ctest_bwdwrw_fp16_031, conv_ctest_bwdwrw_fp16_032, 
conv_ctest_bwdwrw_fp16_033, conv_ctest_bwdwrw_fp16_034, 
conv_ctest_bwdwrw_fp16_035, conv_ctest_bwdwrw_fp16_036, 
conv_ctest_bwdwrw_fp16_037, conv_ctest_bwdwrw_fp16_038, 
conv_ctest_bwdwrw_fp16_039, conv_ctest_bwdwrw_fp16_040, 
conv_ctest_bwdwrw_fp16_041, conv_ctest_bwdwrw_fp16_042, 
conv_ctest_bwdwrw_fp16_043, conv_ctest_bwdwrw_fp16_044, 
conv_ctest_bwdwrw_fp16_045, conv_ctest_bwdwrw_fp16_046, 
conv_ctest_bwdwrw_fp16_047, conv_ctest_bwdwrw_fp16_048, 
conv_ctest_bwdwrw_fp16_049, conv_ctest_bwdwrw_fp16_050, 
conv_ctest_bwdwrw_fp16_051, conv_ctest_bwdwrw_fp16_052, 
conv_ctest_bwdwrw_fp16_053, conv_ctest_bwdwrw_fp16_054, 
conv_ctest_bwdwrw_fp16_055, conv_ctest_bwdwrw_fp16_056, 
conv_ctest_bwdwrw_fp16_057, conv_ctest_bwdwrw_fp16_058, 
conv_ctest_bwdwrw_fp16_059, conv_ctest_bwdwrw_fp16_060, 
conv_ctest_bwdwrw_fp16_061, conv_ctest_bwdwrw_fp16_062, 
conv_ctest_bwdwrw_fp16_063, conv_ctest_bwdwrw_fp16_064, 
conv_ctest_bwdwrw_fp16_065, conv_ctest_bwdwrw_fp16_066, 
conv_ctest_bwdwrw_fp16_067, conv_ctest_bwdwrw_fp16_068, 
conv_ctest_bwdwrw_fp16_069, conv_ctest_bwdwrw_fp16_070, 
conv_ctest_bwdwrw_fp16_071, conv_ctest_bwdwrw_fp16_072, 
conv_ctest_bwdwrw_fp16_073, conv_ctest_bwdwrw_fp16_074, 
conv_ctest_bwdwrw_fp16_075, conv_ctest_bwdwrw_fp16_076, 
conv_ctest_bwdwrw_fp16_077, conv_ctest_bwdwrw_fp16_078, 
conv_ctest_bwdwrw_fp16_079, conv_ctest_bwdwrw_fp16_080, 
conv_ctest_bwdwrw_fp16_081, conv_ctest_bwdwrw_fp16_082, 
conv_ctest_bwdwrw_fp16_083, conv_ctest_bwdwrw_fp16_084, 
conv_ctest_bwdwrw_fp16_085, conv_ctest_bwdwrw_fp16_086, 
conv_ctest_bwdwrw_fp16_087, conv_ctest_bwdwrw_fp16_088, 
conv_ctest_bwdwrw_fp16_089, conv_ctest_bwdwrw_fp16_090, 
conv_ctest_bwdwrw_fp16_091, conv_ctest_bwdwrw_fp16_092, 
conv_ctest_bwdwrw_fp16_093, conv_ctest_bwdwrw_fp16_094, 
conv_ctest_bwdwrw_fp16_095, conv_ctest_bwdwrw_fp16_096, 
conv_ctest_bwdwrw_fp16_097, conv_ctest_bwdwrw_fp16_098, 
conv_ctest_bwdwrw_fp16_099, conv_ctest_bwdwrw_fp16_100, 
conv_ctest_bwdwrw_fp16_101, conv_ctest_bwdwrw_fp16_102, 
conv_ctest_bwdwrw_fp16_103, conv_ctest_bwdwrw_fp16_104, 
conv_ctest_bwdwrw_fp16_105, conv_ctest_bwdwrw_fp16_106, 
conv_ctest_bwdwrw_fp16_107, conv_ctest_bwdwrw_fp16_108, 
conv_ctest_bwdwrw_fp16_109, conv_ctest_bwdwrw_fp16_110, 
conv_ctest_bwdwrw_fp16_111, conv_ctest_bwdwrw_fp16_112, 
conv_ctest_bwdwrw_fp16_113, conv_ctest_bwdwrw_fp16_114, 
conv_ctest_bwdwrw_fp16_115, conv_ctest_bwdwrw_fp16_116, 
conv_ctest_bwdwrw_fp16_117, conv_ctest_bwdwrw_fp16_118, 
conv_ctest_bwdwrw_fp16_119, conv_ctest_bwdwrw_fp16_120, 
conv_ctest_bwdwrw_fp16_121, conv_ctest_bwdwrw_fp16_122, 
conv_ctest_bwdwrw_fp16_123, conv_ctest_bwdwrw_fp16_124, 
conv_ctest_bwdwrw_fp16_125, conv_ctest_bwdwrw_fp16_126, 
conv_ctest_bwdwrw_fp16_127, conv_ctest_bwdwrw_fp16_128, 
conv_ctest_bwdwrw_fp16_129, conv_ctest_bwdwrw_fp16_130, 
conv_ctest_bwdwrw_fp16_131, conv_ctest_bwdwrw_fp16_132, 
conv_ctest_bwdwrw_fp16_133, conv_ctest_bwdwrw_fp16_134, 
conv_ctest_bwdwrw_fp16_135, conv_ctest_bwdwrw_fp16_136, 
conv_ctest_bwdwrw_fp16_137, conv_ctest_bwdwrw_fp16_138, 
conv_ctest_bwdwrw_fp16_139, conv_ctest_bwdwrw_fp16_140, 
conv_ctest_bwdwrw_fp16_141, conv_ctest_bwdwrw_fp16_142, 
conv_ctest_bwdwrw_fp16_143, conv_ctest_bwdwrw_fp16_144, 
conv_ctest_bwdwrw_fp16_145, conv_ctest_bwdwrw_fp16_146, 
conv_ctest_bwdwrw_fp16_147, conv_ctest_bwdwrw_fp16_148, 
conv_ctest_bwdwrw_fp16_149, conv_ctest_bwdwrw_fp16_150, 
conv_ctest_bwdwrw_fp16_151, conv_ctest_bwdwrw_fp16_152, 
conv_ctest_bwdwrw_fp16_153, conv_ctest_bwdwrw_fp16_154, 
conv_ctest_bwdwrw_fp16_155, conv_ctest_bwdwrw_fp16_156, 
conv_ctest_bwdwrw_fp16_157, conv_ctest_bwdwrw_fp16_158, 
conv_ctest_bwdwrw_fp16_159, conv_ctest_bwdwrw_fp16_160, 
conv_ctest_bwdwrw_fp16_161, conv_ctest_bwdwrw_fp16_162, 
conv_ctest_bwdwrw_fp16_163, conv_ctest_bwdwrw_fp16_164, 
conv_ctest_bwdwrw_fp16_165, conv_ctest_bwdwrw_fp16_166, 
conv_ctest_bwdwrw_fp16_167, conv_ctest_bwdwrw_fp16_168, 
conv_ctest_bwdwrw_fp16_169, conv_ctest_bwdwrw_fp16_170, 
conv_ctest_bwdwrw_fp16_171, conv_ctest_bwdwrw_fp16_172, 
conv_ctest_bwdwrw_fp16_173, conv_ctest_bwdwrw_fp16_174, 
conv_ctest_bwdwrw_fp16_175, conv_ctest_bwdwrw_fp16_176, 
conv_ctest_bwdwrw_fp16_177, conv_ctest_bwdwrw_fp16_178, 
conv_ctest_bwdwrw_fp16_179, conv_ctest_bwdwrw_fp16_180, 
conv_ctest_bwdwrw_fp16_181, conv_ctest_bwdwrw_fp16_182, 
conv_ctest_bwdwrw_fp16_183, conv_ctest_bwdwrw_fp16_184, 
conv_ctest_bwdwrw_fp16_185, conv_ctest_bwdwrw_fp16_186, 
conv_ctest_bwdwrw_fp16_187, conv_ctest_bwdwrw_fp16_188, 
conv_ctest_bwdwrw_fp16_189, conv_ctest_bwdwrw_fp16_190, 
conv_ctest_bwdwrw_fp16_191, conv_ctest_bwdwrw_fp16_192, 
conv_ctest_bwdwrw_fp16_193, conv_ctest_bwdwrw_fp16_194, 
conv_ctest_bwdwrw_fp16_195, conv_ctest_bwdwrw_fp16_196, 
conv_ctest_bwdwrw_fp16_197, conv_ctest_bwdwrw_fp16_198, 
conv_ctest_bwdwrw_fp16_199, conv_ctest_bwdwrw_fp16_200, 
conv_ctest_bwdwrw_fp16_201, conv_ctest_bwdwrw_fp16_202, 
conv_ctest_bwdwrw_fp16_203, conv_ctest_bwdwrw_fp16_204, 
conv_ctest_bwdwrw_fp16_205, conv_ctest_bwdwrw_fp16_206, 
conv_ctest_bwdwrw_fp16_207, conv_ctest_bwdwrw_fp16_208, 
conv_ctest_bwdwrw_fp16_209, conv_ctest_bwdwrw_fp16_210, 
conv_ctest_bwdwrw_fp16_211, conv_ctest_bwdwrw_fp16_212, 
conv_ctest_bwdwrw_fp16_213, conv_ctest_bwdwrw_fp16_214, 
conv_ctest_bwdwrw_fp16_215, conv_ctest_bwdwrw_fp16_216, 
conv_ctest_bwdwrw_fp16_217, conv_ctest_bwdwrw_fp16_218, 
conv_ctest_bwdwrw_fp16_219, conv_ctest_bwdwrw_fp16_220, 
conv_ctest_bwdwrw_fp16_221, conv_ctest_bwdwrw_fp16_222, 
conv_ctest_bwdwrw_fp16_223, conv_ctest_bwdwrw_fp16_224, 
conv_ctest_bwdwrw_fp16_225, conv_ctest_bwdwrw_fp16_226, 
conv_ctest_bwdwrw_fp16_227, conv_ctest_bwdwrw_fp16_228, 
conv_ctest_bwdwrw_fp16_229, conv_ctest_bwdwrw_fp16_230, 
conv_ctest_bwdwrw_fp16_231, conv_ctest_bwdwrw_fp16_232, 
conv_ctest_bwdwrw_fp16_233, conv_ctest_bwdwrw_fp16_234, 
conv_ctest_bwdwrw_fp16_235, conv_ctest_bwdwrw_fp16_236, 
conv_ctest_bwdwrw_fp16_237, conv_ctest_bwdwrw_fp16_238, 
conv_ctest_bwdwrw_fp16_239, conv_ctest_bwdwrw_fp16_240, 
conv_ctest_bwdwrw_fp16_241, conv_ctest_bwdwrw_fp16_242, 
conv_ctest_bwdwrw_fp16_243, conv_ctest_bwdwrw_fp16_244, 
conv_ctest_bwdwrw_fp16_245, conv_ctest_bwdwrw_fp16_246, 
conv_ctest_bwdwrw_fp16_247, conv_ctest_bwdwrw_fp16_248, 
conv_ctest_bwdwrw_fp16_249, conv_ctest_bwdwrw_fp16_250, 
conv_ctest_bwdwrw_fp16_251, conv_ctest_bwdwrw_fp16_252, 
conv_ctest_bwdwrw_fp16_253, conv_ctest_bwdwrw_fp16_254, 
conv_ctest_bwdwrw_fp16_255, conv_ctest_bwdwrw_fp16_256, 
conv_ctest_bwdwrw_fp16_257, conv_ctest_bwdwrw_fp16_258, 
conv_ctest_bwdwrw_fp16_259, conv_ctest_bwdwrw_fp16_260, 
conv_ctest_bwdwrw_fp16_261, conv_ctest_bwdwrw_fp16_262, 
conv_ctest_bwdwrw_fp16_263, conv_ctest_bwdwrw_fp16_264, 
conv_ctest_bwdwrw_fp16_265, conv_ctest_bwdwrw_fp16_266, 
conv_ctest_bwdwrw_fp16_267, conv_ctest_bwdwrw_fp16_268, 
conv_ctest_bwdwrw_fp16_269, conv_ctest_bwdwrw_fp16_270, 
conv_ctest_bwdwrw_fp16_271, conv_ctest_bwdwrw_fp16_272, 
conv_ctest_bwdwrw_fp16_273, conv_ctest_bwdwrw_fp16_274, 
conv_ctest_bwdwrw_fp16_275, conv_ctest_bwdwrw_fp16_276, 
conv_ctest_bwdwrw_fp16_277, conv_ctest_bwdwrw_fp16_278, 
conv_ctest_bwdwrw_fp16_279, conv_ctest_bwdwrw_fp16_280, 
conv_ctest_bwdwrw_fp16_281, conv_ctest_bwdwrw_fp16_282, 
conv_ctest_bwdwrw_fp16_283, conv_ctest_bwdwrw_fp16_284, 
conv_ctest_bwdwrw_fp16_285, conv_ctest_bwdwrw_fp16_286, 
conv_ctest_bwdwrw_fp16_287, conv_ctest_bwdwrw_fp16_288, 
conv_ctest_bwdwrw_fp16_289, conv_ctest_bwdwrw_fp16_290, 
conv_ctest_bwdwrw_fp16_291, conv_ctest_bwdwrw_fp16_292, 
conv_ctest_bwdwrw_fp16_293, conv_ctest_bwdwrw_fp16_294, 
conv_ctest_bwdwrw_fp16_295, conv_ctest_bwdwrw_fp16_296, 
conv_ctest_bwdwrw_fp16_297, conv_ctest_bwdwrw_fp16_298, 
conv_ctest_bwdwrw_fp16_299, conv_ctest_bwdwrw_fp16_300, 
conv_ctest_bwdwrw_fp16_301, conv_ctest_bwdwrw_fp16_302, 
conv_ctest_bwdwrw_fp16_303, conv_ctest_bwdwrw_fp16_304, 
conv_ctest_bwdwrw_fp16_305, conv_ctest_bwdwrw_fp16_306, 
conv_ctest_bwdwrw_fp16_307, conv_ctest_bwdwrw_fp16_308, 
conv_ctest_bwdwrw_fp16_309, conv_ctest_bwdwrw_fp16_310, 
conv_ctest_bwdwrw_fp16_311, conv_ctest_bwdwrw_fp16_312, 
conv_ctest_bwdwrw_fp16_313, conv_ctest_bwdwrw_fp16_314, 
conv_ctest_bwdwrw_fp16_315, conv_ctest_bwdwrw_fp16_316, 
conv_ctest_bwdwrw_fp16_317, conv_ctest_bwdwrw_fp16_318, 
conv_ctest_bwdwrw_fp16_319, conv_ctest_bwdwrw_fp16_320, 
conv_ctest_bwdwrw_fp16_321, conv_ctest_bwdwrw_fp16_322, 
conv_ctest_bwdwrw_fp16_323, conv_ctest_bwdwrw_fp16_324, 
conv_ctest_bwdwrw_fp16_325, conv_ctest_bwdwrw_fp16_326, 
conv_ctest_bwdwrw_fp16_327, conv_ctest_bwdwrw_fp16_328, 
conv_ctest_bwdwrw_fp16_329, conv_ctest_bwdwrw_fp16_330, 
conv_ctest_bwdwrw_fp16_331, conv_ctest_bwdwrw_fp16_332, 
conv_ctest_bwdwrw_fp16_333, conv_ctest_bwdwrw_fp16_334, 
conv_ctest_bwdwrw_fp16_335, conv_ctest_bwdwrw_fp16_336, 
conv_ctest_bwdwrw_fp16_337, conv_ctest_bwdwrw_fp16_338, 
conv_ctest_bwdwrw_fp16_339, conv_ctest_bwdwrw_fp16_340, 
conv_ctest_bwdwrw_fp16_341, conv_ctest_bwdwrw_fp16_342, 
conv_ctest_bwdwrw_fp16_343, conv_ctest_bwdwrw_fp16_344, 
conv_ctest_bwdwrw_fp16_345, conv_ctest_bwdwrw_fp16_346, 
conv_ctest_bwdwrw_fp16_347, conv_ctest_bwdwrw_fp16_348, 
conv_ctest_bwdwrw_fp16_349, conv_ctest_bwdwrw_fp16_350, 
conv_ctest_bwdwrw_fp16_351, conv_ctest_bwdwrw_fp16_352, 
conv_ctest_bwdwrw_fp16_353, conv_ctest_bwdwrw_fp16_354, 
conv_ctest_bwdwrw_fp16_355, conv_ctest_bwdwrw_fp16_356, 
conv_ctest_bwdwrw_fp16_357, conv_ctest_bwdwrw_fp16_358, 
conv_ctest_bwdwrw_fp16_359, conv_ctest_bwdwrw_fp16_360, 
conv_ctest_bwdwrw_fp16_361, conv_ctest_bwdwrw_fp16_362, 
conv_ctest_bwdwrw_fp16_363, conv_ctest_bwdwrw_fp16_364, 
conv_ctest_bwdwrw_fp16_365, conv_ctest_bwdwrw_fp16_366, 
conv_ctest_bwdwrw_fp16_367, conv_ctest_bwdwrw_fp16_368, 
conv_ctest_bwdwrw_fp16_369, conv_ctest_bwdwrw_fp16_370, 
conv_ctest_bwdwrw_fp16_371, conv_ctest_bwdwrw_fp16_372, 
conv_ctest_bwdwrw_fp16_373, conv_ctest_bwdwrw_fp16_374, 
conv_ctest_bwdwrw_fp16_375, conv_ctest_bwdwrw_fp16_376, 
conv_ctest_bwdwrw_fp16_377, conv_ctest_bwdwrw_fp16_378, 
conv_ctest_bwdwrw_fp16_379, conv_ctest_bwdwrw_fp16_380, 
conv_ctest_bwdwrw_fp16_381, conv_ctest_bwdwrw_fp16_382, 
conv_ctest_bwdwrw_fp16_383, conv_ctest_bwdwrw_fp16_384, 
conv_ctest_bwdwrw_fp16_385, conv_ctest_bwdwrw_fp16_386, 
conv_ctest_bwdwrw_fp16_387, conv_ctest_bwdwrw_fp16_388, 
conv_ctest_bwdwrw_fp16_389, conv_ctest_bwdwrw_fp16_390, 
conv_ctest_bwdwrw_fp16_391, conv_ctest_bwdwrw_fp16_392, 
conv_ctest_bwdwrw_fp16_393, conv_ctest_bwdwrw_fp16_394, 
conv_ctest_bwdwrw_fp16_395, conv_ctest_bwdwrw_fp16_396, 
conv_ctest_bwdwrw_fp16_397, conv_ctest_bwdwrw_fp16_398, 
conv_ctest_bwdwrw_fp16_399, conv_ctest_bwdwrw_fp16_400, 
conv_ctest_bwdwrw_fp16_401, conv_ctest_bwdwrw_fp16_402, 
conv_ctest_bwdwrw_fp16_403, conv_ctest_bwdwrw_fp16_404, 
conv_ctest_bwdwrw_fp16_405, conv_ctest_bwdwrw_fp16_406, 
conv_ctest_bwdwrw_fp16_407, conv_ctest_bwdwrw_fp16_408, 
conv_ctest_bwdwrw_fp16_409, conv_ctest_bwdwrw_fp16_410, 
conv_ctest_bwdwrw_fp16_411, conv_ctest_bwdwrw_fp16_412, 
conv_ctest_bwdwrw_fp16_413, conv_ctest_bwdwrw_fp16_414, 
conv_ctest_bwdwrw_fp16_415, conv_ctest_bwdwrw_fp16_416, 
conv_ctest_bwdwrw_fp16_417, conv_ctest_bwdwrw_fp16_418, 
conv_ctest_bwdwrw_fp16_419, conv_ctest_bwdwrw_fp16_420, 
conv_ctest_bwdwrw_fp16_421, conv_ctest_bwdwrw_fp16_422, 
conv_ctest_bwdwrw_fp16_423, conv_ctest_bwdwrw_fp16_424, 
conv_ctest_bwdwrw_fp16_425, conv_ctest_bwdwrw_fp16_426, 
conv_ctest_bwdwrw_fp16_427, conv_ctest_bwdwrw_fp16_428, 
conv_ctest_bwdwrw_fp16_429, conv_ctest_bwdwrw_fp16_430, 
conv_ctest_bwdwrw_fp16_431, conv_ctest_bwdwrw_fp16_432, 
conv_ctest_bwdwrw_fp16_433, conv_ctest_bwdwrw_fp16_434, 
conv_ctest_bwdwrw_fp16_435, conv_ctest_bwdwrw_fp16_436, 
conv_ctest_bwdwrw_fp16_437, conv_ctest_bwdwrw_fp16_438, 
conv_ctest_bwdwrw_fp16_439, conv_ctest_bwdwrw_fp16_440, 
conv_ctest_bwdwrw_fp16_441, conv_ctest_bwdwrw_fp16_442, 
conv_ctest_bwdwrw_fp16_443, conv_ctest_bwdwrw_fp16_444, 
conv_ctest_bwdwrw_fp16_445, conv_ctest_bwdwrw_fp16_446, 
conv_ctest_bwdwrw_fp16_447, conv_ctest_bwdwrw_fp16_448, 
conv_ctest_bwdwrw_fp16_449, conv_ctest_bwdwrw_fp16_450, 
conv_ctest_bwdwrw_fp16_451, conv_ctest_bwdwrw_fp16_452, 
conv_ctest_bwdwrw_fp16_453, conv_ctest_bwdwrw_fp16_454, 
conv_ctest_bwdwrw_fp16_455, conv_ctest_bwdwrw_fp16_456, 
conv_ctest_bwdwrw_fp16_457, conv_ctest_bwdwrw_fp16_458, 
conv_ctest_bwdwrw_fp16_459, conv_ctest_bwdwrw_fp16_460, 
conv_ctest_bwdwrw_fp16_461, conv_ctest_bwdwrw_fp16_462, 
conv_ctest_bwdwrw_fp16_463, conv_ctest_bwdwrw_fp16_464, 
conv_ctest_bwdwrw_fp16_465, conv_ctest_bwdwrw_fp16_466, 
conv_ctest_bwdwrw_fp16_467, conv_ctest_bwdwrw_fp16_468, 
conv_ctest_bwdwrw_fp16_469, conv_ctest_bwdwrw_fp16_470, 
conv_ctest_bwdwrw_fp16_471, conv_ctest_bwdwrw_fp16_472, 
conv_ctest_bwdwrw_fp16_473, conv_ctest_bwdwrw_fp16_474, 
conv_ctest_bwdwrw_fp16_475, conv_ctest_bwdwrw_fp16_476, 
conv_ctest_bwdwrw_fp16_477, conv_ctest_bwdwrw_fp16_478, 
conv_ctest_bwdwrw_fp16_479, conv_ctest_bwdwrw_fp16_480, 
conv_ctest_bwdwrw_fp16_481, conv_ctest_bwdwrw_fp16_482, 
conv_ctest_bwdwrw_fp16_483, conv_ctest_bwdwrw_fp16_484, 
conv_ctest_bwdwrw_fp16_485, conv_ctest_bwdwrw_fp16_486, 
conv_ctest_bwdwrw_fp16_487, conv_ctest_bwdwrw_fp16_488, 
conv_ctest_bwdwrw_fp16_489, conv_ctest_bwdwrw_fp16_490, 
conv_ctest_bwdwrw_fp16_491, conv_ctest_bwdwrw_fp16_492, 
conv_ctest_bwdwrw_fp16_493, conv_ctest_bwdwrw_fp16_494, 
conv_ctest_bwdwrw_fp16_495, conv_ctest_bwdwrw_fp16_496, 
conv_ctest_bwdwrw_fp16_497, conv_ctest_bwdwrw_fp16_498, 
conv_ctest_bwdwrw_fp16_499, conv_ctest_bwdwrw_fp16_500, 
conv_ctest_bwdwrw_fp16_501, conv_ctest_bwdwrw_fp16_502, 
conv_ctest_bwdwrw_fp16_503, conv_ctest_bwdwrw_fp16_504, 
conv_ctest_bwdwrw_fp16_505, conv_ctest_bwdwrw_fp16_506, 
conv_ctest_bwdwrw_fp16_507, conv_ctest_bwdwrw_fp16_508, 
conv_ctest_bwdwrw_fp16_509, conv_ctest_bwdwrw_fp16_510, 
conv_ctest_bwdwrw_fp16_511, conv_ctest_bwdwrw_fp16_512, 
conv_ctest_bwdwrw_fp16_513, conv_ctest_bwdwrw_fp16_514, 
conv_ctest_bwdwrw_fp16_515, conv_ctest_bwdwrw_fp16_516, 
conv_ctest_bwdwrw_fp16_517, conv_ctest_bwdwrw_fp16_518, 
conv_ctest_bwdwrw_fp16_519, conv_ctest_bwdwrw_fp16_520, 
conv_ctest_bwdwrw_fp16_521, conv_ctest_bwdwrw_fp16_522, 
conv_ctest_bwdwrw_fp16_523, conv_ctest_bwdwrw_fp16_524, 
conv_ctest_bwdwrw_fp16_525, conv_ctest_bwdwrw_fp16_526, 
conv_ctest_bwdwrw_fp16_527, conv_ctest_bwdwrw_fp16_528, 
conv_ctest_bwdwrw_fp16_529, conv_ctest_bwdwrw_fp16_530, 
conv_ctest_bwdwrw_fp16_531, conv_ctest_bwdwrw_fp16_532, 
conv_ctest_bwdwrw_fp16_533, conv_ctest_bwdwrw_fp16_534, 
conv_ctest_bwdwrw_fp16_535, conv_ctest_bwdwrw_fp16_536, 
conv_ctest_bwdwrw_fp16_537, conv_ctest_bwdwrw_fp16_538, 
conv_ctest_bwdwrw_fp16_539, conv_ctest_bwdwrw_fp16_540, 
conv_ctest_bwdwrw_fp16_541, conv_ctest_bwdwrw_fp16_542, 
conv_ctest_bwdwrw_fp16_543, conv_ctest_bwdwrw_fp16_544, 
conv_ctest_bwdwrw_fp16_545, conv_ctest_bwdwrw_fp16_546, 
conv_ctest_bwdwrw_fp16_547, conv_ctest_bwdwrw_fp16_548, 
conv_ctest_bwdwrw_fp16_549, conv_ctest_bwdwrw_fp16_550, 
conv_ctest_bwdwrw_fp16_551, conv_ctest_bwdwrw_fp16_552, 
conv_ctest_bwdwrw_fp16_553, conv_ctest_bwdwrw_fp16_554, 
conv_ctest_bwdwrw_fp16_555, conv_ctest_bwdwrw_fp16_556, 
conv_ctest_bwdwrw_fp16_557, conv_ctest_bwdwrw_fp16_558, 
conv_ctest_bwdwrw_fp16_559, conv_ctest_bwdwrw_fp16_560, 
conv_ctest_bwdwrw_fp16_561, conv_ctest_bwdwrw_fp16_562, 
conv_ctest_bwdwrw_fp16_563, conv_ctest_bwdwrw_fp16_564, 
conv_ctest_bwdwrw_fp16_565, conv_ctest_bwdwrw_fp16_566, 
conv_ctest_bwdwrw_fp16_567, conv_ctest_bwdwrw_fp16_568, 
conv_ctest_bwdwrw_fp16_569, conv_ctest_bwdwrw_fp16_570, 
conv_ctest_bwdwrw_fp16_571, conv_ctest_bwdwrw_fp16_572, 
conv_ctest_bwdwrw_fp16_573, conv_ctest_bwdwrw_fp16_574, 
conv_ctest_bwdwrw_fp16_575, conv_ctest_bwdwrw_fp16_576, 
conv_ctest_bwdwrw_fp16_577, conv_ctest_bwdwrw_fp16_578, 
conv_ctest_bwdwrw_fp16_579, conv_ctest_bwdwrw_fp16_580, 
conv_ctest_bwdwrw_fp16_581, conv_ctest_bwdwrw_fp16_582, 
conv_ctest_bwdwrw_fp16_583, conv_ctest_bwdwrw_fp16_584, 
conv_ctest_bwdwrw_fp16_585, conv_ctest_bwdwrw_fp16_586, 
conv_ctest_bwdwrw_fp16_587, conv_ctest_bwdwrw_fp16_588, 
conv_ctest_bwdwrw_fp16_589, conv_ctest_bwdwrw_fp16_590, 
conv_ctest_bwdwrw_fp16_591, conv_ctest_bwdwrw_fp16_592, 
conv_ctest_bwdwrw_fp16_593, conv_ctest_bwdwrw_fp16_594, 
conv_ctest_bwdwrw_fp16_595, conv_ctest_bwdwrw_fp16_596, 
conv_ctest_bwdwrw_fp16_597, conv_ctest_bwdwrw_fp16_598, 
conv_ctest_bwdwrw_fp16_599, conv_ctest_bwdwrw_fp16_600, 
conv_ctest_bwdwrw_fp16_601, conv_ctest_bwdwrw_fp16_602, 
conv_ctest_bwdwrw_fp16_603, conv_ctest_bwdwrw_fp16_604, 
conv_ctest_bwdwrw_fp16_605, conv_ctest_bwdwrw_fp16_606, 
conv_ctest_bwdwrw_fp16_607, conv_ctest_bwdwrw_fp16_608, 
conv_ctest_bwdwrw_fp16_609, conv_ctest_bwdwrw_fp16_610, 
conv_ctest_bwdwrw_fp16_611, conv_ctest_bwdwrw_fp16_612, 
conv_ctest_bwdwrw_fp16_613, conv_ctest_bwdwrw_fp16_614, 
conv_ctest_bwdwrw_fp16_615, conv_ctest_bwdwrw_fp16_616, 
conv_ctest_bwdwrw_fp16_617, conv_ctest_bwdwrw_fp16_618, 
conv_ctest_bwdwrw_fp16_619, conv_ctest_bwdwrw_fp16_620, 
conv_ctest_bwdwrw_fp16_621, conv_ctest_bwdwrw_fp16_622, 
conv_ctest_bwdwrw_fp16_623, conv_ctest_bwdwrw_fp16_624, 
conv_ctest_bwdwrw_fp16_625, conv_ctest_bwdwrw_fp16_626, 
conv_ctest_bwdwrw_fp16_627, conv_ctest_bwdwrw_fp16_628, 
conv_ctest_bwdwrw_fp16_629, 
};

gemm_tuple conv_ctest_fwd_fp32_001 {{10000, 1, 363, 10000, 363, 10000}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_002 {{100, 1, 1008, 100, 1008, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_003 {{100, 1, 1152, 100, 1152, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_004 {{100, 1, 128, 100, 128, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_005 {{100, 1, 1296, 100, 1296, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_006 {{100, 1, 1440, 100, 1440, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_007 {{100, 1, 1600, 100, 1600, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_008 {{100, 1, 1728, 100, 1728, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_009 {{100, 1, 192, 100, 192, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_010 {{100, 1, 2304, 100, 2304, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_011 {{100, 1, 2400, 100, 2400, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_012 {{100, 1, 256, 100, 256, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_013 {{100, 1, 400, 100, 400, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_014 {{100, 1, 4608, 100, 4608, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_015 {{100, 1, 480, 100, 480, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_016 {{100, 1, 4, 100, 4, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_017 {{100, 1, 512, 100, 512, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_018 {{100, 1, 528, 100, 528, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_019 {{100, 1, 576, 100, 576, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_020 {{100, 1, 600, 100, 600, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_021 {{100, 1, 608, 100, 608, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_022 {{100, 1, 64, 100, 64, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_023 {{100, 1, 800, 100, 800, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_024 {{100, 1, 864, 100, 864, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_025 {{100, 1, 9216, 100, 9216, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_026 {{100, 1, 9, 100, 9, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_027 {{1024, 1, 128, 1024, 128, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_028 {{1024, 1, 147, 1024, 147, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_029 {{1024, 1, 192, 1024, 192, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_030 {{1024, 1, 256, 1024, 256, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_031 {{1024, 1, 27, 1024, 27, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_032 {{1024, 1, 320, 1024, 320, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_033 {{1024, 1, 363, 1024, 363, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_034 {{1024, 1, 512, 1024, 512, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_035 {{1024, 1, 64, 1024, 64, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_036 {{1024, 1, 75, 1024, 75, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_037 {{10404, 1, 363, 10404, 363, 10404}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_038 {{10609, 1, 147, 10609, 147, 10609}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_039 {{10816, 1, 147, 10816, 147, 10816}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_040 {{10816, 1, 1600, 10816, 1600, 10816}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_041 {{11025, 1, 147, 11025, 147, 11025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_042 {{11236, 1, 147, 11236, 147, 11236}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_043 {{11449, 1, 147, 11449, 147, 11449}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_044 {{11449, 1, 363, 11449, 363, 11449}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_045 {{11449, 1, 75, 11449, 75, 11449}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_046 {{1156, 1, 27, 1156, 27, 1156}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_047 {{11664, 1, 147, 11664, 147, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_048 {{11664, 1, 1600, 11664, 1600, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_049 {{11664, 1, 363, 11664, 363, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_050 {{11664, 1, 576, 11664, 576, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_051 {{11881, 1, 147, 11881, 147, 11881}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_052 {{11881, 1, 363, 11881, 363, 11881}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_053 {{11881, 1, 75, 11881, 75, 11881}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_054 {{12100, 1, 147, 12100, 147, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_055 {{12100, 1, 1600, 12100, 1600, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_056 {{12100, 1, 27, 12100, 27, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_057 {{12100, 1, 363, 12100, 363, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_058 {{12100, 1, 576, 12100, 576, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_059 {{12100, 1, 75, 12100, 75, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_060 {{121, 1, 1024, 121, 1024, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_061 {{121, 1, 1056, 121, 1056, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_062 {{121, 1, 192, 121, 192, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_063 {{121, 1, 2048, 121, 2048, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_064 {{121, 1, 2304, 121, 2304, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_065 {{121, 1, 3456, 121, 3456, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_066 {{121, 1, 363, 121, 363, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_067 {{121, 1, 4, 121, 4, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_068 {{121, 1, 512, 121, 512, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_069 {{121, 1, 75, 121, 75, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_070 {{121, 1, 832, 121, 832, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_071 {{12321, 1, 147, 12321, 147, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_072 {{12321, 1, 27, 12321, 27, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_073 {{12321, 1, 363, 12321, 363, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_074 {{12321, 1, 75, 12321, 75, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_075 {{12544, 1, 147, 12544, 147, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_076 {{12544, 1, 1600, 12544, 1600, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_077 {{12544, 1, 27, 12544, 27, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_078 {{12544, 1, 363, 12544, 363, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_079 {{12544, 1, 576, 12544, 576, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_080 {{12544, 1, 75, 12544, 75, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_081 {{12769, 1, 147, 12769, 147, 12769}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_082 {{12769, 1, 27, 12769, 27, 12769}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_083 {{12769, 1, 75, 12769, 75, 12769}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_084 {{12996, 1, 147, 12996, 147, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_085 {{12996, 1, 27, 12996, 27, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_086 {{12996, 1, 363, 12996, 363, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_087 {{12996, 1, 576, 12996, 576, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_088 {{12996, 1, 64, 12996, 64, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_089 {{12996, 1, 75, 12996, 75, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_090 {{13225, 1, 27, 13225, 27, 13225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_091 {{13225, 1, 75, 13225, 75, 13225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_092 {{13456, 1, 147, 13456, 147, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_093 {{13456, 1, 27, 13456, 27, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_094 {{13456, 1, 363, 13456, 363, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_095 {{13456, 1, 64, 13456, 64, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_096 {{13456, 1, 75, 13456, 75, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_097 {{13689, 1, 75, 13689, 75, 13689}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_098 {{13924, 1, 27, 13924, 27, 13924}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_099 {{144, 1, 1008, 144, 1008, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_100 {{144, 1, 1024, 144, 1024, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_101 {{144, 1, 1152, 144, 1152, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_102 {{144, 1, 1296, 144, 1296, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_103 {{144, 1, 1440, 144, 1440, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_104 {{144, 1, 1600, 144, 1600, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_105 {{144, 1, 1728, 144, 1728, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_106 {{144, 1, 2304, 144, 2304, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_107 {{144, 1, 2400, 144, 2400, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_108 {{144, 1, 256, 144, 256, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_109 {{144, 1, 363, 144, 363, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_110 {{144, 1, 400, 144, 400, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_111 {{144, 1, 4608, 144, 4608, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_112 {{144, 1, 4, 144, 4, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_113 {{144, 1, 512, 144, 512, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_114 {{144, 1, 576, 144, 576, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_115 {{144, 1, 600, 144, 600, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_116 {{144, 1, 800, 144, 800, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_117 {{144, 1, 864, 144, 864, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_118 {{144, 1, 9216, 144, 9216, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_119 {{144, 1, 9, 144, 9, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_120 {{169, 1, 1152, 169, 1152, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_121 {{169, 1, 147, 169, 147, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_122 {{169, 1, 1600, 169, 1600, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_123 {{169, 1, 1728, 169, 1728, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_124 {{169, 1, 2048, 169, 2048, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_125 {{169, 1, 2304, 169, 2304, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_126 {{169, 1, 2400, 169, 2400, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_127 {{169, 1, 256, 169, 256, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_128 {{169, 1, 3456, 169, 3456, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_129 {{169, 1, 400, 169, 400, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_130 {{169, 1, 4608, 169, 4608, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_131 {{169, 1, 4, 169, 4, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_132 {{169, 1, 576, 169, 576, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_133 {{169, 1, 800, 169, 800, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_134 {{169, 1, 864, 169, 864, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_135 {{169, 1, 9, 169, 9, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_136 {{16, 1, 1024, 16, 1024, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_137 {{16, 1, 1056, 16, 1056, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_138 {{16, 1, 1200, 16, 1200, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_139 {{16, 1, 1440, 16, 1440, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_140 {{16, 1, 1728, 16, 1728, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_141 {{16, 1, 192, 16, 192, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_142 {{16, 1, 2016, 16, 2016, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_143 {{16, 1, 2304, 16, 2304, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_144 {{16, 1, 4608, 16, 4608, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_145 {{16, 1, 4, 16, 4, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_146 {{16, 1, 512, 16, 512, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_147 {{16, 1, 528, 16, 528, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_148 {{16, 1, 576, 16, 576, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_149 {{16, 1, 608, 16, 608, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_150 {{16, 1, 800, 16, 800, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_151 {{16, 1, 832, 16, 832, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_152 {{16, 1, 9216, 16, 9216, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_153 {{16, 1, 9, 16, 9, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_154 {{1860, 1, 4608, 1860, 4608, 1860}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_155 {{1953, 1, 4608, 1953, 4608, 1953}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_156 {{196, 1, 1008, 196, 1008, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_157 {{196, 1, 1024, 196, 1024, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_158 {{196, 1, 1152, 196, 1152, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_159 {{196, 1, 128, 196, 128, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_160 {{196, 1, 1296, 196, 1296, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_161 {{196, 1, 1440, 196, 1440, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_162 {{196, 1, 147, 196, 147, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_163 {{196, 1, 1600, 196, 1600, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_164 {{196, 1, 1728, 196, 1728, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_165 {{196, 1, 192, 196, 192, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_166 {{196, 1, 2304, 196, 2304, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_167 {{196, 1, 2400, 196, 2400, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_168 {{196, 1, 256, 196, 256, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_169 {{196, 1, 27, 196, 27, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_170 {{196, 1, 320, 196, 320, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_171 {{196, 1, 363, 196, 363, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_172 {{196, 1, 400, 196, 400, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_173 {{196, 1, 4608, 196, 4608, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_174 {{196, 1, 480, 196, 480, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_175 {{196, 1, 4, 196, 4, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_176 {{196, 1, 512, 196, 512, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_177 {{196, 1, 528, 196, 528, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_178 {{196, 1, 576, 196, 576, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_179 {{196, 1, 600, 196, 600, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_180 {{196, 1, 608, 196, 608, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_181 {{196, 1, 64, 196, 64, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_182 {{196, 1, 75, 196, 75, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_183 {{196, 1, 800, 196, 800, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_184 {{196, 1, 864, 196, 864, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_185 {{196, 1, 9216, 196, 9216, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_186 {{196, 1, 9, 196, 9, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_187 {{1, 1, 1200, 1, 1200, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_188 {{1, 1, 363, 1, 363, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_189 {{1, 1, 4608, 1, 4608, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_190 {{1, 1, 4, 1, 4, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_191 {{1, 1, 800, 1, 800, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_192 {{1, 1, 9, 1, 9, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_193 {{2048, 1, 4608, 2048, 4608, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_194 {{2048, 1, 480, 2048, 480, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_195 {{2048, 1, 512, 2048, 512, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_196 {{2048, 1, 528, 2048, 528, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_197 {{2048, 1, 832, 2048, 832, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_198 {{2145, 1, 480, 2145, 480, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_199 {{2145, 1, 512, 2145, 512, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_200 {{2145, 1, 528, 2145, 528, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_201 {{2145, 1, 832, 2145, 832, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_202 {{2244, 1, 4608, 2244, 4608, 2244}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_203 {{225, 1, 128, 225, 128, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_204 {{225, 1, 1600, 225, 1600, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_205 {{225, 1, 192, 225, 192, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_206 {{225, 1, 2048, 225, 2048, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_207 {{225, 1, 2304, 225, 2304, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_208 {{225, 1, 2400, 225, 2400, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_209 {{225, 1, 256, 225, 256, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_210 {{225, 1, 27, 225, 27, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_211 {{225, 1, 320, 225, 320, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_212 {{225, 1, 3456, 225, 3456, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_213 {{225, 1, 400, 225, 400, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_214 {{225, 1, 4, 225, 4, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_215 {{225, 1, 512, 225, 512, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_216 {{225, 1, 64, 225, 64, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_217 {{225, 1, 75, 225, 75, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_218 {{225, 1, 800, 225, 800, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_219 {{2304, 1, 1600, 2304, 1600, 2304}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_220 {{2345, 1, 480, 2345, 480, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_221 {{2345, 1, 512, 2345, 512, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_222 {{2345, 1, 528, 2345, 528, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_223 {{2345, 1, 832, 2345, 832, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_224 {{256, 1, 1008, 256, 1008, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_225 {{256, 1, 1024, 256, 1024, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_226 {{256, 1, 1152, 256, 1152, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_227 {{256, 1, 128, 256, 128, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_228 {{256, 1, 1296, 256, 1296, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_229 {{256, 1, 1440, 256, 1440, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_230 {{256, 1, 147, 256, 147, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_231 {{256, 1, 1728, 256, 1728, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_232 {{256, 1, 192, 256, 192, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_233 {{256, 1, 2304, 256, 2304, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_234 {{256, 1, 256, 256, 256, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_235 {{256, 1, 27, 256, 27, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_236 {{256, 1, 363, 256, 363, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_237 {{256, 1, 4608, 256, 4608, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_238 {{256, 1, 480, 256, 480, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_239 {{256, 1, 4, 256, 4, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_240 {{256, 1, 512, 256, 512, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_241 {{256, 1, 528, 256, 528, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_242 {{256, 1, 576, 256, 576, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_243 {{256, 1, 608, 256, 608, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_244 {{256, 1, 64, 256, 64, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_245 {{256, 1, 75, 256, 75, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_246 {{256, 1, 800, 256, 800, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_247 {{256, 1, 864, 256, 864, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_248 {{256, 1, 9, 256, 9, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_249 {{25, 1, 1008, 25, 1008, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_250 {{25, 1, 1024, 25, 1024, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_251 {{25, 1, 1056, 25, 1056, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_252 {{25, 1, 1152, 25, 1152, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_253 {{25, 1, 1200, 25, 1200, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_254 {{25, 1, 1296, 25, 1296, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_255 {{25, 1, 1440, 25, 1440, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_256 {{25, 1, 1600, 25, 1600, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_257 {{25, 1, 1728, 25, 1728, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_258 {{25, 1, 192, 25, 192, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_259 {{25, 1, 2016, 25, 2016, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_260 {{25, 1, 2304, 25, 2304, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_261 {{25, 1, 2400, 25, 2400, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_262 {{25, 1, 3456, 25, 3456, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_263 {{25, 1, 400, 25, 400, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_264 {{25, 1, 4608, 25, 4608, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_265 {{25, 1, 4, 25, 4, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_266 {{25, 1, 512, 25, 512, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_267 {{25, 1, 528, 25, 528, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_268 {{25, 1, 576, 25, 576, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_269 {{25, 1, 600, 25, 600, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_270 {{25, 1, 608, 25, 608, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_271 {{25, 1, 800, 25, 800, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_272 {{25, 1, 832, 25, 832, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_273 {{25, 1, 864, 25, 864, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_274 {{25, 1, 9216, 25, 9216, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_275 {{25, 1, 9, 25, 9, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_276 {{2601, 1, 1600, 2601, 1600, 2601}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_277 {{2704, 1, 1152, 2704, 1152, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_278 {{2704, 1, 1600, 2704, 1600, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_279 {{2704, 1, 2304, 2704, 2304, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_280 {{2704, 1, 576, 2704, 576, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_281 {{289, 1, 128, 289, 128, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_282 {{289, 1, 192, 289, 192, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_283 {{289, 1, 256, 289, 256, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_284 {{289, 1, 320, 289, 320, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_285 {{289, 1, 4, 289, 4, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_286 {{289, 1, 512, 289, 512, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_287 {{289, 1, 64, 289, 64, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_288 {{289, 1, 75, 289, 75, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_289 {{2916, 1, 1152, 2916, 1152, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_290 {{2916, 1, 1600, 2916, 1600, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_291 {{2916, 1, 2304, 2916, 2304, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_292 {{2916, 1, 576, 2916, 576, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_293 {{3025, 1, 1600, 3025, 1600, 3025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_294 {{3025, 1, 576, 3025, 576, 3025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_295 {{3136, 1, 1152, 3136, 1152, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_296 {{3136, 1, 1600, 3136, 1600, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_297 {{3136, 1, 2304, 3136, 2304, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_298 {{3136, 1, 576, 3136, 576, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_299 {{3136, 1, 64, 3136, 64, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_300 {{3249, 1, 1600, 3249, 1600, 3249}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_301 {{3249, 1, 64, 3249, 64, 3249}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_302 {{324, 1, 128, 324, 128, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_303 {{324, 1, 192, 324, 192, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_304 {{324, 1, 256, 324, 256, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_305 {{324, 1, 27, 324, 27, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_306 {{324, 1, 480, 324, 480, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_307 {{324, 1, 512, 324, 512, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_308 {{324, 1, 528, 324, 528, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_309 {{324, 1, 576, 324, 576, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_310 {{324, 1, 608, 324, 608, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_311 {{324, 1, 64, 324, 64, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_312 {{33540, 1, 480, 33540, 480, 33540}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_313 {{3364, 1, 1152, 3364, 1152, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_314 {{3364, 1, 128, 3364, 128, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_315 {{3364, 1, 2304, 3364, 2304, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_316 {{3364, 1, 256, 3364, 256, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_317 {{3364, 1, 576, 3364, 576, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_318 {{3364, 1, 64, 3364, 64, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_319 {{34320, 1, 480, 34320, 480, 34320}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_320 {{3481, 1, 64, 3481, 64, 3481}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_321 {{3600, 1, 128, 3600, 128, 3600}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_322 {{3600, 1, 256, 3600, 256, 3600}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_323 {{3600, 1, 64, 3600, 64, 3600}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_324 {{361, 1, 1600, 361, 1600, 361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_325 {{361, 1, 2400, 361, 2400, 361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_326 {{36, 1, 1008, 36, 1008, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_327 {{36, 1, 1024, 36, 1024, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_328 {{36, 1, 1152, 36, 1152, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_329 {{36, 1, 1296, 36, 1296, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_330 {{36, 1, 1440, 36, 1440, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_331 {{36, 1, 1600, 36, 1600, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_332 {{36, 1, 1728, 36, 1728, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_333 {{36, 1, 2016, 36, 2016, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_334 {{36, 1, 2048, 36, 2048, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_335 {{36, 1, 2304, 36, 2304, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_336 {{36, 1, 2400, 36, 2400, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_337 {{36, 1, 256, 36, 256, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_338 {{36, 1, 3456, 36, 3456, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_339 {{36, 1, 400, 36, 400, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_340 {{36, 1, 4608, 36, 4608, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_341 {{36, 1, 4, 36, 4, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_342 {{36, 1, 512, 36, 512, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_343 {{36, 1, 528, 36, 528, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_344 {{36, 1, 576, 36, 576, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_345 {{36, 1, 600, 36, 600, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_346 {{36, 1, 608, 36, 608, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_347 {{36, 1, 800, 36, 800, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_348 {{36, 1, 864, 36, 864, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_349 {{36, 1, 9216, 36, 9216, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_350 {{36, 1, 9, 36, 9, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_351 {{400, 1, 147, 400, 147, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_352 {{400, 1, 1600, 400, 1600, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_353 {{400, 1, 2400, 400, 2400, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_354 {{400, 1, 400, 400, 400, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_355 {{400, 1, 800, 400, 800, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_356 {{41616, 1, 363, 41616, 363, 41616}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_357 {{42849, 1, 363, 42849, 363, 42849}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_358 {{44521, 1, 363, 44521, 363, 44521}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_359 {{44944, 1, 147, 44944, 147, 44944}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_360 {{45796, 1, 363, 45796, 363, 45796}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_361 {{46225, 1, 147, 46225, 147, 46225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_362 {{46656, 1, 363, 46656, 363, 46656}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_363 {{46656, 1, 75, 46656, 75, 46656}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_364 {{47089, 1, 363, 47089, 363, 47089}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_365 {{47524, 1, 147, 47524, 147, 47524}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_366 {{47524, 1, 363, 47524, 363, 47524}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_367 {{47961, 1, 147, 47961, 147, 47961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_368 {{47961, 1, 363, 47961, 363, 47961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_369 {{47961, 1, 75, 47961, 75, 47961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_370 {{48400, 1, 147, 48400, 147, 48400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_371 {{48400, 1, 27, 48400, 27, 48400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_372 {{48400, 1, 75, 48400, 75, 48400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_373 {{484, 1, 363, 484, 363, 484}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_374 {{48841, 1, 147, 48841, 147, 48841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_375 {{48841, 1, 363, 48841, 363, 48841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_376 {{49284, 1, 147, 49284, 147, 49284}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_377 {{49284, 1, 27, 49284, 27, 49284}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_378 {{49284, 1, 75, 49284, 75, 49284}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_379 {{49729, 1, 147, 49729, 147, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_380 {{49729, 1, 27, 49729, 27, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_381 {{49729, 1, 363, 49729, 363, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_382 {{49729, 1, 75, 49729, 75, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_383 {{49, 1, 1008, 49, 1008, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_384 {{49, 1, 1024, 49, 1024, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_385 {{49, 1, 1056, 49, 1056, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_386 {{49, 1, 1152, 49, 1152, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_387 {{49, 1, 1200, 49, 1200, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_388 {{49, 1, 128, 49, 128, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_389 {{49, 1, 1296, 49, 1296, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_390 {{49, 1, 1440, 49, 1440, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_391 {{49, 1, 147, 49, 147, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_392 {{49, 1, 1600, 49, 1600, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_393 {{49, 1, 1728, 49, 1728, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_394 {{49, 1, 192, 49, 192, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_395 {{49, 1, 2016, 49, 2016, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_396 {{49, 1, 2048, 49, 2048, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_397 {{49, 1, 2304, 49, 2304, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_398 {{49, 1, 2400, 49, 2400, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_399 {{49, 1, 256, 49, 256, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_400 {{49, 1, 3456, 49, 3456, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_401 {{49, 1, 400, 49, 400, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_402 {{49, 1, 4608, 49, 4608, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_403 {{49, 1, 480, 49, 480, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_404 {{49, 1, 4, 49, 4, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_405 {{49, 1, 512, 49, 512, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_406 {{49, 1, 528, 49, 528, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_407 {{49, 1, 576, 49, 576, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_408 {{49, 1, 600, 49, 600, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_409 {{49, 1, 608, 49, 608, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_410 {{49, 1, 64, 49, 64, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_411 {{49, 1, 800, 49, 800, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_412 {{49, 1, 832, 49, 832, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_413 {{49, 1, 864, 49, 864, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_414 {{49, 1, 9216, 49, 9216, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_415 {{49, 1, 9, 49, 9, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_416 {{4, 1, 1200, 4, 1200, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_417 {{4, 1, 1440, 4, 1440, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_418 {{4, 1, 1600, 4, 1600, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_419 {{4, 1, 1728, 4, 1728, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_420 {{4, 1, 2016, 4, 2016, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_421 {{4, 1, 2400, 4, 2400, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_422 {{4, 1, 363, 4, 363, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_423 {{4, 1, 400, 4, 400, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_424 {{4, 1, 4608, 4, 4608, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_425 {{4, 1, 4, 4, 4, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_426 {{4, 1, 512, 4, 512, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_427 {{4, 1, 528, 4, 528, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_428 {{4, 1, 576, 4, 576, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_429 {{4, 1, 600, 4, 600, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_430 {{4, 1, 608, 4, 608, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_431 {{4, 1, 800, 4, 800, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_432 {{4, 1, 9216, 4, 9216, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_433 {{4, 1, 9, 4, 9, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_434 {{50176, 1, 147, 50176, 147, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_435 {{50176, 1, 27, 50176, 27, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_436 {{50176, 1, 363, 50176, 363, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_437 {{50176, 1, 75, 50176, 75, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_438 {{50625, 1, 147, 50625, 147, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_439 {{50625, 1, 27, 50625, 27, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_440 {{50625, 1, 363, 50625, 363, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_441 {{50625, 1, 75, 50625, 75, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_442 {{51076, 1, 27, 51076, 27, 51076}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_443 {{51529, 1, 147, 51529, 147, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_444 {{51529, 1, 27, 51529, 27, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_445 {{51529, 1, 363, 51529, 363, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_446 {{51529, 1, 75, 51529, 75, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_447 {{52441, 1, 147, 52441, 147, 52441}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_448 {{52441, 1, 27, 52441, 27, 52441}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_449 {{52441, 1, 75, 52441, 75, 52441}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_450 {{529, 1, 1600, 529, 1600, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_451 {{529, 1, 2400, 529, 2400, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_452 {{529, 1, 576, 529, 576, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_453 {{529, 1, 864, 529, 864, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_454 {{529, 1, 9, 529, 9, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_455 {{53361, 1, 147, 53361, 147, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_456 {{53361, 1, 27, 53361, 27, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_457 {{53361, 1, 363, 53361, 363, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_458 {{53361, 1, 75, 53361, 75, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_459 {{54289, 1, 27, 54289, 27, 54289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_460 {{576, 1, 1152, 576, 1152, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_461 {{576, 1, 1600, 576, 1600, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_462 {{576, 1, 1728, 576, 1728, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_463 {{576, 1, 2304, 576, 2304, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_464 {{576, 1, 2400, 576, 2400, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_465 {{576, 1, 363, 576, 363, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_466 {{576, 1, 400, 576, 400, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_467 {{576, 1, 4608, 576, 4608, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_468 {{576, 1, 576, 576, 576, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_469 {{576, 1, 75, 576, 75, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_470 {{576, 1, 800, 576, 800, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_471 {{576, 1, 864, 576, 864, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_472 {{625, 1, 1600, 625, 1600, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_473 {{625, 1, 2400, 625, 2400, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_474 {{625, 1, 4, 625, 4, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_475 {{625, 1, 576, 625, 576, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_476 {{625, 1, 864, 625, 864, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_477 {{625, 1, 9, 625, 9, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_478 {{64, 1, 128, 64, 128, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_479 {{64, 1, 147, 64, 147, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_480 {{64, 1, 1600, 64, 1600, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_481 {{64, 1, 192, 64, 192, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_482 {{64, 1, 2304, 64, 2304, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_483 {{64, 1, 2400, 64, 2400, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_484 {{64, 1, 256, 64, 256, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_485 {{64, 1, 400, 64, 400, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_486 {{64, 1, 4608, 64, 4608, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_487 {{64, 1, 480, 64, 480, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_488 {{64, 1, 4, 64, 4, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_489 {{64, 1, 512, 64, 512, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_490 {{64, 1, 528, 64, 528, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_491 {{64, 1, 576, 64, 576, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_492 {{64, 1, 600, 64, 600, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_493 {{64, 1, 608, 64, 608, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_494 {{64, 1, 64, 64, 64, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_495 {{64, 1, 800, 64, 800, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_496 {{64, 1, 9216, 64, 9216, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_497 {{64, 1, 9, 64, 9, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_498 {{676, 1, 1152, 676, 1152, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_499 {{676, 1, 147, 676, 147, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_500 {{676, 1, 1600, 676, 1600, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_501 {{676, 1, 1728, 676, 1728, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_502 {{676, 1, 2304, 676, 2304, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_503 {{676, 1, 2400, 676, 2400, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_504 {{676, 1, 363, 676, 363, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_505 {{676, 1, 400, 676, 400, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_506 {{676, 1, 4608, 676, 4608, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_507 {{676, 1, 4, 676, 4, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_508 {{676, 1, 576, 676, 576, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_509 {{676, 1, 800, 676, 800, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_510 {{676, 1, 864, 676, 864, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_511 {{729, 1, 1152, 729, 1152, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_512 {{729, 1, 1600, 729, 1600, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_513 {{729, 1, 2304, 729, 2304, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_514 {{729, 1, 2400, 729, 2400, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_515 {{729, 1, 4, 729, 4, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_516 {{729, 1, 576, 729, 576, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_517 {{729, 1, 864, 729, 864, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_518 {{729, 1, 9, 729, 9, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_519 {{7440, 1, 4608, 7440, 4608, 7440}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_520 {{7812, 1, 4608, 7812, 4608, 7812}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_521 {{784, 1, 1152, 784, 1152, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_522 {{784, 1, 128, 784, 128, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_523 {{784, 1, 147, 784, 147, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_524 {{784, 1, 1600, 784, 1600, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_525 {{784, 1, 1728, 784, 1728, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_526 {{784, 1, 2304, 784, 2304, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_527 {{784, 1, 2400, 784, 2400, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_528 {{784, 1, 256, 784, 256, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_529 {{784, 1, 27, 784, 27, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_530 {{784, 1, 400, 784, 400, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_531 {{784, 1, 4608, 784, 4608, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_532 {{784, 1, 4, 784, 4, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_533 {{784, 1, 576, 784, 576, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_534 {{784, 1, 64, 784, 64, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_535 {{784, 1, 75, 784, 75, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_536 {{784, 1, 800, 784, 800, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_537 {{784, 1, 864, 784, 864, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_538 {{8192, 1, 4608, 8192, 4608, 8192}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_539 {{8192, 1, 480, 8192, 480, 8192}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_540 {{81, 1, 1008, 81, 1008, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_541 {{81, 1, 1024, 81, 1024, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_542 {{81, 1, 1056, 81, 1056, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_543 {{81, 1, 1152, 81, 1152, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_544 {{81, 1, 1296, 81, 1296, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_545 {{81, 1, 1440, 81, 1440, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_546 {{81, 1, 1600, 81, 1600, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_547 {{81, 1, 1728, 81, 1728, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_548 {{81, 1, 192, 81, 192, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_549 {{81, 1, 2016, 81, 2016, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_550 {{81, 1, 2048, 81, 2048, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_551 {{81, 1, 2304, 81, 2304, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_552 {{81, 1, 2400, 81, 2400, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_553 {{81, 1, 256, 81, 256, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_554 {{81, 1, 3456, 81, 3456, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_555 {{81, 1, 400, 81, 400, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_556 {{81, 1, 4608, 81, 4608, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_557 {{81, 1, 4, 81, 4, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_558 {{81, 1, 512, 81, 512, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_559 {{81, 1, 576, 81, 576, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_560 {{81, 1, 800, 81, 800, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_561 {{81, 1, 832, 81, 832, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_562 {{81, 1, 864, 81, 864, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_563 {{81, 1, 9216, 81, 9216, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_564 {{81, 1, 9, 81, 9, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_565 {{8385, 1, 480, 8385, 480, 8385}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_566 {{841, 1, 128, 841, 128, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_567 {{841, 1, 1600, 841, 1600, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_568 {{841, 1, 256, 841, 256, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_569 {{841, 1, 576, 841, 576, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_570 {{841, 1, 64, 841, 64, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_571 {{841, 1, 864, 841, 864, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_572 {{841, 1, 9, 841, 9, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_573 {{8580, 1, 4608, 8580, 4608, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_574 {{8580, 1, 480, 8580, 480, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_575 {{8580, 1, 512, 8580, 512, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_576 {{8580, 1, 528, 8580, 528, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_577 {{8580, 1, 832, 8580, 832, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_578 {{8777, 1, 480, 8777, 480, 8777}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_579 {{8976, 1, 480, 8976, 480, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_580 {{8976, 1, 512, 8976, 512, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_581 {{8976, 1, 528, 8976, 528, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_582 {{8976, 1, 832, 8976, 832, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_583 {{900, 1, 1152, 900, 1152, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_584 {{900, 1, 128, 900, 128, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_585 {{900, 1, 147, 900, 147, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_586 {{900, 1, 1728, 900, 1728, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_587 {{900, 1, 192, 900, 192, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_588 {{900, 1, 2304, 900, 2304, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_589 {{900, 1, 256, 900, 256, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_590 {{900, 1, 27, 900, 27, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_591 {{900, 1, 320, 900, 320, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_592 {{900, 1, 4608, 900, 4608, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_593 {{900, 1, 4, 900, 4, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_594 {{900, 1, 512, 900, 512, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_595 {{900, 1, 576, 900, 576, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_596 {{900, 1, 64, 900, 64, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_597 {{900, 1, 75, 900, 75, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_598 {{900, 1, 864, 900, 864, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_599 {{9025, 1, 363, 9025, 363, 9025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_600 {{9409, 1, 363, 9409, 363, 9409}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_601 {{9604, 1, 363, 9604, 363, 9604}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_602 {{961, 1, 128, 961, 128, 961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_603 {{961, 1, 256, 961, 256, 961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_604 {{961, 1, 64, 961, 64, 961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_605 {{9801, 1, 363, 9801, 363, 9801}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_606 {{9, 1, 1200, 9, 1200, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_607 {{9, 1, 1440, 9, 1440, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_608 {{9, 1, 1728, 9, 1728, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_609 {{9, 1, 2016, 9, 2016, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_610 {{9, 1, 4608, 9, 4608, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_611 {{9, 1, 4, 9, 4, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_612 {{9, 1, 512, 9, 512, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_613 {{9, 1, 528, 9, 528, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_614 {{9, 1, 576, 9, 576, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_615 {{9, 1, 608, 9, 608, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_616 {{9, 1, 800, 9, 800, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_617 {{9, 1, 9216, 9, 9216, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp32_618 {{9, 1, 9, 9, 9, 9}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_ctest_fwd_fp32 = {
conv_ctest_fwd_fp32_001, conv_ctest_fwd_fp32_002, 
conv_ctest_fwd_fp32_003, conv_ctest_fwd_fp32_004, 
conv_ctest_fwd_fp32_005, conv_ctest_fwd_fp32_006, 
conv_ctest_fwd_fp32_007, conv_ctest_fwd_fp32_008, 
conv_ctest_fwd_fp32_009, conv_ctest_fwd_fp32_010, 
conv_ctest_fwd_fp32_011, conv_ctest_fwd_fp32_012, 
conv_ctest_fwd_fp32_013, conv_ctest_fwd_fp32_014, 
conv_ctest_fwd_fp32_015, conv_ctest_fwd_fp32_016, 
conv_ctest_fwd_fp32_017, conv_ctest_fwd_fp32_018, 
conv_ctest_fwd_fp32_019, conv_ctest_fwd_fp32_020, 
conv_ctest_fwd_fp32_021, conv_ctest_fwd_fp32_022, 
conv_ctest_fwd_fp32_023, conv_ctest_fwd_fp32_024, 
conv_ctest_fwd_fp32_025, conv_ctest_fwd_fp32_026, 
conv_ctest_fwd_fp32_027, conv_ctest_fwd_fp32_028, 
conv_ctest_fwd_fp32_029, conv_ctest_fwd_fp32_030, 
conv_ctest_fwd_fp32_031, conv_ctest_fwd_fp32_032, 
conv_ctest_fwd_fp32_033, conv_ctest_fwd_fp32_034, 
conv_ctest_fwd_fp32_035, conv_ctest_fwd_fp32_036, 
conv_ctest_fwd_fp32_037, conv_ctest_fwd_fp32_038, 
conv_ctest_fwd_fp32_039, conv_ctest_fwd_fp32_040, 
conv_ctest_fwd_fp32_041, conv_ctest_fwd_fp32_042, 
conv_ctest_fwd_fp32_043, conv_ctest_fwd_fp32_044, 
conv_ctest_fwd_fp32_045, conv_ctest_fwd_fp32_046, 
conv_ctest_fwd_fp32_047, conv_ctest_fwd_fp32_048, 
conv_ctest_fwd_fp32_049, conv_ctest_fwd_fp32_050, 
conv_ctest_fwd_fp32_051, conv_ctest_fwd_fp32_052, 
conv_ctest_fwd_fp32_053, conv_ctest_fwd_fp32_054, 
conv_ctest_fwd_fp32_055, conv_ctest_fwd_fp32_056, 
conv_ctest_fwd_fp32_057, conv_ctest_fwd_fp32_058, 
conv_ctest_fwd_fp32_059, conv_ctest_fwd_fp32_060, 
conv_ctest_fwd_fp32_061, conv_ctest_fwd_fp32_062, 
conv_ctest_fwd_fp32_063, conv_ctest_fwd_fp32_064, 
conv_ctest_fwd_fp32_065, conv_ctest_fwd_fp32_066, 
conv_ctest_fwd_fp32_067, conv_ctest_fwd_fp32_068, 
conv_ctest_fwd_fp32_069, conv_ctest_fwd_fp32_070, 
conv_ctest_fwd_fp32_071, conv_ctest_fwd_fp32_072, 
conv_ctest_fwd_fp32_073, conv_ctest_fwd_fp32_074, 
conv_ctest_fwd_fp32_075, conv_ctest_fwd_fp32_076, 
conv_ctest_fwd_fp32_077, conv_ctest_fwd_fp32_078, 
conv_ctest_fwd_fp32_079, conv_ctest_fwd_fp32_080, 
conv_ctest_fwd_fp32_081, conv_ctest_fwd_fp32_082, 
conv_ctest_fwd_fp32_083, conv_ctest_fwd_fp32_084, 
conv_ctest_fwd_fp32_085, conv_ctest_fwd_fp32_086, 
conv_ctest_fwd_fp32_087, conv_ctest_fwd_fp32_088, 
conv_ctest_fwd_fp32_089, conv_ctest_fwd_fp32_090, 
conv_ctest_fwd_fp32_091, conv_ctest_fwd_fp32_092, 
conv_ctest_fwd_fp32_093, conv_ctest_fwd_fp32_094, 
conv_ctest_fwd_fp32_095, conv_ctest_fwd_fp32_096, 
conv_ctest_fwd_fp32_097, conv_ctest_fwd_fp32_098, 
conv_ctest_fwd_fp32_099, conv_ctest_fwd_fp32_100, 
conv_ctest_fwd_fp32_101, conv_ctest_fwd_fp32_102, 
conv_ctest_fwd_fp32_103, conv_ctest_fwd_fp32_104, 
conv_ctest_fwd_fp32_105, conv_ctest_fwd_fp32_106, 
conv_ctest_fwd_fp32_107, conv_ctest_fwd_fp32_108, 
conv_ctest_fwd_fp32_109, conv_ctest_fwd_fp32_110, 
conv_ctest_fwd_fp32_111, conv_ctest_fwd_fp32_112, 
conv_ctest_fwd_fp32_113, conv_ctest_fwd_fp32_114, 
conv_ctest_fwd_fp32_115, conv_ctest_fwd_fp32_116, 
conv_ctest_fwd_fp32_117, conv_ctest_fwd_fp32_118, 
conv_ctest_fwd_fp32_119, conv_ctest_fwd_fp32_120, 
conv_ctest_fwd_fp32_121, conv_ctest_fwd_fp32_122, 
conv_ctest_fwd_fp32_123, conv_ctest_fwd_fp32_124, 
conv_ctest_fwd_fp32_125, conv_ctest_fwd_fp32_126, 
conv_ctest_fwd_fp32_127, conv_ctest_fwd_fp32_128, 
conv_ctest_fwd_fp32_129, conv_ctest_fwd_fp32_130, 
conv_ctest_fwd_fp32_131, conv_ctest_fwd_fp32_132, 
conv_ctest_fwd_fp32_133, conv_ctest_fwd_fp32_134, 
conv_ctest_fwd_fp32_135, conv_ctest_fwd_fp32_136, 
conv_ctest_fwd_fp32_137, conv_ctest_fwd_fp32_138, 
conv_ctest_fwd_fp32_139, conv_ctest_fwd_fp32_140, 
conv_ctest_fwd_fp32_141, conv_ctest_fwd_fp32_142, 
conv_ctest_fwd_fp32_143, conv_ctest_fwd_fp32_144, 
conv_ctest_fwd_fp32_145, conv_ctest_fwd_fp32_146, 
conv_ctest_fwd_fp32_147, conv_ctest_fwd_fp32_148, 
conv_ctest_fwd_fp32_149, conv_ctest_fwd_fp32_150, 
conv_ctest_fwd_fp32_151, conv_ctest_fwd_fp32_152, 
conv_ctest_fwd_fp32_153, conv_ctest_fwd_fp32_154, 
conv_ctest_fwd_fp32_155, conv_ctest_fwd_fp32_156, 
conv_ctest_fwd_fp32_157, conv_ctest_fwd_fp32_158, 
conv_ctest_fwd_fp32_159, conv_ctest_fwd_fp32_160, 
conv_ctest_fwd_fp32_161, conv_ctest_fwd_fp32_162, 
conv_ctest_fwd_fp32_163, conv_ctest_fwd_fp32_164, 
conv_ctest_fwd_fp32_165, conv_ctest_fwd_fp32_166, 
conv_ctest_fwd_fp32_167, conv_ctest_fwd_fp32_168, 
conv_ctest_fwd_fp32_169, conv_ctest_fwd_fp32_170, 
conv_ctest_fwd_fp32_171, conv_ctest_fwd_fp32_172, 
conv_ctest_fwd_fp32_173, conv_ctest_fwd_fp32_174, 
conv_ctest_fwd_fp32_175, conv_ctest_fwd_fp32_176, 
conv_ctest_fwd_fp32_177, conv_ctest_fwd_fp32_178, 
conv_ctest_fwd_fp32_179, conv_ctest_fwd_fp32_180, 
conv_ctest_fwd_fp32_181, conv_ctest_fwd_fp32_182, 
conv_ctest_fwd_fp32_183, conv_ctest_fwd_fp32_184, 
conv_ctest_fwd_fp32_185, conv_ctest_fwd_fp32_186, 
conv_ctest_fwd_fp32_187, conv_ctest_fwd_fp32_188, 
conv_ctest_fwd_fp32_189, conv_ctest_fwd_fp32_190, 
conv_ctest_fwd_fp32_191, conv_ctest_fwd_fp32_192, 
conv_ctest_fwd_fp32_193, conv_ctest_fwd_fp32_194, 
conv_ctest_fwd_fp32_195, conv_ctest_fwd_fp32_196, 
conv_ctest_fwd_fp32_197, conv_ctest_fwd_fp32_198, 
conv_ctest_fwd_fp32_199, conv_ctest_fwd_fp32_200, 
conv_ctest_fwd_fp32_201, conv_ctest_fwd_fp32_202, 
conv_ctest_fwd_fp32_203, conv_ctest_fwd_fp32_204, 
conv_ctest_fwd_fp32_205, conv_ctest_fwd_fp32_206, 
conv_ctest_fwd_fp32_207, conv_ctest_fwd_fp32_208, 
conv_ctest_fwd_fp32_209, conv_ctest_fwd_fp32_210, 
conv_ctest_fwd_fp32_211, conv_ctest_fwd_fp32_212, 
conv_ctest_fwd_fp32_213, conv_ctest_fwd_fp32_214, 
conv_ctest_fwd_fp32_215, conv_ctest_fwd_fp32_216, 
conv_ctest_fwd_fp32_217, conv_ctest_fwd_fp32_218, 
conv_ctest_fwd_fp32_219, conv_ctest_fwd_fp32_220, 
conv_ctest_fwd_fp32_221, conv_ctest_fwd_fp32_222, 
conv_ctest_fwd_fp32_223, conv_ctest_fwd_fp32_224, 
conv_ctest_fwd_fp32_225, conv_ctest_fwd_fp32_226, 
conv_ctest_fwd_fp32_227, conv_ctest_fwd_fp32_228, 
conv_ctest_fwd_fp32_229, conv_ctest_fwd_fp32_230, 
conv_ctest_fwd_fp32_231, conv_ctest_fwd_fp32_232, 
conv_ctest_fwd_fp32_233, conv_ctest_fwd_fp32_234, 
conv_ctest_fwd_fp32_235, conv_ctest_fwd_fp32_236, 
conv_ctest_fwd_fp32_237, conv_ctest_fwd_fp32_238, 
conv_ctest_fwd_fp32_239, conv_ctest_fwd_fp32_240, 
conv_ctest_fwd_fp32_241, conv_ctest_fwd_fp32_242, 
conv_ctest_fwd_fp32_243, conv_ctest_fwd_fp32_244, 
conv_ctest_fwd_fp32_245, conv_ctest_fwd_fp32_246, 
conv_ctest_fwd_fp32_247, conv_ctest_fwd_fp32_248, 
conv_ctest_fwd_fp32_249, conv_ctest_fwd_fp32_250, 
conv_ctest_fwd_fp32_251, conv_ctest_fwd_fp32_252, 
conv_ctest_fwd_fp32_253, conv_ctest_fwd_fp32_254, 
conv_ctest_fwd_fp32_255, conv_ctest_fwd_fp32_256, 
conv_ctest_fwd_fp32_257, conv_ctest_fwd_fp32_258, 
conv_ctest_fwd_fp32_259, conv_ctest_fwd_fp32_260, 
conv_ctest_fwd_fp32_261, conv_ctest_fwd_fp32_262, 
conv_ctest_fwd_fp32_263, conv_ctest_fwd_fp32_264, 
conv_ctest_fwd_fp32_265, conv_ctest_fwd_fp32_266, 
conv_ctest_fwd_fp32_267, conv_ctest_fwd_fp32_268, 
conv_ctest_fwd_fp32_269, conv_ctest_fwd_fp32_270, 
conv_ctest_fwd_fp32_271, conv_ctest_fwd_fp32_272, 
conv_ctest_fwd_fp32_273, conv_ctest_fwd_fp32_274, 
conv_ctest_fwd_fp32_275, conv_ctest_fwd_fp32_276, 
conv_ctest_fwd_fp32_277, conv_ctest_fwd_fp32_278, 
conv_ctest_fwd_fp32_279, conv_ctest_fwd_fp32_280, 
conv_ctest_fwd_fp32_281, conv_ctest_fwd_fp32_282, 
conv_ctest_fwd_fp32_283, conv_ctest_fwd_fp32_284, 
conv_ctest_fwd_fp32_285, conv_ctest_fwd_fp32_286, 
conv_ctest_fwd_fp32_287, conv_ctest_fwd_fp32_288, 
conv_ctest_fwd_fp32_289, conv_ctest_fwd_fp32_290, 
conv_ctest_fwd_fp32_291, conv_ctest_fwd_fp32_292, 
conv_ctest_fwd_fp32_293, conv_ctest_fwd_fp32_294, 
conv_ctest_fwd_fp32_295, conv_ctest_fwd_fp32_296, 
conv_ctest_fwd_fp32_297, conv_ctest_fwd_fp32_298, 
conv_ctest_fwd_fp32_299, conv_ctest_fwd_fp32_300, 
conv_ctest_fwd_fp32_301, conv_ctest_fwd_fp32_302, 
conv_ctest_fwd_fp32_303, conv_ctest_fwd_fp32_304, 
conv_ctest_fwd_fp32_305, conv_ctest_fwd_fp32_306, 
conv_ctest_fwd_fp32_307, conv_ctest_fwd_fp32_308, 
conv_ctest_fwd_fp32_309, conv_ctest_fwd_fp32_310, 
conv_ctest_fwd_fp32_311, conv_ctest_fwd_fp32_312, 
conv_ctest_fwd_fp32_313, conv_ctest_fwd_fp32_314, 
conv_ctest_fwd_fp32_315, conv_ctest_fwd_fp32_316, 
conv_ctest_fwd_fp32_317, conv_ctest_fwd_fp32_318, 
conv_ctest_fwd_fp32_319, conv_ctest_fwd_fp32_320, 
conv_ctest_fwd_fp32_321, conv_ctest_fwd_fp32_322, 
conv_ctest_fwd_fp32_323, conv_ctest_fwd_fp32_324, 
conv_ctest_fwd_fp32_325, conv_ctest_fwd_fp32_326, 
conv_ctest_fwd_fp32_327, conv_ctest_fwd_fp32_328, 
conv_ctest_fwd_fp32_329, conv_ctest_fwd_fp32_330, 
conv_ctest_fwd_fp32_331, conv_ctest_fwd_fp32_332, 
conv_ctest_fwd_fp32_333, conv_ctest_fwd_fp32_334, 
conv_ctest_fwd_fp32_335, conv_ctest_fwd_fp32_336, 
conv_ctest_fwd_fp32_337, conv_ctest_fwd_fp32_338, 
conv_ctest_fwd_fp32_339, conv_ctest_fwd_fp32_340, 
conv_ctest_fwd_fp32_341, conv_ctest_fwd_fp32_342, 
conv_ctest_fwd_fp32_343, conv_ctest_fwd_fp32_344, 
conv_ctest_fwd_fp32_345, conv_ctest_fwd_fp32_346, 
conv_ctest_fwd_fp32_347, conv_ctest_fwd_fp32_348, 
conv_ctest_fwd_fp32_349, conv_ctest_fwd_fp32_350, 
conv_ctest_fwd_fp32_351, conv_ctest_fwd_fp32_352, 
conv_ctest_fwd_fp32_353, conv_ctest_fwd_fp32_354, 
conv_ctest_fwd_fp32_355, conv_ctest_fwd_fp32_356, 
conv_ctest_fwd_fp32_357, conv_ctest_fwd_fp32_358, 
conv_ctest_fwd_fp32_359, conv_ctest_fwd_fp32_360, 
conv_ctest_fwd_fp32_361, conv_ctest_fwd_fp32_362, 
conv_ctest_fwd_fp32_363, conv_ctest_fwd_fp32_364, 
conv_ctest_fwd_fp32_365, conv_ctest_fwd_fp32_366, 
conv_ctest_fwd_fp32_367, conv_ctest_fwd_fp32_368, 
conv_ctest_fwd_fp32_369, conv_ctest_fwd_fp32_370, 
conv_ctest_fwd_fp32_371, conv_ctest_fwd_fp32_372, 
conv_ctest_fwd_fp32_373, conv_ctest_fwd_fp32_374, 
conv_ctest_fwd_fp32_375, conv_ctest_fwd_fp32_376, 
conv_ctest_fwd_fp32_377, conv_ctest_fwd_fp32_378, 
conv_ctest_fwd_fp32_379, conv_ctest_fwd_fp32_380, 
conv_ctest_fwd_fp32_381, conv_ctest_fwd_fp32_382, 
conv_ctest_fwd_fp32_383, conv_ctest_fwd_fp32_384, 
conv_ctest_fwd_fp32_385, conv_ctest_fwd_fp32_386, 
conv_ctest_fwd_fp32_387, conv_ctest_fwd_fp32_388, 
conv_ctest_fwd_fp32_389, conv_ctest_fwd_fp32_390, 
conv_ctest_fwd_fp32_391, conv_ctest_fwd_fp32_392, 
conv_ctest_fwd_fp32_393, conv_ctest_fwd_fp32_394, 
conv_ctest_fwd_fp32_395, conv_ctest_fwd_fp32_396, 
conv_ctest_fwd_fp32_397, conv_ctest_fwd_fp32_398, 
conv_ctest_fwd_fp32_399, conv_ctest_fwd_fp32_400, 
conv_ctest_fwd_fp32_401, conv_ctest_fwd_fp32_402, 
conv_ctest_fwd_fp32_403, conv_ctest_fwd_fp32_404, 
conv_ctest_fwd_fp32_405, conv_ctest_fwd_fp32_406, 
conv_ctest_fwd_fp32_407, conv_ctest_fwd_fp32_408, 
conv_ctest_fwd_fp32_409, conv_ctest_fwd_fp32_410, 
conv_ctest_fwd_fp32_411, conv_ctest_fwd_fp32_412, 
conv_ctest_fwd_fp32_413, conv_ctest_fwd_fp32_414, 
conv_ctest_fwd_fp32_415, conv_ctest_fwd_fp32_416, 
conv_ctest_fwd_fp32_417, conv_ctest_fwd_fp32_418, 
conv_ctest_fwd_fp32_419, conv_ctest_fwd_fp32_420, 
conv_ctest_fwd_fp32_421, conv_ctest_fwd_fp32_422, 
conv_ctest_fwd_fp32_423, conv_ctest_fwd_fp32_424, 
conv_ctest_fwd_fp32_425, conv_ctest_fwd_fp32_426, 
conv_ctest_fwd_fp32_427, conv_ctest_fwd_fp32_428, 
conv_ctest_fwd_fp32_429, conv_ctest_fwd_fp32_430, 
conv_ctest_fwd_fp32_431, conv_ctest_fwd_fp32_432, 
conv_ctest_fwd_fp32_433, conv_ctest_fwd_fp32_434, 
conv_ctest_fwd_fp32_435, conv_ctest_fwd_fp32_436, 
conv_ctest_fwd_fp32_437, conv_ctest_fwd_fp32_438, 
conv_ctest_fwd_fp32_439, conv_ctest_fwd_fp32_440, 
conv_ctest_fwd_fp32_441, conv_ctest_fwd_fp32_442, 
conv_ctest_fwd_fp32_443, conv_ctest_fwd_fp32_444, 
conv_ctest_fwd_fp32_445, conv_ctest_fwd_fp32_446, 
conv_ctest_fwd_fp32_447, conv_ctest_fwd_fp32_448, 
conv_ctest_fwd_fp32_449, conv_ctest_fwd_fp32_450, 
conv_ctest_fwd_fp32_451, conv_ctest_fwd_fp32_452, 
conv_ctest_fwd_fp32_453, conv_ctest_fwd_fp32_454, 
conv_ctest_fwd_fp32_455, conv_ctest_fwd_fp32_456, 
conv_ctest_fwd_fp32_457, conv_ctest_fwd_fp32_458, 
conv_ctest_fwd_fp32_459, conv_ctest_fwd_fp32_460, 
conv_ctest_fwd_fp32_461, conv_ctest_fwd_fp32_462, 
conv_ctest_fwd_fp32_463, conv_ctest_fwd_fp32_464, 
conv_ctest_fwd_fp32_465, conv_ctest_fwd_fp32_466, 
conv_ctest_fwd_fp32_467, conv_ctest_fwd_fp32_468, 
conv_ctest_fwd_fp32_469, conv_ctest_fwd_fp32_470, 
conv_ctest_fwd_fp32_471, conv_ctest_fwd_fp32_472, 
conv_ctest_fwd_fp32_473, conv_ctest_fwd_fp32_474, 
conv_ctest_fwd_fp32_475, conv_ctest_fwd_fp32_476, 
conv_ctest_fwd_fp32_477, conv_ctest_fwd_fp32_478, 
conv_ctest_fwd_fp32_479, conv_ctest_fwd_fp32_480, 
conv_ctest_fwd_fp32_481, conv_ctest_fwd_fp32_482, 
conv_ctest_fwd_fp32_483, conv_ctest_fwd_fp32_484, 
conv_ctest_fwd_fp32_485, conv_ctest_fwd_fp32_486, 
conv_ctest_fwd_fp32_487, conv_ctest_fwd_fp32_488, 
conv_ctest_fwd_fp32_489, conv_ctest_fwd_fp32_490, 
conv_ctest_fwd_fp32_491, conv_ctest_fwd_fp32_492, 
conv_ctest_fwd_fp32_493, conv_ctest_fwd_fp32_494, 
conv_ctest_fwd_fp32_495, conv_ctest_fwd_fp32_496, 
conv_ctest_fwd_fp32_497, conv_ctest_fwd_fp32_498, 
conv_ctest_fwd_fp32_499, conv_ctest_fwd_fp32_500, 
conv_ctest_fwd_fp32_501, conv_ctest_fwd_fp32_502, 
conv_ctest_fwd_fp32_503, conv_ctest_fwd_fp32_504, 
conv_ctest_fwd_fp32_505, conv_ctest_fwd_fp32_506, 
conv_ctest_fwd_fp32_507, conv_ctest_fwd_fp32_508, 
conv_ctest_fwd_fp32_509, conv_ctest_fwd_fp32_510, 
conv_ctest_fwd_fp32_511, conv_ctest_fwd_fp32_512, 
conv_ctest_fwd_fp32_513, conv_ctest_fwd_fp32_514, 
conv_ctest_fwd_fp32_515, conv_ctest_fwd_fp32_516, 
conv_ctest_fwd_fp32_517, conv_ctest_fwd_fp32_518, 
conv_ctest_fwd_fp32_519, conv_ctest_fwd_fp32_520, 
conv_ctest_fwd_fp32_521, conv_ctest_fwd_fp32_522, 
conv_ctest_fwd_fp32_523, conv_ctest_fwd_fp32_524, 
conv_ctest_fwd_fp32_525, conv_ctest_fwd_fp32_526, 
conv_ctest_fwd_fp32_527, conv_ctest_fwd_fp32_528, 
conv_ctest_fwd_fp32_529, conv_ctest_fwd_fp32_530, 
conv_ctest_fwd_fp32_531, conv_ctest_fwd_fp32_532, 
conv_ctest_fwd_fp32_533, conv_ctest_fwd_fp32_534, 
conv_ctest_fwd_fp32_535, conv_ctest_fwd_fp32_536, 
conv_ctest_fwd_fp32_537, conv_ctest_fwd_fp32_538, 
conv_ctest_fwd_fp32_539, conv_ctest_fwd_fp32_540, 
conv_ctest_fwd_fp32_541, conv_ctest_fwd_fp32_542, 
conv_ctest_fwd_fp32_543, conv_ctest_fwd_fp32_544, 
conv_ctest_fwd_fp32_545, conv_ctest_fwd_fp32_546, 
conv_ctest_fwd_fp32_547, conv_ctest_fwd_fp32_548, 
conv_ctest_fwd_fp32_549, conv_ctest_fwd_fp32_550, 
conv_ctest_fwd_fp32_551, conv_ctest_fwd_fp32_552, 
conv_ctest_fwd_fp32_553, conv_ctest_fwd_fp32_554, 
conv_ctest_fwd_fp32_555, conv_ctest_fwd_fp32_556, 
conv_ctest_fwd_fp32_557, conv_ctest_fwd_fp32_558, 
conv_ctest_fwd_fp32_559, conv_ctest_fwd_fp32_560, 
conv_ctest_fwd_fp32_561, conv_ctest_fwd_fp32_562, 
conv_ctest_fwd_fp32_563, conv_ctest_fwd_fp32_564, 
conv_ctest_fwd_fp32_565, conv_ctest_fwd_fp32_566, 
conv_ctest_fwd_fp32_567, conv_ctest_fwd_fp32_568, 
conv_ctest_fwd_fp32_569, conv_ctest_fwd_fp32_570, 
conv_ctest_fwd_fp32_571, conv_ctest_fwd_fp32_572, 
conv_ctest_fwd_fp32_573, conv_ctest_fwd_fp32_574, 
conv_ctest_fwd_fp32_575, conv_ctest_fwd_fp32_576, 
conv_ctest_fwd_fp32_577, conv_ctest_fwd_fp32_578, 
conv_ctest_fwd_fp32_579, conv_ctest_fwd_fp32_580, 
conv_ctest_fwd_fp32_581, conv_ctest_fwd_fp32_582, 
conv_ctest_fwd_fp32_583, conv_ctest_fwd_fp32_584, 
conv_ctest_fwd_fp32_585, conv_ctest_fwd_fp32_586, 
conv_ctest_fwd_fp32_587, conv_ctest_fwd_fp32_588, 
conv_ctest_fwd_fp32_589, conv_ctest_fwd_fp32_590, 
conv_ctest_fwd_fp32_591, conv_ctest_fwd_fp32_592, 
conv_ctest_fwd_fp32_593, conv_ctest_fwd_fp32_594, 
conv_ctest_fwd_fp32_595, conv_ctest_fwd_fp32_596, 
conv_ctest_fwd_fp32_597, conv_ctest_fwd_fp32_598, 
conv_ctest_fwd_fp32_599, conv_ctest_fwd_fp32_600, 
conv_ctest_fwd_fp32_601, conv_ctest_fwd_fp32_602, 
conv_ctest_fwd_fp32_603, conv_ctest_fwd_fp32_604, 
conv_ctest_fwd_fp32_605, conv_ctest_fwd_fp32_606, 
conv_ctest_fwd_fp32_607, conv_ctest_fwd_fp32_608, 
conv_ctest_fwd_fp32_609, conv_ctest_fwd_fp32_610, 
conv_ctest_fwd_fp32_611, conv_ctest_fwd_fp32_612, 
conv_ctest_fwd_fp32_613, conv_ctest_fwd_fp32_614, 
conv_ctest_fwd_fp32_615, conv_ctest_fwd_fp32_616, 
conv_ctest_fwd_fp32_617, conv_ctest_fwd_fp32_618, 
};

gemm_tuple conv_ctest_fwd_fp16_001 {{10000, 1, 363, 10000, 363, 10000}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_002 {{100, 1, 1008, 100, 1008, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_003 {{100, 1, 1152, 100, 1152, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_004 {{100, 1, 128, 100, 128, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_005 {{100, 1, 1296, 100, 1296, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_006 {{100, 1, 1440, 100, 1440, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_007 {{100, 1, 1600, 100, 1600, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_008 {{100, 1, 1728, 100, 1728, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_009 {{100, 1, 192, 100, 192, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_010 {{100, 1, 2304, 100, 2304, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_011 {{100, 1, 2400, 100, 2400, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_012 {{100, 1, 256, 100, 256, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_013 {{100, 1, 400, 100, 400, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_014 {{100, 1, 4608, 100, 4608, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_015 {{100, 1, 480, 100, 480, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_016 {{100, 1, 4, 100, 4, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_017 {{100, 1, 512, 100, 512, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_018 {{100, 1, 528, 100, 528, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_019 {{100, 1, 576, 100, 576, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_020 {{100, 1, 600, 100, 600, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_021 {{100, 1, 608, 100, 608, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_022 {{100, 1, 64, 100, 64, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_023 {{100, 1, 800, 100, 800, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_024 {{100, 1, 864, 100, 864, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_025 {{100, 1, 9216, 100, 9216, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_026 {{100, 1, 9, 100, 9, 100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_027 {{1024, 1, 128, 1024, 128, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_028 {{1024, 1, 147, 1024, 147, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_029 {{1024, 1, 192, 1024, 192, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_030 {{1024, 1, 256, 1024, 256, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_031 {{1024, 1, 27, 1024, 27, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_032 {{1024, 1, 320, 1024, 320, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_033 {{1024, 1, 363, 1024, 363, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_034 {{1024, 1, 512, 1024, 512, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_035 {{1024, 1, 64, 1024, 64, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_036 {{1024, 1, 75, 1024, 75, 1024}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_037 {{10404, 1, 363, 10404, 363, 10404}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_038 {{10609, 1, 147, 10609, 147, 10609}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_039 {{10816, 1, 147, 10816, 147, 10816}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_040 {{10816, 1, 1600, 10816, 1600, 10816}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_041 {{11025, 1, 147, 11025, 147, 11025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_042 {{11236, 1, 147, 11236, 147, 11236}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_043 {{11449, 1, 147, 11449, 147, 11449}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_044 {{11449, 1, 363, 11449, 363, 11449}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_045 {{11449, 1, 75, 11449, 75, 11449}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_046 {{1156, 1, 27, 1156, 27, 1156}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_047 {{11664, 1, 147, 11664, 147, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_048 {{11664, 1, 1600, 11664, 1600, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_049 {{11664, 1, 363, 11664, 363, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_050 {{11664, 1, 576, 11664, 576, 11664}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_051 {{11881, 1, 147, 11881, 147, 11881}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_052 {{11881, 1, 363, 11881, 363, 11881}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_053 {{11881, 1, 75, 11881, 75, 11881}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_054 {{12100, 1, 147, 12100, 147, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_055 {{12100, 1, 1600, 12100, 1600, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_056 {{12100, 1, 27, 12100, 27, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_057 {{12100, 1, 363, 12100, 363, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_058 {{12100, 1, 576, 12100, 576, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_059 {{12100, 1, 75, 12100, 75, 12100}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_060 {{121, 1, 1024, 121, 1024, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_061 {{121, 1, 1056, 121, 1056, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_062 {{121, 1, 192, 121, 192, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_063 {{121, 1, 2048, 121, 2048, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_064 {{121, 1, 2304, 121, 2304, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_065 {{121, 1, 3456, 121, 3456, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_066 {{121, 1, 363, 121, 363, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_067 {{121, 1, 4, 121, 4, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_068 {{121, 1, 512, 121, 512, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_069 {{121, 1, 75, 121, 75, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_070 {{121, 1, 832, 121, 832, 121}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_071 {{12321, 1, 147, 12321, 147, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_072 {{12321, 1, 27, 12321, 27, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_073 {{12321, 1, 363, 12321, 363, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_074 {{12321, 1, 75, 12321, 75, 12321}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_075 {{12544, 1, 147, 12544, 147, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_076 {{12544, 1, 1600, 12544, 1600, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_077 {{12544, 1, 27, 12544, 27, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_078 {{12544, 1, 363, 12544, 363, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_079 {{12544, 1, 576, 12544, 576, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_080 {{12544, 1, 75, 12544, 75, 12544}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_081 {{12769, 1, 147, 12769, 147, 12769}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_082 {{12769, 1, 27, 12769, 27, 12769}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_083 {{12769, 1, 75, 12769, 75, 12769}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_084 {{12996, 1, 147, 12996, 147, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_085 {{12996, 1, 27, 12996, 27, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_086 {{12996, 1, 363, 12996, 363, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_087 {{12996, 1, 576, 12996, 576, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_088 {{12996, 1, 64, 12996, 64, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_089 {{12996, 1, 75, 12996, 75, 12996}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_090 {{13225, 1, 27, 13225, 27, 13225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_091 {{13225, 1, 75, 13225, 75, 13225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_092 {{13456, 1, 147, 13456, 147, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_093 {{13456, 1, 27, 13456, 27, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_094 {{13456, 1, 363, 13456, 363, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_095 {{13456, 1, 64, 13456, 64, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_096 {{13456, 1, 75, 13456, 75, 13456}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_097 {{13689, 1, 75, 13689, 75, 13689}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_098 {{13924, 1, 27, 13924, 27, 13924}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_099 {{144, 1, 1008, 144, 1008, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_100 {{144, 1, 1024, 144, 1024, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_101 {{144, 1, 1152, 144, 1152, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_102 {{144, 1, 1296, 144, 1296, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_103 {{144, 1, 1440, 144, 1440, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_104 {{144, 1, 1600, 144, 1600, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_105 {{144, 1, 1728, 144, 1728, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_106 {{144, 1, 2304, 144, 2304, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_107 {{144, 1, 2400, 144, 2400, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_108 {{144, 1, 256, 144, 256, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_109 {{144, 1, 363, 144, 363, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_110 {{144, 1, 400, 144, 400, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_111 {{144, 1, 4608, 144, 4608, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_112 {{144, 1, 4, 144, 4, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_113 {{144, 1, 512, 144, 512, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_114 {{144, 1, 576, 144, 576, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_115 {{144, 1, 600, 144, 600, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_116 {{144, 1, 800, 144, 800, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_117 {{144, 1, 864, 144, 864, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_118 {{144, 1, 9216, 144, 9216, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_119 {{144, 1, 9, 144, 9, 144}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_120 {{169, 1, 1152, 169, 1152, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_121 {{169, 1, 147, 169, 147, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_122 {{169, 1, 1600, 169, 1600, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_123 {{169, 1, 1728, 169, 1728, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_124 {{169, 1, 2048, 169, 2048, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_125 {{169, 1, 2304, 169, 2304, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_126 {{169, 1, 2400, 169, 2400, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_127 {{169, 1, 256, 169, 256, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_128 {{169, 1, 3456, 169, 3456, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_129 {{169, 1, 400, 169, 400, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_130 {{169, 1, 4608, 169, 4608, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_131 {{169, 1, 4, 169, 4, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_132 {{169, 1, 576, 169, 576, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_133 {{169, 1, 800, 169, 800, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_134 {{169, 1, 864, 169, 864, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_135 {{169, 1, 9, 169, 9, 169}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_136 {{16, 1, 1024, 16, 1024, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_137 {{16, 1, 1056, 16, 1056, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_138 {{16, 1, 1200, 16, 1200, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_139 {{16, 1, 1440, 16, 1440, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_140 {{16, 1, 1728, 16, 1728, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_141 {{16, 1, 192, 16, 192, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_142 {{16, 1, 2016, 16, 2016, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_143 {{16, 1, 2304, 16, 2304, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_144 {{16, 1, 4608, 16, 4608, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_145 {{16, 1, 4, 16, 4, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_146 {{16, 1, 512, 16, 512, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_147 {{16, 1, 528, 16, 528, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_148 {{16, 1, 576, 16, 576, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_149 {{16, 1, 608, 16, 608, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_150 {{16, 1, 800, 16, 800, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_151 {{16, 1, 832, 16, 832, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_152 {{16, 1, 9216, 16, 9216, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_153 {{16, 1, 9, 16, 9, 16}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_154 {{1860, 1, 4608, 1860, 4608, 1860}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_155 {{1953, 1, 4608, 1953, 4608, 1953}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_156 {{196, 1, 1008, 196, 1008, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_157 {{196, 1, 1024, 196, 1024, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_158 {{196, 1, 1152, 196, 1152, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_159 {{196, 1, 128, 196, 128, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_160 {{196, 1, 1296, 196, 1296, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_161 {{196, 1, 1440, 196, 1440, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_162 {{196, 1, 147, 196, 147, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_163 {{196, 1, 1600, 196, 1600, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_164 {{196, 1, 1728, 196, 1728, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_165 {{196, 1, 192, 196, 192, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_166 {{196, 1, 2304, 196, 2304, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_167 {{196, 1, 2400, 196, 2400, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_168 {{196, 1, 256, 196, 256, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_169 {{196, 1, 27, 196, 27, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_170 {{196, 1, 320, 196, 320, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_171 {{196, 1, 363, 196, 363, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_172 {{196, 1, 400, 196, 400, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_173 {{196, 1, 4608, 196, 4608, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_174 {{196, 1, 480, 196, 480, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_175 {{196, 1, 4, 196, 4, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_176 {{196, 1, 512, 196, 512, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_177 {{196, 1, 528, 196, 528, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_178 {{196, 1, 576, 196, 576, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_179 {{196, 1, 600, 196, 600, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_180 {{196, 1, 608, 196, 608, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_181 {{196, 1, 64, 196, 64, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_182 {{196, 1, 75, 196, 75, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_183 {{196, 1, 800, 196, 800, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_184 {{196, 1, 864, 196, 864, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_185 {{196, 1, 9216, 196, 9216, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_186 {{196, 1, 9, 196, 9, 196}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_187 {{1, 1, 1200, 1, 1200, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_188 {{1, 1, 363, 1, 363, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_189 {{1, 1, 4608, 1, 4608, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_190 {{1, 1, 4, 1, 4, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_191 {{1, 1, 800, 1, 800, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_192 {{1, 1, 9, 1, 9, 1}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_193 {{2048, 1, 4608, 2048, 4608, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_194 {{2048, 1, 480, 2048, 480, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_195 {{2048, 1, 512, 2048, 512, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_196 {{2048, 1, 528, 2048, 528, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_197 {{2048, 1, 832, 2048, 832, 2048}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_198 {{2145, 1, 480, 2145, 480, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_199 {{2145, 1, 512, 2145, 512, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_200 {{2145, 1, 528, 2145, 528, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_201 {{2145, 1, 832, 2145, 832, 2145}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_202 {{2244, 1, 4608, 2244, 4608, 2244}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_203 {{225, 1, 128, 225, 128, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_204 {{225, 1, 1600, 225, 1600, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_205 {{225, 1, 192, 225, 192, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_206 {{225, 1, 2048, 225, 2048, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_207 {{225, 1, 2304, 225, 2304, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_208 {{225, 1, 2400, 225, 2400, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_209 {{225, 1, 256, 225, 256, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_210 {{225, 1, 27, 225, 27, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_211 {{225, 1, 320, 225, 320, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_212 {{225, 1, 3456, 225, 3456, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_213 {{225, 1, 400, 225, 400, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_214 {{225, 1, 4, 225, 4, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_215 {{225, 1, 512, 225, 512, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_216 {{225, 1, 64, 225, 64, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_217 {{225, 1, 75, 225, 75, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_218 {{225, 1, 800, 225, 800, 225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_219 {{2304, 1, 1600, 2304, 1600, 2304}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_220 {{2345, 1, 480, 2345, 480, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_221 {{2345, 1, 512, 2345, 512, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_222 {{2345, 1, 528, 2345, 528, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_223 {{2345, 1, 832, 2345, 832, 2345}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_224 {{256, 1, 1008, 256, 1008, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_225 {{256, 1, 1024, 256, 1024, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_226 {{256, 1, 1152, 256, 1152, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_227 {{256, 1, 128, 256, 128, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_228 {{256, 1, 1296, 256, 1296, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_229 {{256, 1, 1440, 256, 1440, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_230 {{256, 1, 147, 256, 147, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_231 {{256, 1, 1728, 256, 1728, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_232 {{256, 1, 192, 256, 192, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_233 {{256, 1, 2304, 256, 2304, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_234 {{256, 1, 256, 256, 256, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_235 {{256, 1, 27, 256, 27, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_236 {{256, 1, 363, 256, 363, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_237 {{256, 1, 4608, 256, 4608, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_238 {{256, 1, 480, 256, 480, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_239 {{256, 1, 4, 256, 4, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_240 {{256, 1, 512, 256, 512, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_241 {{256, 1, 528, 256, 528, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_242 {{256, 1, 576, 256, 576, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_243 {{256, 1, 608, 256, 608, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_244 {{256, 1, 64, 256, 64, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_245 {{256, 1, 75, 256, 75, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_246 {{256, 1, 800, 256, 800, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_247 {{256, 1, 864, 256, 864, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_248 {{256, 1, 9, 256, 9, 256}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_249 {{25, 1, 1008, 25, 1008, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_250 {{25, 1, 1024, 25, 1024, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_251 {{25, 1, 1056, 25, 1056, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_252 {{25, 1, 1152, 25, 1152, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_253 {{25, 1, 1200, 25, 1200, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_254 {{25, 1, 1296, 25, 1296, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_255 {{25, 1, 1440, 25, 1440, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_256 {{25, 1, 1600, 25, 1600, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_257 {{25, 1, 1728, 25, 1728, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_258 {{25, 1, 192, 25, 192, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_259 {{25, 1, 2016, 25, 2016, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_260 {{25, 1, 2304, 25, 2304, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_261 {{25, 1, 2400, 25, 2400, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_262 {{25, 1, 3456, 25, 3456, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_263 {{25, 1, 400, 25, 400, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_264 {{25, 1, 4608, 25, 4608, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_265 {{25, 1, 4, 25, 4, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_266 {{25, 1, 512, 25, 512, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_267 {{25, 1, 528, 25, 528, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_268 {{25, 1, 576, 25, 576, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_269 {{25, 1, 600, 25, 600, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_270 {{25, 1, 608, 25, 608, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_271 {{25, 1, 800, 25, 800, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_272 {{25, 1, 832, 25, 832, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_273 {{25, 1, 864, 25, 864, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_274 {{25, 1, 9216, 25, 9216, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_275 {{25, 1, 9, 25, 9, 25}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_276 {{2601, 1, 1600, 2601, 1600, 2601}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_277 {{2704, 1, 1152, 2704, 1152, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_278 {{2704, 1, 1600, 2704, 1600, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_279 {{2704, 1, 2304, 2704, 2304, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_280 {{2704, 1, 576, 2704, 576, 2704}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_281 {{289, 1, 128, 289, 128, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_282 {{289, 1, 192, 289, 192, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_283 {{289, 1, 256, 289, 256, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_284 {{289, 1, 320, 289, 320, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_285 {{289, 1, 4, 289, 4, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_286 {{289, 1, 512, 289, 512, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_287 {{289, 1, 64, 289, 64, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_288 {{289, 1, 75, 289, 75, 289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_289 {{2916, 1, 1152, 2916, 1152, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_290 {{2916, 1, 1600, 2916, 1600, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_291 {{2916, 1, 2304, 2916, 2304, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_292 {{2916, 1, 576, 2916, 576, 2916}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_293 {{3025, 1, 1600, 3025, 1600, 3025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_294 {{3025, 1, 576, 3025, 576, 3025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_295 {{3136, 1, 1152, 3136, 1152, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_296 {{3136, 1, 1600, 3136, 1600, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_297 {{3136, 1, 2304, 3136, 2304, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_298 {{3136, 1, 576, 3136, 576, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_299 {{3136, 1, 64, 3136, 64, 3136}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_300 {{3249, 1, 1600, 3249, 1600, 3249}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_301 {{3249, 1, 64, 3249, 64, 3249}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_302 {{324, 1, 128, 324, 128, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_303 {{324, 1, 192, 324, 192, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_304 {{324, 1, 256, 324, 256, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_305 {{324, 1, 27, 324, 27, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_306 {{324, 1, 480, 324, 480, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_307 {{324, 1, 512, 324, 512, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_308 {{324, 1, 528, 324, 528, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_309 {{324, 1, 576, 324, 576, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_310 {{324, 1, 608, 324, 608, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_311 {{324, 1, 64, 324, 64, 324}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_312 {{33540, 1, 480, 33540, 480, 33540}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_313 {{3364, 1, 1152, 3364, 1152, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_314 {{3364, 1, 128, 3364, 128, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_315 {{3364, 1, 2304, 3364, 2304, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_316 {{3364, 1, 256, 3364, 256, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_317 {{3364, 1, 576, 3364, 576, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_318 {{3364, 1, 64, 3364, 64, 3364}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_319 {{34320, 1, 480, 34320, 480, 34320}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_320 {{3481, 1, 64, 3481, 64, 3481}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_321 {{3600, 1, 128, 3600, 128, 3600}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_322 {{3600, 1, 256, 3600, 256, 3600}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_323 {{3600, 1, 64, 3600, 64, 3600}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_324 {{361, 1, 1600, 361, 1600, 361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_325 {{361, 1, 2400, 361, 2400, 361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_326 {{36, 1, 1008, 36, 1008, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_327 {{36, 1, 1024, 36, 1024, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_328 {{36, 1, 1152, 36, 1152, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_329 {{36, 1, 1296, 36, 1296, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_330 {{36, 1, 1440, 36, 1440, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_331 {{36, 1, 1600, 36, 1600, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_332 {{36, 1, 1728, 36, 1728, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_333 {{36, 1, 2016, 36, 2016, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_334 {{36, 1, 2048, 36, 2048, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_335 {{36, 1, 2304, 36, 2304, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_336 {{36, 1, 2400, 36, 2400, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_337 {{36, 1, 256, 36, 256, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_338 {{36, 1, 3456, 36, 3456, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_339 {{36, 1, 400, 36, 400, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_340 {{36, 1, 4608, 36, 4608, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_341 {{36, 1, 4, 36, 4, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_342 {{36, 1, 512, 36, 512, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_343 {{36, 1, 528, 36, 528, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_344 {{36, 1, 576, 36, 576, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_345 {{36, 1, 600, 36, 600, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_346 {{36, 1, 608, 36, 608, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_347 {{36, 1, 800, 36, 800, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_348 {{36, 1, 864, 36, 864, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_349 {{36, 1, 9216, 36, 9216, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_350 {{36, 1, 9, 36, 9, 36}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_351 {{400, 1, 147, 400, 147, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_352 {{400, 1, 1600, 400, 1600, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_353 {{400, 1, 2400, 400, 2400, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_354 {{400, 1, 400, 400, 400, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_355 {{400, 1, 800, 400, 800, 400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_356 {{41616, 1, 363, 41616, 363, 41616}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_357 {{42849, 1, 363, 42849, 363, 42849}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_358 {{44521, 1, 363, 44521, 363, 44521}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_359 {{44944, 1, 147, 44944, 147, 44944}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_360 {{45796, 1, 363, 45796, 363, 45796}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_361 {{46225, 1, 147, 46225, 147, 46225}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_362 {{46656, 1, 363, 46656, 363, 46656}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_363 {{46656, 1, 75, 46656, 75, 46656}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_364 {{47089, 1, 363, 47089, 363, 47089}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_365 {{47524, 1, 147, 47524, 147, 47524}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_366 {{47524, 1, 363, 47524, 363, 47524}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_367 {{47961, 1, 147, 47961, 147, 47961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_368 {{47961, 1, 363, 47961, 363, 47961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_369 {{47961, 1, 75, 47961, 75, 47961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_370 {{48400, 1, 147, 48400, 147, 48400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_371 {{48400, 1, 27, 48400, 27, 48400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_372 {{48400, 1, 75, 48400, 75, 48400}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_373 {{484, 1, 363, 484, 363, 484}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_374 {{48841, 1, 147, 48841, 147, 48841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_375 {{48841, 1, 363, 48841, 363, 48841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_376 {{49284, 1, 147, 49284, 147, 49284}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_377 {{49284, 1, 27, 49284, 27, 49284}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_378 {{49284, 1, 75, 49284, 75, 49284}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_379 {{49729, 1, 147, 49729, 147, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_380 {{49729, 1, 27, 49729, 27, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_381 {{49729, 1, 363, 49729, 363, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_382 {{49729, 1, 75, 49729, 75, 49729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_383 {{49, 1, 1008, 49, 1008, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_384 {{49, 1, 1024, 49, 1024, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_385 {{49, 1, 1056, 49, 1056, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_386 {{49, 1, 1152, 49, 1152, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_387 {{49, 1, 1200, 49, 1200, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_388 {{49, 1, 128, 49, 128, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_389 {{49, 1, 1296, 49, 1296, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_390 {{49, 1, 1440, 49, 1440, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_391 {{49, 1, 147, 49, 147, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_392 {{49, 1, 1600, 49, 1600, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_393 {{49, 1, 1728, 49, 1728, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_394 {{49, 1, 192, 49, 192, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_395 {{49, 1, 2016, 49, 2016, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_396 {{49, 1, 2048, 49, 2048, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_397 {{49, 1, 2304, 49, 2304, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_398 {{49, 1, 2400, 49, 2400, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_399 {{49, 1, 256, 49, 256, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_400 {{49, 1, 3456, 49, 3456, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_401 {{49, 1, 400, 49, 400, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_402 {{49, 1, 4608, 49, 4608, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_403 {{49, 1, 480, 49, 480, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_404 {{49, 1, 4, 49, 4, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_405 {{49, 1, 512, 49, 512, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_406 {{49, 1, 528, 49, 528, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_407 {{49, 1, 576, 49, 576, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_408 {{49, 1, 600, 49, 600, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_409 {{49, 1, 608, 49, 608, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_410 {{49, 1, 64, 49, 64, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_411 {{49, 1, 800, 49, 800, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_412 {{49, 1, 832, 49, 832, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_413 {{49, 1, 864, 49, 864, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_414 {{49, 1, 9216, 49, 9216, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_415 {{49, 1, 9, 49, 9, 49}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_416 {{4, 1, 1200, 4, 1200, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_417 {{4, 1, 1440, 4, 1440, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_418 {{4, 1, 1600, 4, 1600, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_419 {{4, 1, 1728, 4, 1728, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_420 {{4, 1, 2016, 4, 2016, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_421 {{4, 1, 2400, 4, 2400, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_422 {{4, 1, 363, 4, 363, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_423 {{4, 1, 400, 4, 400, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_424 {{4, 1, 4608, 4, 4608, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_425 {{4, 1, 4, 4, 4, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_426 {{4, 1, 512, 4, 512, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_427 {{4, 1, 528, 4, 528, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_428 {{4, 1, 576, 4, 576, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_429 {{4, 1, 600, 4, 600, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_430 {{4, 1, 608, 4, 608, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_431 {{4, 1, 800, 4, 800, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_432 {{4, 1, 9216, 4, 9216, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_433 {{4, 1, 9, 4, 9, 4}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_434 {{50176, 1, 147, 50176, 147, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_435 {{50176, 1, 27, 50176, 27, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_436 {{50176, 1, 363, 50176, 363, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_437 {{50176, 1, 75, 50176, 75, 50176}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_438 {{50625, 1, 147, 50625, 147, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_439 {{50625, 1, 27, 50625, 27, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_440 {{50625, 1, 363, 50625, 363, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_441 {{50625, 1, 75, 50625, 75, 50625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_442 {{51076, 1, 27, 51076, 27, 51076}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_443 {{51529, 1, 147, 51529, 147, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_444 {{51529, 1, 27, 51529, 27, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_445 {{51529, 1, 363, 51529, 363, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_446 {{51529, 1, 75, 51529, 75, 51529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_447 {{52441, 1, 147, 52441, 147, 52441}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_448 {{52441, 1, 27, 52441, 27, 52441}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_449 {{52441, 1, 75, 52441, 75, 52441}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_450 {{529, 1, 1600, 529, 1600, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_451 {{529, 1, 2400, 529, 2400, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_452 {{529, 1, 576, 529, 576, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_453 {{529, 1, 864, 529, 864, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_454 {{529, 1, 9, 529, 9, 529}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_455 {{53361, 1, 147, 53361, 147, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_456 {{53361, 1, 27, 53361, 27, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_457 {{53361, 1, 363, 53361, 363, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_458 {{53361, 1, 75, 53361, 75, 53361}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_459 {{54289, 1, 27, 54289, 27, 54289}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_460 {{576, 1, 1152, 576, 1152, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_461 {{576, 1, 1600, 576, 1600, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_462 {{576, 1, 1728, 576, 1728, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_463 {{576, 1, 2304, 576, 2304, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_464 {{576, 1, 2400, 576, 2400, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_465 {{576, 1, 363, 576, 363, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_466 {{576, 1, 400, 576, 400, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_467 {{576, 1, 4608, 576, 4608, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_468 {{576, 1, 576, 576, 576, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_469 {{576, 1, 75, 576, 75, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_470 {{576, 1, 800, 576, 800, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_471 {{576, 1, 864, 576, 864, 576}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_472 {{625, 1, 1600, 625, 1600, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_473 {{625, 1, 2400, 625, 2400, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_474 {{625, 1, 4, 625, 4, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_475 {{625, 1, 576, 625, 576, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_476 {{625, 1, 864, 625, 864, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_477 {{625, 1, 9, 625, 9, 625}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_478 {{64, 1, 128, 64, 128, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_479 {{64, 1, 147, 64, 147, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_480 {{64, 1, 1600, 64, 1600, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_481 {{64, 1, 192, 64, 192, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_482 {{64, 1, 2304, 64, 2304, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_483 {{64, 1, 2400, 64, 2400, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_484 {{64, 1, 256, 64, 256, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_485 {{64, 1, 400, 64, 400, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_486 {{64, 1, 4608, 64, 4608, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_487 {{64, 1, 480, 64, 480, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_488 {{64, 1, 4, 64, 4, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_489 {{64, 1, 512, 64, 512, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_490 {{64, 1, 528, 64, 528, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_491 {{64, 1, 576, 64, 576, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_492 {{64, 1, 600, 64, 600, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_493 {{64, 1, 608, 64, 608, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_494 {{64, 1, 64, 64, 64, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_495 {{64, 1, 800, 64, 800, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_496 {{64, 1, 9216, 64, 9216, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_497 {{64, 1, 9, 64, 9, 64}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_498 {{676, 1, 1152, 676, 1152, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_499 {{676, 1, 147, 676, 147, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_500 {{676, 1, 1600, 676, 1600, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_501 {{676, 1, 1728, 676, 1728, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_502 {{676, 1, 2304, 676, 2304, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_503 {{676, 1, 2400, 676, 2400, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_504 {{676, 1, 363, 676, 363, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_505 {{676, 1, 400, 676, 400, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_506 {{676, 1, 4608, 676, 4608, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_507 {{676, 1, 4, 676, 4, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_508 {{676, 1, 576, 676, 576, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_509 {{676, 1, 800, 676, 800, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_510 {{676, 1, 864, 676, 864, 676}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_511 {{729, 1, 1152, 729, 1152, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_512 {{729, 1, 1600, 729, 1600, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_513 {{729, 1, 2304, 729, 2304, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_514 {{729, 1, 2400, 729, 2400, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_515 {{729, 1, 4, 729, 4, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_516 {{729, 1, 576, 729, 576, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_517 {{729, 1, 864, 729, 864, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_518 {{729, 1, 9, 729, 9, 729}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_519 {{7440, 1, 4608, 7440, 4608, 7440}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_520 {{7812, 1, 4608, 7812, 4608, 7812}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_521 {{784, 1, 1152, 784, 1152, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_522 {{784, 1, 128, 784, 128, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_523 {{784, 1, 147, 784, 147, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_524 {{784, 1, 1600, 784, 1600, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_525 {{784, 1, 1728, 784, 1728, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_526 {{784, 1, 2304, 784, 2304, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_527 {{784, 1, 2400, 784, 2400, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_528 {{784, 1, 256, 784, 256, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_529 {{784, 1, 27, 784, 27, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_530 {{784, 1, 400, 784, 400, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_531 {{784, 1, 4608, 784, 4608, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_532 {{784, 1, 4, 784, 4, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_533 {{784, 1, 576, 784, 576, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_534 {{784, 1, 64, 784, 64, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_535 {{784, 1, 75, 784, 75, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_536 {{784, 1, 800, 784, 800, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_537 {{784, 1, 864, 784, 864, 784}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_538 {{8192, 1, 4608, 8192, 4608, 8192}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_539 {{8192, 1, 480, 8192, 480, 8192}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_540 {{81, 1, 1008, 81, 1008, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_541 {{81, 1, 1024, 81, 1024, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_542 {{81, 1, 1056, 81, 1056, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_543 {{81, 1, 1152, 81, 1152, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_544 {{81, 1, 1296, 81, 1296, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_545 {{81, 1, 1440, 81, 1440, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_546 {{81, 1, 1600, 81, 1600, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_547 {{81, 1, 1728, 81, 1728, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_548 {{81, 1, 192, 81, 192, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_549 {{81, 1, 2016, 81, 2016, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_550 {{81, 1, 2048, 81, 2048, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_551 {{81, 1, 2304, 81, 2304, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_552 {{81, 1, 2400, 81, 2400, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_553 {{81, 1, 256, 81, 256, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_554 {{81, 1, 3456, 81, 3456, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_555 {{81, 1, 400, 81, 400, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_556 {{81, 1, 4608, 81, 4608, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_557 {{81, 1, 4, 81, 4, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_558 {{81, 1, 512, 81, 512, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_559 {{81, 1, 576, 81, 576, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_560 {{81, 1, 800, 81, 800, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_561 {{81, 1, 832, 81, 832, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_562 {{81, 1, 864, 81, 864, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_563 {{81, 1, 9216, 81, 9216, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_564 {{81, 1, 9, 81, 9, 81}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_565 {{8385, 1, 480, 8385, 480, 8385}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_566 {{841, 1, 128, 841, 128, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_567 {{841, 1, 1600, 841, 1600, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_568 {{841, 1, 256, 841, 256, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_569 {{841, 1, 576, 841, 576, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_570 {{841, 1, 64, 841, 64, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_571 {{841, 1, 864, 841, 864, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_572 {{841, 1, 9, 841, 9, 841}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_573 {{8580, 1, 4608, 8580, 4608, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_574 {{8580, 1, 480, 8580, 480, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_575 {{8580, 1, 512, 8580, 512, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_576 {{8580, 1, 528, 8580, 528, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_577 {{8580, 1, 832, 8580, 832, 8580}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_578 {{8777, 1, 480, 8777, 480, 8777}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_579 {{8976, 1, 480, 8976, 480, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_580 {{8976, 1, 512, 8976, 512, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_581 {{8976, 1, 528, 8976, 528, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_582 {{8976, 1, 832, 8976, 832, 8976}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_583 {{900, 1, 1152, 900, 1152, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_584 {{900, 1, 128, 900, 128, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_585 {{900, 1, 147, 900, 147, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_586 {{900, 1, 1728, 900, 1728, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_587 {{900, 1, 192, 900, 192, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_588 {{900, 1, 2304, 900, 2304, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_589 {{900, 1, 256, 900, 256, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_590 {{900, 1, 27, 900, 27, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_591 {{900, 1, 320, 900, 320, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_592 {{900, 1, 4608, 900, 4608, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_593 {{900, 1, 4, 900, 4, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_594 {{900, 1, 512, 900, 512, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_595 {{900, 1, 576, 900, 576, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_596 {{900, 1, 64, 900, 64, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_597 {{900, 1, 75, 900, 75, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_598 {{900, 1, 864, 900, 864, 900}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_599 {{9025, 1, 363, 9025, 363, 9025}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_600 {{9409, 1, 363, 9409, 363, 9409}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_601 {{9604, 1, 363, 9604, 363, 9604}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_602 {{961, 1, 128, 961, 128, 961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_603 {{961, 1, 256, 961, 256, 961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_604 {{961, 1, 64, 961, 64, 961}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_605 {{9801, 1, 363, 9801, 363, 9801}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_606 {{9, 1, 1200, 9, 1200, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_607 {{9, 1, 1440, 9, 1440, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_608 {{9, 1, 1728, 9, 1728, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_609 {{9, 1, 2016, 9, 2016, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_610 {{9, 1, 4608, 9, 4608, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_611 {{9, 1, 4, 9, 4, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_612 {{9, 1, 512, 9, 512, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_613 {{9, 1, 528, 9, 528, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_614 {{9, 1, 576, 9, 576, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_615 {{9, 1, 608, 9, 608, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_616 {{9, 1, 800, 9, 800, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_617 {{9, 1, 9216, 9, 9216, 9}, {1, 0}, {'N', 'N'}};
gemm_tuple conv_ctest_fwd_fp16_618 {{9, 1, 9, 9, 9, 9}, {1, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_ctest_fwd_fp16 = {
conv_ctest_fwd_fp16_001, conv_ctest_fwd_fp16_002, 
conv_ctest_fwd_fp16_003, conv_ctest_fwd_fp16_004, 
conv_ctest_fwd_fp16_005, conv_ctest_fwd_fp16_006, 
conv_ctest_fwd_fp16_007, conv_ctest_fwd_fp16_008, 
conv_ctest_fwd_fp16_009, conv_ctest_fwd_fp16_010, 
conv_ctest_fwd_fp16_011, conv_ctest_fwd_fp16_012, 
conv_ctest_fwd_fp16_013, conv_ctest_fwd_fp16_014, 
conv_ctest_fwd_fp16_015, conv_ctest_fwd_fp16_016, 
conv_ctest_fwd_fp16_017, conv_ctest_fwd_fp16_018, 
conv_ctest_fwd_fp16_019, conv_ctest_fwd_fp16_020, 
conv_ctest_fwd_fp16_021, conv_ctest_fwd_fp16_022, 
conv_ctest_fwd_fp16_023, conv_ctest_fwd_fp16_024, 
conv_ctest_fwd_fp16_025, conv_ctest_fwd_fp16_026, 
conv_ctest_fwd_fp16_027, conv_ctest_fwd_fp16_028, 
conv_ctest_fwd_fp16_029, conv_ctest_fwd_fp16_030, 
conv_ctest_fwd_fp16_031, conv_ctest_fwd_fp16_032, 
conv_ctest_fwd_fp16_033, conv_ctest_fwd_fp16_034, 
conv_ctest_fwd_fp16_035, conv_ctest_fwd_fp16_036, 
conv_ctest_fwd_fp16_037, conv_ctest_fwd_fp16_038, 
conv_ctest_fwd_fp16_039, conv_ctest_fwd_fp16_040, 
conv_ctest_fwd_fp16_041, conv_ctest_fwd_fp16_042, 
conv_ctest_fwd_fp16_043, conv_ctest_fwd_fp16_044, 
conv_ctest_fwd_fp16_045, conv_ctest_fwd_fp16_046, 
conv_ctest_fwd_fp16_047, conv_ctest_fwd_fp16_048, 
conv_ctest_fwd_fp16_049, conv_ctest_fwd_fp16_050, 
conv_ctest_fwd_fp16_051, conv_ctest_fwd_fp16_052, 
conv_ctest_fwd_fp16_053, conv_ctest_fwd_fp16_054, 
conv_ctest_fwd_fp16_055, conv_ctest_fwd_fp16_056, 
conv_ctest_fwd_fp16_057, conv_ctest_fwd_fp16_058, 
conv_ctest_fwd_fp16_059, conv_ctest_fwd_fp16_060, 
conv_ctest_fwd_fp16_061, conv_ctest_fwd_fp16_062, 
conv_ctest_fwd_fp16_063, conv_ctest_fwd_fp16_064, 
conv_ctest_fwd_fp16_065, conv_ctest_fwd_fp16_066, 
conv_ctest_fwd_fp16_067, conv_ctest_fwd_fp16_068, 
conv_ctest_fwd_fp16_069, conv_ctest_fwd_fp16_070, 
conv_ctest_fwd_fp16_071, conv_ctest_fwd_fp16_072, 
conv_ctest_fwd_fp16_073, conv_ctest_fwd_fp16_074, 
conv_ctest_fwd_fp16_075, conv_ctest_fwd_fp16_076, 
conv_ctest_fwd_fp16_077, conv_ctest_fwd_fp16_078, 
conv_ctest_fwd_fp16_079, conv_ctest_fwd_fp16_080, 
conv_ctest_fwd_fp16_081, conv_ctest_fwd_fp16_082, 
conv_ctest_fwd_fp16_083, conv_ctest_fwd_fp16_084, 
conv_ctest_fwd_fp16_085, conv_ctest_fwd_fp16_086, 
conv_ctest_fwd_fp16_087, conv_ctest_fwd_fp16_088, 
conv_ctest_fwd_fp16_089, conv_ctest_fwd_fp16_090, 
conv_ctest_fwd_fp16_091, conv_ctest_fwd_fp16_092, 
conv_ctest_fwd_fp16_093, conv_ctest_fwd_fp16_094, 
conv_ctest_fwd_fp16_095, conv_ctest_fwd_fp16_096, 
conv_ctest_fwd_fp16_097, conv_ctest_fwd_fp16_098, 
conv_ctest_fwd_fp16_099, conv_ctest_fwd_fp16_100, 
conv_ctest_fwd_fp16_101, conv_ctest_fwd_fp16_102, 
conv_ctest_fwd_fp16_103, conv_ctest_fwd_fp16_104, 
conv_ctest_fwd_fp16_105, conv_ctest_fwd_fp16_106, 
conv_ctest_fwd_fp16_107, conv_ctest_fwd_fp16_108, 
conv_ctest_fwd_fp16_109, conv_ctest_fwd_fp16_110, 
conv_ctest_fwd_fp16_111, conv_ctest_fwd_fp16_112, 
conv_ctest_fwd_fp16_113, conv_ctest_fwd_fp16_114, 
conv_ctest_fwd_fp16_115, conv_ctest_fwd_fp16_116, 
conv_ctest_fwd_fp16_117, conv_ctest_fwd_fp16_118, 
conv_ctest_fwd_fp16_119, conv_ctest_fwd_fp16_120, 
conv_ctest_fwd_fp16_121, conv_ctest_fwd_fp16_122, 
conv_ctest_fwd_fp16_123, conv_ctest_fwd_fp16_124, 
conv_ctest_fwd_fp16_125, conv_ctest_fwd_fp16_126, 
conv_ctest_fwd_fp16_127, conv_ctest_fwd_fp16_128, 
conv_ctest_fwd_fp16_129, conv_ctest_fwd_fp16_130, 
conv_ctest_fwd_fp16_131, conv_ctest_fwd_fp16_132, 
conv_ctest_fwd_fp16_133, conv_ctest_fwd_fp16_134, 
conv_ctest_fwd_fp16_135, conv_ctest_fwd_fp16_136, 
conv_ctest_fwd_fp16_137, conv_ctest_fwd_fp16_138, 
conv_ctest_fwd_fp16_139, conv_ctest_fwd_fp16_140, 
conv_ctest_fwd_fp16_141, conv_ctest_fwd_fp16_142, 
conv_ctest_fwd_fp16_143, conv_ctest_fwd_fp16_144, 
conv_ctest_fwd_fp16_145, conv_ctest_fwd_fp16_146, 
conv_ctest_fwd_fp16_147, conv_ctest_fwd_fp16_148, 
conv_ctest_fwd_fp16_149, conv_ctest_fwd_fp16_150, 
conv_ctest_fwd_fp16_151, conv_ctest_fwd_fp16_152, 
conv_ctest_fwd_fp16_153, conv_ctest_fwd_fp16_154, 
conv_ctest_fwd_fp16_155, conv_ctest_fwd_fp16_156, 
conv_ctest_fwd_fp16_157, conv_ctest_fwd_fp16_158, 
conv_ctest_fwd_fp16_159, conv_ctest_fwd_fp16_160, 
conv_ctest_fwd_fp16_161, conv_ctest_fwd_fp16_162, 
conv_ctest_fwd_fp16_163, conv_ctest_fwd_fp16_164, 
conv_ctest_fwd_fp16_165, conv_ctest_fwd_fp16_166, 
conv_ctest_fwd_fp16_167, conv_ctest_fwd_fp16_168, 
conv_ctest_fwd_fp16_169, conv_ctest_fwd_fp16_170, 
conv_ctest_fwd_fp16_171, conv_ctest_fwd_fp16_172, 
conv_ctest_fwd_fp16_173, conv_ctest_fwd_fp16_174, 
conv_ctest_fwd_fp16_175, conv_ctest_fwd_fp16_176, 
conv_ctest_fwd_fp16_177, conv_ctest_fwd_fp16_178, 
conv_ctest_fwd_fp16_179, conv_ctest_fwd_fp16_180, 
conv_ctest_fwd_fp16_181, conv_ctest_fwd_fp16_182, 
conv_ctest_fwd_fp16_183, conv_ctest_fwd_fp16_184, 
conv_ctest_fwd_fp16_185, conv_ctest_fwd_fp16_186, 
conv_ctest_fwd_fp16_187, conv_ctest_fwd_fp16_188, 
conv_ctest_fwd_fp16_189, conv_ctest_fwd_fp16_190, 
conv_ctest_fwd_fp16_191, conv_ctest_fwd_fp16_192, 
conv_ctest_fwd_fp16_193, conv_ctest_fwd_fp16_194, 
conv_ctest_fwd_fp16_195, conv_ctest_fwd_fp16_196, 
conv_ctest_fwd_fp16_197, conv_ctest_fwd_fp16_198, 
conv_ctest_fwd_fp16_199, conv_ctest_fwd_fp16_200, 
conv_ctest_fwd_fp16_201, conv_ctest_fwd_fp16_202, 
conv_ctest_fwd_fp16_203, conv_ctest_fwd_fp16_204, 
conv_ctest_fwd_fp16_205, conv_ctest_fwd_fp16_206, 
conv_ctest_fwd_fp16_207, conv_ctest_fwd_fp16_208, 
conv_ctest_fwd_fp16_209, conv_ctest_fwd_fp16_210, 
conv_ctest_fwd_fp16_211, conv_ctest_fwd_fp16_212, 
conv_ctest_fwd_fp16_213, conv_ctest_fwd_fp16_214, 
conv_ctest_fwd_fp16_215, conv_ctest_fwd_fp16_216, 
conv_ctest_fwd_fp16_217, conv_ctest_fwd_fp16_218, 
conv_ctest_fwd_fp16_219, conv_ctest_fwd_fp16_220, 
conv_ctest_fwd_fp16_221, conv_ctest_fwd_fp16_222, 
conv_ctest_fwd_fp16_223, conv_ctest_fwd_fp16_224, 
conv_ctest_fwd_fp16_225, conv_ctest_fwd_fp16_226, 
conv_ctest_fwd_fp16_227, conv_ctest_fwd_fp16_228, 
conv_ctest_fwd_fp16_229, conv_ctest_fwd_fp16_230, 
conv_ctest_fwd_fp16_231, conv_ctest_fwd_fp16_232, 
conv_ctest_fwd_fp16_233, conv_ctest_fwd_fp16_234, 
conv_ctest_fwd_fp16_235, conv_ctest_fwd_fp16_236, 
conv_ctest_fwd_fp16_237, conv_ctest_fwd_fp16_238, 
conv_ctest_fwd_fp16_239, conv_ctest_fwd_fp16_240, 
conv_ctest_fwd_fp16_241, conv_ctest_fwd_fp16_242, 
conv_ctest_fwd_fp16_243, conv_ctest_fwd_fp16_244, 
conv_ctest_fwd_fp16_245, conv_ctest_fwd_fp16_246, 
conv_ctest_fwd_fp16_247, conv_ctest_fwd_fp16_248, 
conv_ctest_fwd_fp16_249, conv_ctest_fwd_fp16_250, 
conv_ctest_fwd_fp16_251, conv_ctest_fwd_fp16_252, 
conv_ctest_fwd_fp16_253, conv_ctest_fwd_fp16_254, 
conv_ctest_fwd_fp16_255, conv_ctest_fwd_fp16_256, 
conv_ctest_fwd_fp16_257, conv_ctest_fwd_fp16_258, 
conv_ctest_fwd_fp16_259, conv_ctest_fwd_fp16_260, 
conv_ctest_fwd_fp16_261, conv_ctest_fwd_fp16_262, 
conv_ctest_fwd_fp16_263, conv_ctest_fwd_fp16_264, 
conv_ctest_fwd_fp16_265, conv_ctest_fwd_fp16_266, 
conv_ctest_fwd_fp16_267, conv_ctest_fwd_fp16_268, 
conv_ctest_fwd_fp16_269, conv_ctest_fwd_fp16_270, 
conv_ctest_fwd_fp16_271, conv_ctest_fwd_fp16_272, 
conv_ctest_fwd_fp16_273, conv_ctest_fwd_fp16_274, 
conv_ctest_fwd_fp16_275, conv_ctest_fwd_fp16_276, 
conv_ctest_fwd_fp16_277, conv_ctest_fwd_fp16_278, 
conv_ctest_fwd_fp16_279, conv_ctest_fwd_fp16_280, 
conv_ctest_fwd_fp16_281, conv_ctest_fwd_fp16_282, 
conv_ctest_fwd_fp16_283, conv_ctest_fwd_fp16_284, 
conv_ctest_fwd_fp16_285, conv_ctest_fwd_fp16_286, 
conv_ctest_fwd_fp16_287, conv_ctest_fwd_fp16_288, 
conv_ctest_fwd_fp16_289, conv_ctest_fwd_fp16_290, 
conv_ctest_fwd_fp16_291, conv_ctest_fwd_fp16_292, 
conv_ctest_fwd_fp16_293, conv_ctest_fwd_fp16_294, 
conv_ctest_fwd_fp16_295, conv_ctest_fwd_fp16_296, 
conv_ctest_fwd_fp16_297, conv_ctest_fwd_fp16_298, 
conv_ctest_fwd_fp16_299, conv_ctest_fwd_fp16_300, 
conv_ctest_fwd_fp16_301, conv_ctest_fwd_fp16_302, 
conv_ctest_fwd_fp16_303, conv_ctest_fwd_fp16_304, 
conv_ctest_fwd_fp16_305, conv_ctest_fwd_fp16_306, 
conv_ctest_fwd_fp16_307, conv_ctest_fwd_fp16_308, 
conv_ctest_fwd_fp16_309, conv_ctest_fwd_fp16_310, 
conv_ctest_fwd_fp16_311, conv_ctest_fwd_fp16_312, 
conv_ctest_fwd_fp16_313, conv_ctest_fwd_fp16_314, 
conv_ctest_fwd_fp16_315, conv_ctest_fwd_fp16_316, 
conv_ctest_fwd_fp16_317, conv_ctest_fwd_fp16_318, 
conv_ctest_fwd_fp16_319, conv_ctest_fwd_fp16_320, 
conv_ctest_fwd_fp16_321, conv_ctest_fwd_fp16_322, 
conv_ctest_fwd_fp16_323, conv_ctest_fwd_fp16_324, 
conv_ctest_fwd_fp16_325, conv_ctest_fwd_fp16_326, 
conv_ctest_fwd_fp16_327, conv_ctest_fwd_fp16_328, 
conv_ctest_fwd_fp16_329, conv_ctest_fwd_fp16_330, 
conv_ctest_fwd_fp16_331, conv_ctest_fwd_fp16_332, 
conv_ctest_fwd_fp16_333, conv_ctest_fwd_fp16_334, 
conv_ctest_fwd_fp16_335, conv_ctest_fwd_fp16_336, 
conv_ctest_fwd_fp16_337, conv_ctest_fwd_fp16_338, 
conv_ctest_fwd_fp16_339, conv_ctest_fwd_fp16_340, 
conv_ctest_fwd_fp16_341, conv_ctest_fwd_fp16_342, 
conv_ctest_fwd_fp16_343, conv_ctest_fwd_fp16_344, 
conv_ctest_fwd_fp16_345, conv_ctest_fwd_fp16_346, 
conv_ctest_fwd_fp16_347, conv_ctest_fwd_fp16_348, 
conv_ctest_fwd_fp16_349, conv_ctest_fwd_fp16_350, 
conv_ctest_fwd_fp16_351, conv_ctest_fwd_fp16_352, 
conv_ctest_fwd_fp16_353, conv_ctest_fwd_fp16_354, 
conv_ctest_fwd_fp16_355, conv_ctest_fwd_fp16_356, 
conv_ctest_fwd_fp16_357, conv_ctest_fwd_fp16_358, 
conv_ctest_fwd_fp16_359, conv_ctest_fwd_fp16_360, 
conv_ctest_fwd_fp16_361, conv_ctest_fwd_fp16_362, 
conv_ctest_fwd_fp16_363, conv_ctest_fwd_fp16_364, 
conv_ctest_fwd_fp16_365, conv_ctest_fwd_fp16_366, 
conv_ctest_fwd_fp16_367, conv_ctest_fwd_fp16_368, 
conv_ctest_fwd_fp16_369, conv_ctest_fwd_fp16_370, 
conv_ctest_fwd_fp16_371, conv_ctest_fwd_fp16_372, 
conv_ctest_fwd_fp16_373, conv_ctest_fwd_fp16_374, 
conv_ctest_fwd_fp16_375, conv_ctest_fwd_fp16_376, 
conv_ctest_fwd_fp16_377, conv_ctest_fwd_fp16_378, 
conv_ctest_fwd_fp16_379, conv_ctest_fwd_fp16_380, 
conv_ctest_fwd_fp16_381, conv_ctest_fwd_fp16_382, 
conv_ctest_fwd_fp16_383, conv_ctest_fwd_fp16_384, 
conv_ctest_fwd_fp16_385, conv_ctest_fwd_fp16_386, 
conv_ctest_fwd_fp16_387, conv_ctest_fwd_fp16_388, 
conv_ctest_fwd_fp16_389, conv_ctest_fwd_fp16_390, 
conv_ctest_fwd_fp16_391, conv_ctest_fwd_fp16_392, 
conv_ctest_fwd_fp16_393, conv_ctest_fwd_fp16_394, 
conv_ctest_fwd_fp16_395, conv_ctest_fwd_fp16_396, 
conv_ctest_fwd_fp16_397, conv_ctest_fwd_fp16_398, 
conv_ctest_fwd_fp16_399, conv_ctest_fwd_fp16_400, 
conv_ctest_fwd_fp16_401, conv_ctest_fwd_fp16_402, 
conv_ctest_fwd_fp16_403, conv_ctest_fwd_fp16_404, 
conv_ctest_fwd_fp16_405, conv_ctest_fwd_fp16_406, 
conv_ctest_fwd_fp16_407, conv_ctest_fwd_fp16_408, 
conv_ctest_fwd_fp16_409, conv_ctest_fwd_fp16_410, 
conv_ctest_fwd_fp16_411, conv_ctest_fwd_fp16_412, 
conv_ctest_fwd_fp16_413, conv_ctest_fwd_fp16_414, 
conv_ctest_fwd_fp16_415, conv_ctest_fwd_fp16_416, 
conv_ctest_fwd_fp16_417, conv_ctest_fwd_fp16_418, 
conv_ctest_fwd_fp16_419, conv_ctest_fwd_fp16_420, 
conv_ctest_fwd_fp16_421, conv_ctest_fwd_fp16_422, 
conv_ctest_fwd_fp16_423, conv_ctest_fwd_fp16_424, 
conv_ctest_fwd_fp16_425, conv_ctest_fwd_fp16_426, 
conv_ctest_fwd_fp16_427, conv_ctest_fwd_fp16_428, 
conv_ctest_fwd_fp16_429, conv_ctest_fwd_fp16_430, 
conv_ctest_fwd_fp16_431, conv_ctest_fwd_fp16_432, 
conv_ctest_fwd_fp16_433, conv_ctest_fwd_fp16_434, 
conv_ctest_fwd_fp16_435, conv_ctest_fwd_fp16_436, 
conv_ctest_fwd_fp16_437, conv_ctest_fwd_fp16_438, 
conv_ctest_fwd_fp16_439, conv_ctest_fwd_fp16_440, 
conv_ctest_fwd_fp16_441, conv_ctest_fwd_fp16_442, 
conv_ctest_fwd_fp16_443, conv_ctest_fwd_fp16_444, 
conv_ctest_fwd_fp16_445, conv_ctest_fwd_fp16_446, 
conv_ctest_fwd_fp16_447, conv_ctest_fwd_fp16_448, 
conv_ctest_fwd_fp16_449, conv_ctest_fwd_fp16_450, 
conv_ctest_fwd_fp16_451, conv_ctest_fwd_fp16_452, 
conv_ctest_fwd_fp16_453, conv_ctest_fwd_fp16_454, 
conv_ctest_fwd_fp16_455, conv_ctest_fwd_fp16_456, 
conv_ctest_fwd_fp16_457, conv_ctest_fwd_fp16_458, 
conv_ctest_fwd_fp16_459, conv_ctest_fwd_fp16_460, 
conv_ctest_fwd_fp16_461, conv_ctest_fwd_fp16_462, 
conv_ctest_fwd_fp16_463, conv_ctest_fwd_fp16_464, 
conv_ctest_fwd_fp16_465, conv_ctest_fwd_fp16_466, 
conv_ctest_fwd_fp16_467, conv_ctest_fwd_fp16_468, 
conv_ctest_fwd_fp16_469, conv_ctest_fwd_fp16_470, 
conv_ctest_fwd_fp16_471, conv_ctest_fwd_fp16_472, 
conv_ctest_fwd_fp16_473, conv_ctest_fwd_fp16_474, 
conv_ctest_fwd_fp16_475, conv_ctest_fwd_fp16_476, 
conv_ctest_fwd_fp16_477, conv_ctest_fwd_fp16_478, 
conv_ctest_fwd_fp16_479, conv_ctest_fwd_fp16_480, 
conv_ctest_fwd_fp16_481, conv_ctest_fwd_fp16_482, 
conv_ctest_fwd_fp16_483, conv_ctest_fwd_fp16_484, 
conv_ctest_fwd_fp16_485, conv_ctest_fwd_fp16_486, 
conv_ctest_fwd_fp16_487, conv_ctest_fwd_fp16_488, 
conv_ctest_fwd_fp16_489, conv_ctest_fwd_fp16_490, 
conv_ctest_fwd_fp16_491, conv_ctest_fwd_fp16_492, 
conv_ctest_fwd_fp16_493, conv_ctest_fwd_fp16_494, 
conv_ctest_fwd_fp16_495, conv_ctest_fwd_fp16_496, 
conv_ctest_fwd_fp16_497, conv_ctest_fwd_fp16_498, 
conv_ctest_fwd_fp16_499, conv_ctest_fwd_fp16_500, 
conv_ctest_fwd_fp16_501, conv_ctest_fwd_fp16_502, 
conv_ctest_fwd_fp16_503, conv_ctest_fwd_fp16_504, 
conv_ctest_fwd_fp16_505, conv_ctest_fwd_fp16_506, 
conv_ctest_fwd_fp16_507, conv_ctest_fwd_fp16_508, 
conv_ctest_fwd_fp16_509, conv_ctest_fwd_fp16_510, 
conv_ctest_fwd_fp16_511, conv_ctest_fwd_fp16_512, 
conv_ctest_fwd_fp16_513, conv_ctest_fwd_fp16_514, 
conv_ctest_fwd_fp16_515, conv_ctest_fwd_fp16_516, 
conv_ctest_fwd_fp16_517, conv_ctest_fwd_fp16_518, 
conv_ctest_fwd_fp16_519, conv_ctest_fwd_fp16_520, 
conv_ctest_fwd_fp16_521, conv_ctest_fwd_fp16_522, 
conv_ctest_fwd_fp16_523, conv_ctest_fwd_fp16_524, 
conv_ctest_fwd_fp16_525, conv_ctest_fwd_fp16_526, 
conv_ctest_fwd_fp16_527, conv_ctest_fwd_fp16_528, 
conv_ctest_fwd_fp16_529, conv_ctest_fwd_fp16_530, 
conv_ctest_fwd_fp16_531, conv_ctest_fwd_fp16_532, 
conv_ctest_fwd_fp16_533, conv_ctest_fwd_fp16_534, 
conv_ctest_fwd_fp16_535, conv_ctest_fwd_fp16_536, 
conv_ctest_fwd_fp16_537, conv_ctest_fwd_fp16_538, 
conv_ctest_fwd_fp16_539, conv_ctest_fwd_fp16_540, 
conv_ctest_fwd_fp16_541, conv_ctest_fwd_fp16_542, 
conv_ctest_fwd_fp16_543, conv_ctest_fwd_fp16_544, 
conv_ctest_fwd_fp16_545, conv_ctest_fwd_fp16_546, 
conv_ctest_fwd_fp16_547, conv_ctest_fwd_fp16_548, 
conv_ctest_fwd_fp16_549, conv_ctest_fwd_fp16_550, 
conv_ctest_fwd_fp16_551, conv_ctest_fwd_fp16_552, 
conv_ctest_fwd_fp16_553, conv_ctest_fwd_fp16_554, 
conv_ctest_fwd_fp16_555, conv_ctest_fwd_fp16_556, 
conv_ctest_fwd_fp16_557, conv_ctest_fwd_fp16_558, 
conv_ctest_fwd_fp16_559, conv_ctest_fwd_fp16_560, 
conv_ctest_fwd_fp16_561, conv_ctest_fwd_fp16_562, 
conv_ctest_fwd_fp16_563, conv_ctest_fwd_fp16_564, 
conv_ctest_fwd_fp16_565, conv_ctest_fwd_fp16_566, 
conv_ctest_fwd_fp16_567, conv_ctest_fwd_fp16_568, 
conv_ctest_fwd_fp16_569, conv_ctest_fwd_fp16_570, 
conv_ctest_fwd_fp16_571, conv_ctest_fwd_fp16_572, 
conv_ctest_fwd_fp16_573, conv_ctest_fwd_fp16_574, 
conv_ctest_fwd_fp16_575, conv_ctest_fwd_fp16_576, 
conv_ctest_fwd_fp16_577, conv_ctest_fwd_fp16_578, 
conv_ctest_fwd_fp16_579, conv_ctest_fwd_fp16_580, 
conv_ctest_fwd_fp16_581, conv_ctest_fwd_fp16_582, 
conv_ctest_fwd_fp16_583, conv_ctest_fwd_fp16_584, 
conv_ctest_fwd_fp16_585, conv_ctest_fwd_fp16_586, 
conv_ctest_fwd_fp16_587, conv_ctest_fwd_fp16_588, 
conv_ctest_fwd_fp16_589, conv_ctest_fwd_fp16_590, 
conv_ctest_fwd_fp16_591, conv_ctest_fwd_fp16_592, 
conv_ctest_fwd_fp16_593, conv_ctest_fwd_fp16_594, 
conv_ctest_fwd_fp16_595, conv_ctest_fwd_fp16_596, 
conv_ctest_fwd_fp16_597, conv_ctest_fwd_fp16_598, 
conv_ctest_fwd_fp16_599, conv_ctest_fwd_fp16_600, 
conv_ctest_fwd_fp16_601, conv_ctest_fwd_fp16_602, 
conv_ctest_fwd_fp16_603, conv_ctest_fwd_fp16_604, 
conv_ctest_fwd_fp16_605, conv_ctest_fwd_fp16_606, 
conv_ctest_fwd_fp16_607, conv_ctest_fwd_fp16_608, 
conv_ctest_fwd_fp16_609, conv_ctest_fwd_fp16_610, 
conv_ctest_fwd_fp16_611, conv_ctest_fwd_fp16_612, 
conv_ctest_fwd_fp16_613, conv_ctest_fwd_fp16_614, 
conv_ctest_fwd_fp16_615, conv_ctest_fwd_fp16_616, 
conv_ctest_fwd_fp16_617, conv_ctest_fwd_fp16_618, 
};

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

Arguments setup_gemm_sweep_arguments(gemm_sweep_tuple tup)
{
    Arguments arg;

    arg.M                      = std::get<0>(tup);
    arg.N                      = std::get<1>(tup);
    arg.K                      = std::get<2>(tup);
    vector<double> alpha_beta  = std::get<3>(tup);
    vector<char> transA_transB = std::get<4>(tup);

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    arg.alpha = alpha_beta[0];
    arg.beta  = alpha_beta[1];

    arg.transA_option = transA_transB[0];
    arg.transB_option = transA_transB[1];

    arg.lda = arg.transA_option == 'N' ? arg.M : arg.K;
    arg.ldb = arg.transB_option == 'N' ? arg.K : arg.N;
    arg.ldc = arg.M;

    arg.timing = 0;

    return arg;
}

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

class parameterized_gemm_sweep : public ::TestWithParam<gemm_sweep_tuple>
{
    protected:
    parameterized_gemm_sweep() {}
    virtual ~parameterized_gemm_sweep() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_gemm : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_gemm() {}
    virtual ~parameterized_gemm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm_sweep, half)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_sweep_arguments(GetParam());

    rocblas_status status = testing_gemm_sweep<rocblas_half>(arg);

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

TEST_P(parameterized_gemm_sweep, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_sweep_arguments(GetParam());

    rocblas_status status = testing_gemm_sweep<float>(arg);

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

TEST_P(parameterized_gemm_sweep, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_gemm_sweep_arguments(GetParam());

    rocblas_status status = testing_gemm_sweep<double>(arg);

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

class parameterized_chunk_gemm : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_chunk_gemm() {}
    virtual ~parameterized_chunk_gemm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_chunk_gemm, float)
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

class parameterized_gemm_half : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_gemm_half() {}
    virtual ~parameterized_gemm_half() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm_half, half)
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

class parameterized_gemm_float : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_gemm_float() {}
    virtual ~parameterized_gemm_float() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm_float, float)
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

class parameterized_gemm_double : public ::TestWithParam<gemm_tuple>
{
    protected:
    parameterized_gemm_double() {}
    virtual ~parameterized_gemm_double() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(parameterized_gemm_double, double)
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

TEST(pre_checkin_blas3_bad_arg, gemm_half) { testing_gemm_bad_arg<rocblas_half>(); }

TEST(pre_checkin_blas3_bad_arg, gemm_float) { testing_gemm_bad_arg<float>(); }

TEST(pre_checkin_blas3_bad_arg, gemm_double) { testing_gemm_bad_arg<double>(); }

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, K, lda, ldb, ldc}, {alpha, beta}, {transA, transB} }

// INSTANTIATE_TEST_CASE_P(rocblas_gemm_beta_eq_0, parameterized_gemm_NaN,
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_NaN,
                        parameterized_gemm_NaN,
                        Combine(ValuesIn(NaN_matrix_size_range),
                                ValuesIn(NaN_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(quick_blas3_small,
                        parameterized_gemm,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(full_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(quick_blas3_small,
                        parameterized_gemm_half,
                        Combine(ValuesIn(small_matrix_size_range),
                                ValuesIn(full_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_medium,
                        parameterized_gemm,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(full_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_medium,
                        parameterized_gemm_half,
                        Combine(ValuesIn(medium_matrix_size_range),
                                ValuesIn(full_alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_large,
                        parameterized_gemm,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_large,
                        parameterized_gemm_half,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_chunk,
                        parameterized_chunk_gemm,
                        Combine(ValuesIn(chunk_matrix_size_range),
                                ValuesIn(alpha_beta_2_3_range),
                                ValuesIn(transA_transB_range)));

// clang-format off
INSTANTIATE_TEST_CASE_P(nightly_blas3_deepbench_sizes, parameterized_gemm, ValuesIn(deepbench_vec));

INSTANTIATE_TEST_CASE_P(nightly_blas3_fixed_bug_sizes, parameterized_gemm, ValuesIn(fixed_bug_vec));
INSTANTIATE_TEST_CASE_P(nightly_blas3_fixed_bug_sizes, parameterized_gemm_half, ValuesIn(fixed_bug_vec));

INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_fwd_fp32, parameterized_gemm_float, ValuesIn(conv_resnet50_fwd_fp32));
INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_fwd_fp16, parameterized_gemm_half, ValuesIn(conv_resnet50_fwd_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_bwdwrw_fp32, parameterized_gemm_float, ValuesIn(conv_resnet50_bwdwrw_fp32));
INSTANTIATE_TEST_CASE_P(known_bug_conv_resnet50_bwdwrw_fp16, parameterized_gemm_half, ValuesIn(conv_resnet50_bwdwrw_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_bwddata_fp32, parameterized_gemm_float, ValuesIn(conv_resnet50_bwddata_fp32));
INSTANTIATE_TEST_CASE_P(nightly_conv_resnet50_bwddata_fp16, parameterized_gemm_half, ValuesIn(conv_resnet50_bwddata_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_fwd_fp32, parameterized_gemm_float, ValuesIn(conv_inception4_fwd_fp32));
INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_fwd_fp16, parameterized_gemm_half, ValuesIn(conv_inception4_fwd_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_bwdwrw_fp32, parameterized_gemm_float, ValuesIn(conv_inception4_bwdwrw_fp32));
INSTANTIATE_TEST_CASE_P(known_bug_conv_inception4_bwdwrw_fp16, parameterized_gemm_half, ValuesIn(conv_inception4_bwdwrw_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_bwddata_fp32, parameterized_gemm_float, ValuesIn(conv_inception4_bwddata_fp32));
INSTANTIATE_TEST_CASE_P(nightly_conv_inception4_bwddata_fp16, parameterized_gemm_half, ValuesIn(conv_inception4_bwddata_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_bwddata_fp32, parameterized_gemm_float, ValuesIn(conv_ctest_bwddata_fp32));
INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_bwddata_fp16, parameterized_gemm_half, ValuesIn(conv_ctest_bwddata_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_bwdwrw_fp32, parameterized_gemm_float, ValuesIn(conv_ctest_bwdwrw_fp32));
INSTANTIATE_TEST_CASE_P(known_bug_conv_ctest_bwdwrw_fp16, parameterized_gemm_half, ValuesIn(conv_ctest_bwdwrw_fp16));

INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_fwd_fp32, parameterized_gemm_float, ValuesIn(conv_ctest_fwd_fp32));
INSTANTIATE_TEST_CASE_P(nightly_conv_ctest_fwd_fp16, parameterized_gemm_half, ValuesIn(conv_ctest_fwd_fp16));

// clang-format on

INSTANTIATE_TEST_CASE_P(nightly_blas3_deepbench_sizes,
                        parameterized_gemm_half,
                        ValuesIn(deepbench_vec));

//--- sweep tests
INSTANTIATE_TEST_CASE_P(quick_blas3_sweep_1_8,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_1_8),
                                ValuesIn(size_range_1_8),
                                ValuesIn(size_range_1_8),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_9_12,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_9_12),
                                ValuesIn(size_range_9_12),
                                ValuesIn(size_range_9_12),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_13_16,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_13_16),
                                ValuesIn(size_range_13_16),
                                ValuesIn(size_range_13_16),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_17_20,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_17_20),
                                ValuesIn(size_range_17_20),
                                ValuesIn(size_range_17_20),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_20_23,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_20_23),
                                ValuesIn(size_range_20_23),
                                ValuesIn(size_range_20_23),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_24_27,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_24_27),
                                ValuesIn(size_range_24_27),
                                ValuesIn(size_range_24_27),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_28_31,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_28_31),
                                ValuesIn(size_range_28_31),
                                ValuesIn(size_range_28_31),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));
//---32
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_32,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_32),
                                ValuesIn(size_range_32),
                                ValuesIn(size_range_32),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_sweep_32_9_129,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_9_129),
                                ValuesIn(size_range_32),
                                ValuesIn(size_range_32),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//---48
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_48,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_48),
                                ValuesIn(size_range_48),
                                ValuesIn(size_range_48),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_sweep_48_9_129,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_48),
                                ValuesIn(size_range_9_129),
                                ValuesIn(size_range_48),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//---64
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_64,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_64),
                                ValuesIn(size_range_64),
                                ValuesIn(size_range_64),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(nightly_blas3_sweep_64_9_129,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_64),
                                ValuesIn(size_range_64),
                                ValuesIn(size_range_9_129),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(quick_blas3_sweep_64_1_4_5_8,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_64),
                                ValuesIn(size_range_1_4),
                                ValuesIn(size_range_5_8),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(quick_blas3_sweep_5_8_64_1_4,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_5_8),
                                ValuesIn(size_range_64),
                                ValuesIn(size_range_1_4),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(quick_blas3_sweep_1_4_5_8_64,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_1_4),
                                ValuesIn(size_range_5_8),
                                ValuesIn(size_range_64),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//--- 96
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_96,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_96),
                                ValuesIn(size_range_96),
                                ValuesIn(size_range_96),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//--- 128
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_128,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_128),
                                ValuesIn(size_range_128),
                                ValuesIn(size_range_128),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//--- 256
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_256,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_256),
                                ValuesIn(size_range_256),
                                ValuesIn(size_range_256),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_256_9_12_13_16,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_256),
                                ValuesIn(size_range_9_12),
                                ValuesIn(size_range_13_16),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_13_16_256_9_12,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_13_16),
                                ValuesIn(size_range_256),
                                ValuesIn(size_range_9_12),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_9_12_13_16_256,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_9_12),
                                ValuesIn(size_range_13_16),
                                ValuesIn(size_range_256),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//--- 512
INSTANTIATE_TEST_CASE_P(pre_checkin_blas3_sweep_512,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_512),
                                ValuesIn(size_range_512),
                                ValuesIn(size_range_512),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));

//--- 1024
INSTANTIATE_TEST_CASE_P(nightly_blas3_sweep_1024,
                        parameterized_gemm_sweep,
                        Combine(ValuesIn(size_range_1024),
                                ValuesIn(size_range_1024),
                                ValuesIn(size_range_1024),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(transA_transB_range)));
