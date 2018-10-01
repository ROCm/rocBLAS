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
conv_resnet50_fwd_fp32_001, conv_resnet50_fwd_fp32_002, conv_resnet50_fwd_fp32_003, conv_resnet50_fwd_fp32_004,
conv_resnet50_fwd_fp32_005, conv_resnet50_fwd_fp32_006, conv_resnet50_fwd_fp32_007, conv_resnet50_fwd_fp32_008,
conv_resnet50_fwd_fp32_009, conv_resnet50_fwd_fp32_010, conv_resnet50_fwd_fp32_011, conv_resnet50_fwd_fp32_012,
conv_resnet50_fwd_fp32_013, conv_resnet50_fwd_fp32_014, conv_resnet50_fwd_fp32_015, conv_resnet50_fwd_fp32_016,
};

gemm_tuple conv_resnet50_fwd_fp16_001 {{12544, 1024, 256, 12544, 256, 12544}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_002 {{12544, 1024, 512, 12544, 512, 12544}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_003 {{12544, 256, 1024, 12544, 1024, 12544}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_004 {{12544, 256, 512, 12544, 512, 12544}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_005 {{12544, 64, 147, 12544, 147, 12544}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_006 {{196, 256, 2304, 196, 2304, 196}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_007 {{3025, 64, 576, 3025, 576, 3025}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_008 {{3136, 2048, 1024, 3136, 1024, 3136}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_009 {{3136, 2048, 512, 3136, 512, 3136}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_010 {{3136, 512, 1024, 3136, 1024, 3136}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_011 {{3136, 512, 2048, 3136, 2048, 3136}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_012 {{3136, 64, 576, 3136, 576, 3136}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_013 {{49, 512, 4608, 49, 4608, 49}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_014 {{50176, 128, 256, 50176, 256, 50176}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_015 {{50176, 512, 256, 50176, 256, 50176}, {15360, 0}, {'N', 'N'}};
gemm_tuple conv_resnet50_fwd_fp16_016 {{784, 128, 1152, 784, 1152, 784}, {15360, 0}, {'N', 'N'}};

const vector<gemm_tuple> conv_resnet50_fwd_fp16 = {
conv_resnet50_fwd_fp16_001, conv_resnet50_fwd_fp16_002, conv_resnet50_fwd_fp16_003, conv_resnet50_fwd_fp16_004, 
conv_resnet50_fwd_fp16_005, conv_resnet50_fwd_fp16_006, conv_resnet50_fwd_fp16_007, conv_resnet50_fwd_fp16_008, 
conv_resnet50_fwd_fp16_009, conv_resnet50_fwd_fp16_010, conv_resnet50_fwd_fp16_011, conv_resnet50_fwd_fp16_012, 
conv_resnet50_fwd_fp16_013, conv_resnet50_fwd_fp16_014, conv_resnet50_fwd_fp16_015, conv_resnet50_fwd_fp16_016, 
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
conv_resnet50_bwdwrw_fp32_001, conv_resnet50_bwdwrw_fp32_002, conv_resnet50_bwdwrw_fp32_003, conv_resnet50_bwdwrw_fp32_004, 
conv_resnet50_bwdwrw_fp32_005, conv_resnet50_bwdwrw_fp32_006, conv_resnet50_bwdwrw_fp32_007, conv_resnet50_bwdwrw_fp32_008, 
conv_resnet50_bwdwrw_fp32_009, conv_resnet50_bwdwrw_fp32_010, conv_resnet50_bwdwrw_fp32_011, conv_resnet50_bwdwrw_fp32_012, 
conv_resnet50_bwdwrw_fp32_013, conv_resnet50_bwdwrw_fp32_014, conv_resnet50_bwdwrw_fp32_015, conv_resnet50_bwdwrw_fp32_016, 
conv_resnet50_bwdwrw_fp32_017, conv_resnet50_bwdwrw_fp32_018, conv_resnet50_bwdwrw_fp32_019, conv_resnet50_bwdwrw_fp32_020, 
conv_resnet50_bwdwrw_fp32_021, conv_resnet50_bwdwrw_fp32_022, conv_resnet50_bwdwrw_fp32_023, conv_resnet50_bwdwrw_fp32_024, 
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
gemm_tuple conv_resnet50_bwdwrw_fp16_025 {{1024, 2048, 49, 49, 49, 1024}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_026 {{1024, 256, 196, 196, 196, 1024}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_027 {{1024, 512, 49, 49, 49, 1024}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_028 {{1152, 128, 784, 784, 784, 1152}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_029 {{128, 512, 784, 784, 784, 128}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_030 {{147, 64, 12544, 12544, 12544, 147}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_031 {{2048, 512, 49, 49, 49, 2048}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_032 {{2304, 256, 196, 196, 196, 2304}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_033 {{256, 1024, 196, 196, 196, 256}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_034 {{256, 128, 784, 784, 784, 256}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_035 {{256, 512, 784, 784, 784, 256}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_036 {{256, 64, 3025, 3025, 3025, 256}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_037 {{256, 64, 3136, 3136, 3136, 256}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_038 {{4608, 512, 49, 49, 49, 4608}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_039 {{512, 1024, 196, 196, 196, 512}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_040 {{512, 128, 784, 784, 784, 512}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_041 {{512, 2048, 49, 49, 49, 512}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_042 {{512, 256, 196, 196, 196, 512}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_043 {{576, 64, 3025, 3025, 3025, 576}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_044 {{576, 64, 3136, 3136, 3136, 576}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_045 {{64, 256, 3025, 3025, 3025, 64}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_046 {{64, 256, 3136, 3136, 3136, 64}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_047 {{64, 64, 3025, 3025, 3025, 64}, {15360, 15360}, {'T', 'N'}};
gemm_tuple conv_resnet50_bwdwrw_fp16_048 {{64, 64, 3136, 3136, 3136, 64}, {15360, 15360}, {'T', 'N'}};

const vector<gemm_tuple> conv_resnet50_bwdwrw_fp16 = {
conv_resnet50_bwdwrw_fp16_001, conv_resnet50_bwdwrw_fp16_002, conv_resnet50_bwdwrw_fp16_003, conv_resnet50_bwdwrw_fp16_004, 
conv_resnet50_bwdwrw_fp16_005, conv_resnet50_bwdwrw_fp16_006, conv_resnet50_bwdwrw_fp16_007, conv_resnet50_bwdwrw_fp16_008, 
conv_resnet50_bwdwrw_fp16_009, conv_resnet50_bwdwrw_fp16_010, conv_resnet50_bwdwrw_fp16_011, conv_resnet50_bwdwrw_fp16_012, 
conv_resnet50_bwdwrw_fp16_013, conv_resnet50_bwdwrw_fp16_014, conv_resnet50_bwdwrw_fp16_015, conv_resnet50_bwdwrw_fp16_016, 
conv_resnet50_bwdwrw_fp16_017, conv_resnet50_bwdwrw_fp16_018, conv_resnet50_bwdwrw_fp16_019, conv_resnet50_bwdwrw_fp16_020, 
conv_resnet50_bwdwrw_fp16_021, conv_resnet50_bwdwrw_fp16_022, conv_resnet50_bwdwrw_fp16_023, conv_resnet50_bwdwrw_fp16_024, 
conv_resnet50_bwdwrw_fp16_025, conv_resnet50_bwdwrw_fp16_026, conv_resnet50_bwdwrw_fp16_027, conv_resnet50_bwdwrw_fp16_028, 
conv_resnet50_bwdwrw_fp16_029, conv_resnet50_bwdwrw_fp16_030, conv_resnet50_bwdwrw_fp16_031, conv_resnet50_bwdwrw_fp16_032, 
conv_resnet50_bwdwrw_fp16_033, conv_resnet50_bwdwrw_fp16_034, conv_resnet50_bwdwrw_fp16_035, conv_resnet50_bwdwrw_fp16_036, 
conv_resnet50_bwdwrw_fp16_037, conv_resnet50_bwdwrw_fp16_038, conv_resnet50_bwdwrw_fp16_039, conv_resnet50_bwdwrw_fp16_040, 
conv_resnet50_bwdwrw_fp16_041, conv_resnet50_bwdwrw_fp16_042, conv_resnet50_bwdwrw_fp16_043, conv_resnet50_bwdwrw_fp16_044, 
conv_resnet50_bwdwrw_fp16_045, conv_resnet50_bwdwrw_fp16_046, conv_resnet50_bwdwrw_fp16_047, conv_resnet50_bwdwrw_fp16_048, 
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

gemm_tuple conv_resnet50_bwddata_fp16_001 {{12544, 147, 64, 12544, 147, 12544}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_002 {{12544, 512, 1024, 12544, 512, 12544}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_003 {{12544, 512, 256, 12544, 512, 12544}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_004 {{196, 2304, 256, 196, 2304, 196}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_005 {{3025, 576, 64, 3025, 576, 3025}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_006 {{3136, 1024, 2048, 3136, 1024, 3136}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_007 {{3136, 1024, 512, 3136, 1024, 3136}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_008 {{3136, 576, 64, 3136, 576, 3136}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_009 {{49, 4608, 512, 49, 4608, 49}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_010 {{50176, 256, 128, 50176, 256, 50176}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_011 {{50176, 256, 512, 50176, 256, 50176}, {15360, 0}, {'N', 'T'}};
gemm_tuple conv_resnet50_bwddata_fp16_012 {{784, 1152, 128, 784, 1152, 784}, {15360, 0}, {'N', 'T'}};

const vector<gemm_tuple> conv_resnet50_bwddata_fp16 = {
conv_resnet50_bwddata_fp16_001, conv_resnet50_bwddata_fp16_002, 
conv_resnet50_bwddata_fp16_003, conv_resnet50_bwddata_fp16_004, 
conv_resnet50_bwddata_fp16_005, conv_resnet50_bwddata_fp16_006, 
conv_resnet50_bwddata_fp16_007, conv_resnet50_bwddata_fp16_008, 
conv_resnet50_bwddata_fp16_009, conv_resnet50_bwddata_fp16_010, 
conv_resnet50_bwddata_fp16_011, conv_resnet50_bwddata_fp16_012, 
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
