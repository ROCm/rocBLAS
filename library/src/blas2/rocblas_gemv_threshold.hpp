/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#include "rocblas/rocblas.h"

// Threshold values of (M, N) in gfx908 and gfx906 below which the threads per block should be 512 or less to get better performance
constexpr int gemvn_gfx908_threshold        = 15000;
constexpr int zgemvn_gfx908_threshold       = 18000;
constexpr int gemvn_gfx906_threshold        = 6000;
constexpr int dgemvn_gfx906_lower_threshold = 15000;
constexpr int dgemvn_gfx906_upper_threshold = 24000;
constexpr int gemvt_threshold               = 6000;

// Threshold values of (M, N) in gfx1030
constexpr int sgemvt_gfx1030_threshold = 4000;
