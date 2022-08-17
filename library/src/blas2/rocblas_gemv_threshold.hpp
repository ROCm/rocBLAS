/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#include "rocblas.h"

// Threshold values of (M, N) in gfx908 and gfx906 below which the threads per block should be 512 or less to get better performance
constexpr int gemvn_gfx908_threshold        = 15000;
constexpr int zgemvn_gfx908_threshold       = 18000;
constexpr int gemvn_gfx906_threshold        = 6000;
constexpr int dgemvn_gfx906_lower_threshold = 15000;
constexpr int dgemvn_gfx906_upper_threshold = 24000;
constexpr int gemvt_threshold               = 6000;

// Threshold values of (M, N) in gfx10 and gfx11
constexpr int sgemvt_gfx_arch_10_11_threshold = 4000;

// Double buffered load optimized for single and double precision
constexpr int xgemvn_gfx908_upper_threshold = 3000;
constexpr int xgemvn_gfx90a_upper_threshold = 13000;
