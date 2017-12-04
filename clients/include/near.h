/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _NEAR_H
#define _NEAR_H

#include "rocblas.h"

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/* =====================================================================

    Google Near check: ASSERT_EQ( elementof(A), elementof(B))

   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Near check.
 */

/* ========================================Gtest Near Check
 * ==================================================== */

/*! \brief Template: gtest near compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going
template <typename T1, typename T2>
void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, T1* hCPU, T1* hGPU, T2 abs_error);

#endif
