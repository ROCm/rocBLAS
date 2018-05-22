/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _UNIT_H
#define _UNIT_H

#include "rocblas.h"

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/* =====================================================================

    Google Unit check: ASSERT_EQ( elementof(A), elementof(B))

   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Unit check.
 */

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going
template <typename T>
void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU);

template <typename T>
void trsm_err_res_check(T max_error, rocblas_int M, T forward_tolerance, T eps);

template <typename T>
void potf2_err_res_check(T max_error, rocblas_int N, T forward_tolerance, T eps);

#endif
