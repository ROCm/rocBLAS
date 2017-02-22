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


/* ========================================Gtest Unit Check ==================================================== */


    /*! \brief Template: gtest unit compare two matrices float/double/complex */
    //Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test case
    // a wrapper will cause the loop keep going
    template<typename T>
    void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, T *hCPU, T *hGPU);


    template<typename T>
    void unit_check_trsm(rocblas_int M, rocblas_int N, rocblas_int lda, double hGPU, T tolerance);

    template<typename T>
    T get_trsm_tolerance();

#endif
