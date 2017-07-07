/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ARG_CHECK_H
#define _ARG_CHECK_H

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


/* ========================================Gtest Arg Check ===================================================== */


    /*! \brief Template: tests arguments are valid */

    void set_get_matrix_arg_check(rocblas_status status, rocblas_int rows, rocblas_int cols, rocblas_int lda, rocblas_int ldb, rocblas_int ldc);

    void gemv_ger_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_int incx, rocblas_int incy);

    void gemm_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int K, rocblas_int lda, rocblas_int ldb, rocblas_int ldc);

    void trsm_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_int ldb);

    void symv_arg_check(rocblas_status status, rocblas_int N, rocblas_int lda, rocblas_int incx, rocblas_int incy);

    void amax_arg_check(rocblas_status status, rocblas_int* d_rocblas_result);

    template<typename T2> 
    void asum_arg_check(rocblas_status status, T2 d_rocblas_result);

    template<typename T> 
    void nrm2_dot_arg_check(rocblas_status status, T d_rocblas_result);

    void rocblas_status_success_check(rocblas_status status);

    void pointer_check(rocblas_status status, const char* message);

    void handle_check(rocblas_status status);

#endif
