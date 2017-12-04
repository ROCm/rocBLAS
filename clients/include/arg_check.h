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

/* ========================================Gtest Arg Check
 * ===================================================== */

/*! \brief Template: tests arguments are valid */

void set_get_matrix_arg_check(rocblas_status status,
                              rocblas_int rows,
                              rocblas_int cols,
                              rocblas_int lda,
                              rocblas_int ldb,
                              rocblas_int ldc);

void set_get_vector_arg_check(
    rocblas_status status, rocblas_int M, rocblas_int incx, rocblas_int incy, rocblas_int incd);

void gemv_ger_arg_check(rocblas_status status,
                        rocblas_int M,
                        rocblas_int N,
                        rocblas_int lda,
                        rocblas_int incx,
                        rocblas_int incy);

void gemm_arg_check(rocblas_status status,
                    rocblas_int M,
                    rocblas_int N,
                    rocblas_int K,
                    rocblas_int lda,
                    rocblas_int ldb,
                    rocblas_int ldc);

void gemm_strided_batched_arg_check(rocblas_status status,
                                    rocblas_int M,
                                    rocblas_int N,
                                    rocblas_int K,
                                    rocblas_int lda,
                                    rocblas_int ldb,
                                    rocblas_int ldc,
                                    rocblas_int batch_count);

void geam_arg_check(rocblas_status status,
                    rocblas_int M,
                    rocblas_int N,
                    rocblas_int lda,
                    rocblas_int ldb,
                    rocblas_int ldc);

void trsm_arg_check(
    rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_int ldb);

void symv_arg_check(
    rocblas_status status, rocblas_int N, rocblas_int lda, rocblas_int incx, rocblas_int incy);

void iamax_arg_check(rocblas_status status, rocblas_int* d_rocblas_result);

template <typename T2>
void asum_arg_check(rocblas_status status, T2 d_rocblas_result);

template <typename T>
void nrm2_dot_arg_check(rocblas_status status, T d_rocblas_result);

void verify_rocblas_status_invalid_pointer(rocblas_status status, const char* message);

void verify_rocblas_status_invalid_size(rocblas_status status, const char* message);

void verify_rocblas_status_invalid_handle(rocblas_status status);

void verify_rocblas_status_success(rocblas_status status, const char* message);

template <typename T>
void verify_not_nan(T arg);

template <typename T>
void verify_equal(T arg1, T arg2, const char* message);

#endif
