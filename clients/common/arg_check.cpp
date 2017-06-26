/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <iostream>
#include "rocblas.h"
#include "arg_check.h"

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK) {\
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != hipSuccess) { \
        fprintf(stderr, "hip error code: %d at %s:%d\n",  TMP_STATUS_FOR_CHECK,__FILE__, __LINE__); \
} }


/* ========================================Gtest Arg Check ===================================================== */


    /*! \brief Template: checks if arguments are valid
    */


void set_get_matrix_arg_check(rocblas_status status, rocblas_int rows, rocblas_int cols, 
    rocblas_int lda, rocblas_int ldb, rocblas_int ldc){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
#endif
#ifndef GOOGLE_TEST
    std::cout << "check arguments rows, cols, lda, ldb, ldc: ";
    std::cout << rows << ',' << cols << ',' << lda << ',' << ldb << ',' << ldc << std::endl;
#endif
}


void gemv_ger_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int lda, 
    rocblas_int incx, rocblas_int incy){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
#endif
#ifndef GOOGLE_TEST
    std::cout << "check arguments M, N, lda, incx, incy: ";
    std::cout << M << ',' << N << ',' << lda << ',' << incx << ',' << incy << std::endl;
#endif
}

void gemm_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int K, 
    rocblas_int lda, rocblas_int ldb, rocblas_int ldc){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
#endif
#ifndef GOOGLE_TEST
    std::cout << "check arguments M, N, K, lda, ldb, ldc: ";
    std::cout << M << ',' << N << ',' << K << ',' << lda << ',' << ldb << ',' << ldc << std::endl;
#endif
}

void trsm_arg_check(rocblas_status status, rocblas_int M, rocblas_int N,
    rocblas_int lda, rocblas_int ldb){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
#endif
#ifndef GOOGLE_TEST
    std::cout << "check arguments M, N, lda, ldb: ";
    std::cout << M << ',' << N << ',' << lda << ',' << ldb << std::endl;
#endif
}

void symv_arg_check(rocblas_status status, rocblas_int N, rocblas_int lda, rocblas_int incx, rocblas_int incy){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
#endif
#ifndef GOOGLE_TEST
    std::cout << "check arguments N, lda, incx, incy: ";
    std::cout << N << ',' << lda << ',' << incx << ',' << incy << std::endl;
#endif
}


void amax_arg_check(rocblas_status status, rocblas_int* d_rocblas_result){
    rocblas_int h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device ){
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    }
    else{
        h_rocblas_result = *d_rocblas_result;
    }
#ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0);
    ASSERT_EQ(status, rocblas_status_success);
#endif
#ifndef GOOGLE_TEST
    if ( h_rocblas_result != 0 ){
        std::cout << "result should be 0, result =  " << h_rocblas_result << std::endl;
    }
#endif
}

template<>
void asum_arg_check(rocblas_status status, float* d_rocblas_result){
    float h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device ){
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(float), hipMemcpyDeviceToHost));
    }
    else{
        h_rocblas_result = *d_rocblas_result;
    }
#ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
    ASSERT_EQ(status, rocblas_status_success);
#endif
#ifndef GOOGLE_TEST
    if ( h_rocblas_result != 0.0 ){
        std::cout << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
#endif
}

template<>
void asum_arg_check(rocblas_status status, double* d_rocblas_result){
    double h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device ){
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(double), hipMemcpyDeviceToHost));
    }
    else{
        h_rocblas_result = *d_rocblas_result;
    }
#ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
#endif
#ifndef GOOGLE_TEST
    if ( h_rocblas_result != 0.0 ){
        std::cout << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
#endif
}

template<>
void nrm2_dot_arg_check(rocblas_status status, double* d_rocblas_result){
    double h_rocblas_result;

    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device ){
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(double), hipMemcpyDeviceToHost));
    }
    else{
        h_rocblas_result = *d_rocblas_result;
    }
#ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
    ASSERT_EQ(status, rocblas_status_success);
#endif
#ifndef GOOGLE_TEST
    if ( h_rocblas_result != 0.0 ){
        std::cout << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
#endif
}

template<>
void nrm2_dot_arg_check(rocblas_status status, float* d_rocblas_result){
    float h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device ){
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(float), hipMemcpyDeviceToHost));
    }
    else{
        h_rocblas_result = *d_rocblas_result;
    }
#ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
#endif
#ifndef GOOGLE_TEST
    if ( h_rocblas_result != 0.0 ){
        std::cout << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
#endif
}

void rocblas_status_success_check(rocblas_status status){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_success);
#endif
}

void pointer_check(rocblas_status status, const char* message){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_pointer);
#endif
#ifndef GOOGLE_TEST
    std::cout << message << std::endl;
#endif
}

void handle_check(rocblas_status status){
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_handle);
#endif
#ifndef GOOGLE_TEST
    std::cout << "handle is null pointer";
#endif
}
