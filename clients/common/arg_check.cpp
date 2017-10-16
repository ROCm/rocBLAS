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
    rocblas_int lda, rocblas_int ldb, rocblas_int ldc)
{
    if (status != rocblas_status_invalid_size)
    {
        std::cerr << "ERROR in arguments rows, cols, lda, ldb, ldc: ";
        std::cerr << rows << ',' << cols << ',' << lda << ',' << ldb << ',' << ldc << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
    #endif
}

void gemv_ger_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int lda, 
    rocblas_int incx, rocblas_int incy)
{
    if (M < 0 || N < 0 || lda < M || lda < 1 || 0 == incx || 0 == incy)
    {
        if (status != rocblas_status_invalid_size)
        {
            std::cerr << "ERROR: (M < 0 || N < 0 || lda < M || lda < 1 || 0 == incx || 0 == incy) " << std::endl;
            std::cerr << "ERROR: and (status != rocblas_status_invalid_size)" << std::endl;
	        std::cerr << "ERROR: status = " << status << std::endl;
        }
    }
    else if (0 == M || 0 == N)
    {
        if (status != rocblas_status_success)
        {
            std::cerr << "ERROR: (0 == M || 0 == N)" << std::endl;
            std::cerr << "ERROR: and (status != rocblas_status_success)" << std::endl;
	        std::cerr << "ERROR: status = " << status << std::endl;
        }
    }
    #ifdef GOOGLE_TEST
    if (M < 0 || N < 0 || lda < M || lda < 1 || 0 == incx || 0 == incy)
    {
        ASSERT_EQ(status, rocblas_status_invalid_size);
    }
    else if (0 == M || 0 == N)
    {
        ASSERT_EQ(status, rocblas_status_success);
    }
    else
    {
        EXPECT_TRUE(false) << "error in gemv_ger_arg_check";
    }
    #endif
}

void gemm_arg_check(rocblas_status status, rocblas_int M, rocblas_int N, rocblas_int K, 
    rocblas_int lda, rocblas_int ldb, rocblas_int ldc)
{
    std::cerr << "ERROR in arguments M, N, K, lda, ldb, ldc: ";
    std::cerr << M << ',' << N << ',' << K << ',' << lda << ',' << ldb << ',' << ldc << std::endl;
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
    #endif
}

void geam_arg_check(rocblas_status status, rocblas_int M, rocblas_int N,
    rocblas_int lda, rocblas_int ldb, rocblas_int ldc)
{
    std::cerr << "ERROR in arguments M, N, lda, ldb, ldc: ";
    std::cerr << M << ',' << N << ',' << lda << ',' << ldb << ',' << ldc << std::endl;
    #ifdef GOOGLE_TEST
    if (M == 0 || N == 0)
    {
        ASSERT_EQ(status, rocblas_status_success);
    }
    else
    {
        ASSERT_EQ(status, rocblas_status_invalid_size);
    }
    #endif
}

void trsm_arg_check(rocblas_status status, rocblas_int M, rocblas_int N,
    rocblas_int lda, rocblas_int ldb)
{
    std::cerr << "ERROR in arguments M, N, lda, ldb: ";
    std::cerr << M << ',' << N << ',' << lda << ',' << ldb << std::endl;
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
    #endif
}

void symv_arg_check(rocblas_status status, rocblas_int N, rocblas_int lda, rocblas_int incx, rocblas_int incy)
{
    std::cerr << "ERROR in arguments N, lda, incx, incy: ";
    std::cerr << N << ',' << lda << ',' << incx << ',' << incy << std::endl;
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
    #endif
}

void amax_arg_check(rocblas_status status, rocblas_int* d_rocblas_result)
{
    rocblas_int h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device )
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    }
    else
    {
        h_rocblas_result = *d_rocblas_result;
    }
    if ( h_rocblas_result != 0 )
    {
        std::cerr << "result should be 0, result =  " << h_rocblas_result << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0);
    ASSERT_EQ(status, rocblas_status_success);
    #endif
}

template<>
void asum_arg_check(rocblas_status status, float* d_rocblas_result)
{
    float h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device )
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(float), hipMemcpyDeviceToHost));
    }
    else
    {
        h_rocblas_result = *d_rocblas_result;
    }
    if ( h_rocblas_result != 0.0 )
    {
        std::cerr << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
    ASSERT_EQ(status, rocblas_status_success);
    #endif
}

template<>
void asum_arg_check(rocblas_status status, double* d_rocblas_result)
{
    double h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device )
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(double), hipMemcpyDeviceToHost));
    }
    else
    {
        h_rocblas_result = *d_rocblas_result;
    }
    if ( h_rocblas_result != 0.0 )
    {
        std::cerr << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
    #endif
}

template<>
void nrm2_dot_arg_check(rocblas_status status, double* d_rocblas_result)
{
    double h_rocblas_result;

    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device )
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(double), hipMemcpyDeviceToHost));
    }
    else
    {
        h_rocblas_result = *d_rocblas_result;
    }
    if ( h_rocblas_result != 0.0 )
    {
        std::cerr << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
    ASSERT_EQ(status, rocblas_status_success);
    #endif
}

template<>
void nrm2_dot_arg_check(rocblas_status status, float* d_rocblas_result)
{
    float h_rocblas_result;
    if( rocblas_pointer_to_mode(d_rocblas_result) == rocblas_pointer_mode_device )
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(&h_rocblas_result, d_rocblas_result, sizeof(float), hipMemcpyDeviceToHost));
    }
    else
    {
        h_rocblas_result = *d_rocblas_result;
    }
    if ( h_rocblas_result != 0.0 )
    {
        std::cerr << "result should be 0.0, result =  " << h_rocblas_result << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(h_rocblas_result, 0.0);
    #endif
}

void verify_rocblas_status_invalid_pointer(rocblas_status status, const char* message)
{
    if (status != rocblas_status_invalid_pointer)
    {
        std::cerr << message << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_pointer);
    #endif
}

void verify_rocblas_status_invalid_size(rocblas_status status, const char* message)
{
    if (status != rocblas_status_invalid_size)
    {
        std::cerr << "***** ERROR: status != rocblas_status_invalid_size, ";
        std::cerr << message << " *****" << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_size);
    #endif
}

//void handle_check(rocblas_status status)
void verify_rocblas_status_invalid_handle(rocblas_status status)
{
    if (status != rocblas_status_invalid_handle)
    {
        std::cerr << "ERROR: handle is null pointer" << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_invalid_handle);
    #endif
}

void verify_rocblas_status_success(rocblas_status status, const char* message)
{
    if(status != rocblas_status_success)
    {
        std::cerr << message << std::endl;
        std::cerr << "ERROR: status should be rocblas_status_success" << std::endl;
        std::cerr << "ERROR: status = " << status << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocblas_status_success);
    #endif
}

template<>
void verify_not_nan(float arg)
{
    if(arg != arg)
    {
        std::cerr << "ERROR: argument is NaN" << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(arg, arg);
    #endif
}

template<>
void verify_not_nan(double arg)
{
    if(arg != arg)
    {
        std::cerr << "ERROR: argument is NaN" << std::endl;
    }
    #ifdef GOOGLE_TEST
    ASSERT_EQ(arg, arg);
    #endif
}
