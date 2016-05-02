/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas.h"
#include "unit.h"

/* ========================================Gtest Unit Check ==================================================== */


    /*! \brief Template: gtest unit compare two matrices float/double/complex */
    //Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test case
    // a wrapper will cause the loop keep going



    template<>
    void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, float *hCPU, float *hGPU){
        for(rocblas_int j=0; j<N; j++){
            for(rocblas_int i=0;i<M;i++){
#ifdef GOOGLE_TEST
                ASSERT_FLOAT_EQ(hCPU[i+j*lda], hGPU[i+j*lda]);
#endif
            }
        }
    }

    template<>
    void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, double *hCPU, double *hGPU){
        for(rocblas_int j=0; j<N; j++){
            for(rocblas_int i=0;i<M;i++){
#ifdef GOOGLE_TEST
                ASSERT_DOUBLE_EQ(hCPU[i+j*lda], hGPU[i+j*lda]);
#endif
            }
        }
    }

    template<>
    void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_float_complex *hCPU, rocblas_float_complex *hGPU){
        for(rocblas_int j=0; j<N; j++){
            for(rocblas_int i=0;i<M;i++){
#ifdef GOOGLE_TEST
                ASSERT_FLOAT_EQ(hCPU[i+j*lda].x, hGPU[i+j*lda].x);
                ASSERT_FLOAT_EQ(hCPU[i+j*lda].y, hGPU[i+j*lda].y);
#endif
            }
        }
    }

    template<>
    void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_double_complex *hCPU, rocblas_double_complex *hGPU){
        for(rocblas_int j=0; j<N; j++){
            for(rocblas_int i=0;i<M;i++){
#ifdef GOOGLE_TEST
                ASSERT_DOUBLE_EQ(hCPU[i+j*lda].x, hGPU[i+j*lda].x);
                ASSERT_DOUBLE_EQ(hCPU[i+j*lda].y, hGPU[i+j*lda].y);
#endif
            }
        }
    }
