/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include <vector>
#include "rocblas.h"
#include <sys/time.h>

using namespace std;

/*!\file
 * \brief provide data initialization, timing, rocblas type <-> lapack char conversion utilities.
 */

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
      exit(EXIT_FAILURE);\
    }

    /* ============================================================================================ */
    /* generate random number :*/

    /*! \brief  generate a random number between [0, 0.999...] . */
    template<typename T>
    T random_generator(){
        return rand()/( (T)RAND_MAX + 1);
    };


    /* ============================================================================================ */
    /*! \brief  matrix/vector initialization: */
    // for vector x (M=1, N=lengthX, lda=incx);
    // for complex number, the real/imag part would be initialized with the same value
    template<typename T>
    void rocblas_init(vector<T> &A, rocblas_int M, rocblas_int N, rocblas_int lda){
        for (rocblas_int i = 0; i < M; ++i){
            for (rocblas_int j = 0; j < N; ++j){
                A[i+j*lda] = random_generator<T>();
            }
        }
    };

    /*! \brief  symmetric matrix initialization: */
    // for real matrix only
    template<typename T>
    void rocblas_init_symmetric(vector<T> &A, rocblas_int N, rocblas_int lda){
        for (rocblas_int i = 0; i < N; ++i){
            for (rocblas_int j = 0; j <= i; ++j){
                A[j+i*lda] = A[i+j*lda] = random_generator<T>();
            }
        }
    };

    /*! \brief  hermitian matrix initialization: */
    // for complex matrix only, the real/imag part would be initialized with the same value
    // except the diagonal elment must be real
    template<typename T>
    void rocblas_init_hermitian(vector<T> &A, rocblas_int N, rocblas_int lda){
        for (rocblas_int i = 0; i < N; ++i){
            for (rocblas_int j = 0; j <= i; ++j){
                A[j+i*lda] = A[i+j*lda] = random_generator<T>();
                if(i==j) A[j+i*lda].y = 0.0;
            }
        }
    };

    /* ============================================================================================ */
    /*! \brief  turn float -> 's', double -> 'd', rocblas_float_complex -> 'c', rocblas_double_complex -> 'z' */
    template<typename T>
    char type2char();

#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */
    /*  device query and print out their ID and name */
    rocblas_int query_device_property();

    /*  set current device to device_id */
    void set_device(rocblas_int device_id);

    /* ============================================================================================ */
    /*  timing: HIP only provides very limited timers function clock() and not general;
                rocblas sync CPU and device and use more accurate CPU timer*/

    /*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
    double get_time_us( void );


    /*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
    double get_time_us_sync( hipStream_t stream );

    /* ============================================================================================ */
    /*  Convert rocblas constants to lapack char. */

    char
    rocblas2char_operation(rocblas_operation value);

    char
    rocblas2char_fill(rocblas_fill value);

    char
    rocblas2char_diagonal(rocblas_diagonal value);

    char
    rocblas2char_side(rocblas_side value);

    /* ============================================================================================ */
    /*  Convert lapack char constants to rocblas type. */

    rocblas_operation
    char2rocblas_operation(char value);

    rocblas_fill
    char2rocblas_fill(char value);

    rocblas_diagonal
    char2rocblas_diagonal(char value);

    rocblas_side
    char2rocblas_side(char value);

#ifdef __cplusplus
}
#endif


/* ============================================================================================ */


/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this rocblas library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments {
    public:
    rocblas_int M = 128;
    rocblas_int N = 128;
    rocblas_int K = 128;

    rocblas_int lda = 128;
    rocblas_int ldb = 128;
    rocblas_int ldc = 128;

    rocblas_int incx = 1 ;
    rocblas_int incy = 1 ;

    rocblas_int start = 1024;
    rocblas_int end   = 10240;
    rocblas_int step  = 1000;

    double alpha = 1.0;
    double beta  = 0.0;

    char transA_option = 'N';
    char transB_option = 'N';
    char side_option = 'L';
    char uplo_option = 'L';
    char diag_option = 'N';

    rocblas_int apiCallCount = 1;
    rocblas_int order_option = 0;// 0 is column  major, 1 is row major
    rocblas_int batch_count = 1000;

    rocblas_int norm_check = 0;
    rocblas_int unit_check = 1;
    rocblas_int timing = 0;

    Arguments & operator=(const Arguments &rhs)
    {
        M = rhs.M;
        N = rhs.N;
        K = rhs.K;

        lda = rhs.lda;
        ldb = rhs.ldb;
        ldc = rhs.ldc;

        incx = rhs.incx;
        incy = rhs.incy;

        start = rhs.start;
        end = rhs.end;
        step = rhs.step;

        alpha = rhs.alpha;
        beta = rhs.beta;

        transA_option = rhs.transA_option;
        transB_option = rhs.transB_option;
        side_option = rhs.side_option;
        uplo_option = rhs.uplo_option;
        diag_option = rhs.diag_option;

        apiCallCount = rhs.apiCallCount;
        order_option = rhs.order_option;
        batch_count = rhs.batch_count;

        norm_check = rhs.norm_check;
        unit_check = rhs.unit_check;
        timing = rhs.timing;

        return *this;
    }

};



#endif
