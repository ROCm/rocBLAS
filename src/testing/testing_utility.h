/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "ablas_types.h" 
#include <sys/time.h> 

/*!\file
 * \brief provide random generator, device query, timing, etc, utilities.
 */

    /* ============================================================================================ */
    /* generate random number :*/

     /*! \brief  generate a random number between [0, 0.999...] . */
    template<typename T>
    T random_generator(){
        return rand()/( (T)RAND_MAX + 1)
    }


#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */
    /*  timing:*/

    /*! \brief  CPU Timer(in millisecond): synchronize with the default device and return wall time */
    double ablas_wtime( void );


    /*! \brief  CPU Timer(in millisecond): synchronize with given queue/stream and return wall time */
    double ablas_sync_wtime( ablas_queue queue );
  
    /* ============================================================================================ */
    /*  Convert ablas constants to lapack char. */

    char
    ablas2lapack_transpose(ablas_transpose value);

    char
    ablas2lapack_uplo(ablas_uplo value);

    char
    ablas2lapack_diag(ablas_diag value);

    char
    ablas2lapack_side(ablas_side value);

    /* ============================================================================================ */
    /*  Convert lapack char constants to ablas type. */

    ablas_transpose
    lapack2ablas_transpose(char value);

    ablas_uplo
    lapack2ablas_uplo(char value);

    ablas_diag
    lapack2ablas_diag(char value);

    ablas_side
    lapack2ablas_side(char value);

#ifdef __cplusplus
}
#endif


/* ============================================================================================ */

/*! \brief Struct used to parse command line arguments in testing. */

struct arguments {
    ablas_int M;
    ablas_int N;
    ablas_int K;

    ablas_int start;
    ablas_int end;
    ablas_int step;

    double alpha;
    double beta;

    char transA_option;
    char transB_option;
    char side_option;
    char uplo_option;
    char diag_option;

    ablas_int apiCallCount;
    ablas_int order_option;
    ablas_int validate;
} ;


#endif

