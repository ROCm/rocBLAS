/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include "ablas_types.h" 
#include <sys/time.h> 


#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */
    /*  timing:*/

    /*! \brief  CPU Timer(in millisecond): synchronize with the default device and return wall time */
    double ablas_wtime( void ){
        ablas_device_synchronize();
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * 1000) + tv.tv_usec /1000;

    };


    /*! \brief  CPU Timer(in millisecond): synchronize with given queue/stream and return wall time */
    double ablas_sync_wtime( ablas_queue queue ){
        ablas_stream_synchronize (queue);
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * 1000) + tv.tv_usec /1000;
    };
  

    /* ============================================================================================ */
    /*  Convert ablas constants to lapack char. */

    char
    ablas2lapack_transpose(ablas_transpose value)
    {
        switch (value) {
            case ablas_no_trans:      return 'N';
            case ablas_trans:         return 'T';
            case ablas_conj_trans:    return 'C';
        }
        return '\0';
    }

    char
    ablas2lapack_uplo(ablas_uplo value)
    {
        switch (value) {
            case ablas_upper:  return 'U';
            case ablas_lower:  return 'L';
        }
        return '\0';
    }

    char
    ablas2lapack_diag(ablas_diag value)
    {
        switch (value) {
            case ablas_unit:        return 'U';
            case ablas_non_unit:    return 'N';
        }
        return '\0';
    }

    char
    ablas2lapack_side(ablas_side value)
    {
        switch (value) {
            case ablas_left:   return 'L';
            case ablas_right:  return 'R';
        }
        return '\0';
    }

    /* ============================================================================================ */
    /*  Convert lapack char constants to ablas type. */

    ablas_transpose
    lapack2ablas_transpose(char value)
    {
        switch (value) {
            case 'N':      return ablas_no_trans;
            case 'T':         return ablas_trans;
            case 'C':    return ablas_conj_trans;
            case 'n':      return ablas_no_trans;
            case 't':         return ablas_trans;
            case 'c':    return ablas_conj_trans;
        }
        return ablas_no_trans;
    }

    ablas_uplo
    lapack2ablas_uplo(char value)
    {
        switch (value) {
            case 'U':  return ablas_upper;
            case 'L':  return ablas_lower;
            case 'u':  return ablas_upper;
            case 'l':  return ablas_lower;
        }
        return ablas_lower;
    }

    ablas_diag
    lapack2ablas_diag(char value)
    {
        switch (value) {
            case 'U':        return ablas_unit;
            case 'N':    return ablas_non_unit;
            case 'u':        return ablas_unit;
            case 'n':    return ablas_non_unit;
        }
        return ablas_non_unit;
    }

    ablas_side
    lapack2ablas_side(char value)
    {
        switch (value) {
            case 'L':   return ablas_left;
            case 'R':  return ablas_right;
            case 'l':   return ablas_left;
            case 'r':  return ablas_right;
        }
        return ablas_left;
    }

#ifdef __cplusplus
}
#endif


