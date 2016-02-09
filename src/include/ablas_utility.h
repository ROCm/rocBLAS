/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#pragma once
#ifndef _ABLAS_UTILITY_H_
#define _ABLAS_UTILITY_H_

#include "ablas_types.h" 
#include "ablas_runtime.h" 
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
    /*  query device :*/

    void ablas_get_device_property()
    {

        int num_device, device_id=0;

        ablas_get_device_count(&num_device);

        ablas_set_device(device_id);

        printf("There are %d GPU devices; running on device ID %d \n", num_device, device_id);

    }

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
    /* integer functions */

    /*! \brief  For integers x >= 0, y > 0, returns ceil( x/y ).
     *          For x == 0, this is 0.
     */
    __host__ __device__
    static inline ablas_int ablas_ceildiv( ablas_int x, ablas_int y )
    {
        return (x + y - 1)/y;
    }

    /*! \brief  For integers x >= 0, y > 0, returns x rounded up to multiple of y.
     *          For x == 0, this is 0. y is not necessarily a power of 2.         
     */
    __host__ __device__
    static inline ablas_int ablas_roundup( ablas_int x, ablas_int y )
    {
        return ablas_ceildiv( x, y ) * y;
    }

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

#endif

