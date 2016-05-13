/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */


#include <sys/time.h>
#include <hip_runtime_api.h>
#include "rocblas.h"
#include "utility.h"

    template<>
    char type2char<float>(){
        return 's';
    }

    template<>
    char type2char<double>(){
        return 'd';
    }

    template<>
    char type2char<rocblas_float_complex>(){
        return 'c';
    }

    template<>
    char type2char<rocblas_double_complex>(){
        return 'z';
    }

#ifdef __cplusplus
extern "C" {
#endif


    /* ============================================================================================ */
    /*  timing:*/

    /*! \brief  CPU Timer(in millisecond): synchronize with the default device and return wall time */
    double get_time_ms( void ){
        hipDeviceSynchronize();
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * 1000) + tv.tv_usec /1000;

    };


    /*! \brief  CPU Timer(in millisecond): synchronize with given queue/stream and return wall time */
    double get_time_ms_sync( hipStream_t stream ){
        hipStreamSynchronize (stream);
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * 1000) + tv.tv_usec /1000;
    };

    /* ============================================================================================ */
    /*  device query and print out their ID and name; return number of compute-capable devices. */
    rocblas_int query_device_property(){
        int device_count;
        rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
        if(status != rocblas_success){
               printf ("Query device error: cannot get device count \n");
               return -1;
        }
        else{
            printf("Query device success: there are %d devices \n", device_count);
        }

        for(rocblas_int i=0;i<device_count; i++)
        {
            hipDeviceProp_t props;
            rocblas_status status = (rocblas_status)hipGetDeviceProperties(&props, i);
            if(status != rocblas_success){
               printf ("Query device error: cannot get device ID %d's property\n", i);
            }
            else{
                printf("Device ID %d : %s ------------------------------------------------------\n", i, props.name);
                printf ("with %3.1f GB memory, clock rate %dMHz @ computing capability %d.%d \n",
                         props.totalGlobalMem/1e9, (int)(props.clockRate/1000), props.major, props.minor);
                printf ("regsPerBlock %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                         props.regsPerBlock, props.sharedMemPerBlock/1e3, props.maxThreadsPerBlock, props.warpSize);

                printf("-------------------------------------------------------------------------\n");
            }
        }

        return device_count;
    }

    /*  set current device to device_id */
    void set_device(rocblas_int device_id){
        rocblas_status status = (rocblas_status)hipSetDevice(device_id);
        if(status != rocblas_success){
               printf ("Set device error: cannot set device ID %d, there may not be such device ID\n", (int)device_id);
        }
    }

    /* ============================================================================================ */
    /*  Convert rocblas constants to lapack char. */

    char
    rocblas2char_transpose(rocblas_transpose value)
    {
        switch (value) {
            case rocblas_no_trans:      return 'N';
            case rocblas_trans:         return 'T';
            case rocblas_conj_trans:    return 'C';
        }
        return '\0';
    }

    char
    rocblas2char_uplo(rocblas_uplo value)
    {
        switch (value) {
            case rocblas_upper:  return 'U';
            case rocblas_lower:  return 'L';
            case rocblas_full :  return 'F';
        }
        return '\0';
    }

    char
    rocblas2char_diag(rocblas_diag value)
    {
        switch (value) {
            case rocblas_unit:        return 'U';
            case rocblas_non_unit:    return 'N';
        }
        return '\0';
    }

    char
    rocblas2char_side(rocblas_side value)
    {
        switch (value) {
            case rocblas_left:   return 'L';
            case rocblas_right:  return 'R';
            case rocblas_both:   return 'B';
        }
        return '\0';
    }

    /* ============================================================================================ */
    /*  Convert lapack char constants to rocblas type. */

    rocblas_transpose
    char2rocblas_transpose(char value)
    {
        switch (value) {
            case 'N':      return rocblas_no_trans;
            case 'T':         return rocblas_trans;
            case 'C':    return rocblas_conj_trans;
            case 'n':      return rocblas_no_trans;
            case 't':         return rocblas_trans;
            case 'c':    return rocblas_conj_trans;
        }
        return rocblas_no_trans;
    }

    rocblas_uplo
    char2rocblas_uplo(char value)
    {
        switch (value) {
            case 'U':  return rocblas_upper;
            case 'L':  return rocblas_lower;
            case 'u':  return rocblas_upper;
            case 'l':  return rocblas_lower;
        }
        return rocblas_lower;
    }

    rocblas_diag
    char2rocblas_diag(char value)
    {
        switch (value) {
            case 'U':        return rocblas_unit;
            case 'N':    return rocblas_non_unit;
            case 'u':        return rocblas_unit;
            case 'n':    return rocblas_non_unit;
        }
        return rocblas_non_unit;
    }

    rocblas_side
    char2rocblas_side(char value)
    {
        switch (value) {
            case 'L':   return rocblas_left;
            case 'R':  return rocblas_right;
            case 'l':   return rocblas_left;
            case 'r':  return rocblas_right;
        }
        return rocblas_left;
    }

#ifdef __cplusplus
}
#endif
