/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include "rocblas.h"
#include "utility.h"
#include <random>

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
rocblas_rng_t rocblas_rng(69069);
rocblas_rng_t rocblas_seed(rocblas_rng);

template <>
char type2char<float>()
{
    return 's';
}

template <>
char type2char<double>()
{
    return 'd';
}

template <>
char type2char<rocblas_float_complex>()
{
    return 'c';
}

template <>
char type2char<rocblas_double_complex>()
{
    return 'z';
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    hipDeviceSynchronize();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
rocblas_int query_device_property()
{
    int device_count;
    rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
    if(status != rocblas_status_success)
    {
        printf("Query device error: cannot get device count \n");
        return -1;
    }
    else
    {
        printf("Query device success: there are %d devices \n", device_count);
    }

    for(rocblas_int i = 0;; i++)
    {
        puts("-------------------------------------------------------------------------------");

        if(i >= device_count)
        {
            break;
        }

        hipDeviceProp_t props;
        rocblas_status status = (rocblas_status)hipGetDeviceProperties(&props, i);
        if(status != rocblas_status_success)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s\n", i, props.name);
            printf("with %3.1f GB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem / 1e9,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf(
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    rocblas_status status = (rocblas_status)hipSetDevice(device_id);
    if(status != rocblas_status_success)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

char rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none: return 'N';
    case rocblas_operation_transpose: return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return '\0';
}

char rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full: return 'F';
    }
    return '\0';
}

char rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit: return 'U';
    case rocblas_diagonal_non_unit: return 'N';
    }
    return '\0';
}

char rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left: return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both: return 'B';
    }
    return '\0';
}

char rocblas_datatype2char(rocblas_datatype value)
{
    switch(value)
    {
    case rocblas_datatype_f16_r: return 'h';
    case rocblas_datatype_f32_r: return 's';
    case rocblas_datatype_f64_r: return 'd';
    case rocblas_datatype_f16_c: return 'k';
    case rocblas_datatype_f32_c: return 'c';
    case rocblas_datatype_f64_c: return 'z';
    default:
        return 'e'; // todo, handle integer types
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N': return rocblas_operation_none;
    case 'T': return rocblas_operation_transpose;
    case 'C': return rocblas_operation_conjugate_transpose;
    case 'n': return rocblas_operation_none;
    case 't': return rocblas_operation_transpose;
    case 'c': return rocblas_operation_conjugate_transpose;
    default: return static_cast<rocblas_operation>(-1);
    }
}

rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U': return rocblas_fill_upper;
    case 'L': return rocblas_fill_lower;
    case 'u': return rocblas_fill_upper;
    case 'l': return rocblas_fill_lower;
    default: return static_cast<rocblas_fill>(-1);
    }
}

rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U': return rocblas_diagonal_unit;
    case 'N': return rocblas_diagonal_non_unit;
    case 'u': return rocblas_diagonal_unit;
    case 'n': return rocblas_diagonal_non_unit;
    default: return static_cast<rocblas_diagonal>(-1);
    }
}

rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L': return rocblas_side_left;
    case 'R': return rocblas_side_right;
    case 'l': return rocblas_side_left;
    case 'r': return rocblas_side_right;
    default: return static_cast<rocblas_side>(-1);
    }
}

rocblas_datatype char2rocblas_datatype(char value)
{
    switch(value)
    {
    case 'h': return rocblas_datatype_f16_r;
    case 'H': return rocblas_datatype_f16_r;
    case 's': return rocblas_datatype_f32_r;
    case 'S': return rocblas_datatype_f32_r;
    case 'd': return rocblas_datatype_f64_r;
    case 'D': return rocblas_datatype_f64_r;
    case 'c': return rocblas_datatype_f32_c;
    case 'C': return rocblas_datatype_f32_c;
    case 'z': return rocblas_datatype_f64_c;
    case 'Z': return rocblas_datatype_f64_c;
    default: return static_cast<rocblas_datatype>(-1);
    }
}

#ifdef __cplusplus
}
#endif
