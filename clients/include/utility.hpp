/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "rocblas.h"
#include "rocblas_test.hpp"
#include "utility.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/*!\file
 * \brief provide common utilities
 */

// We use std::cout and std::cerr instead of stdout, stderr, std::cout and std::cerr
// This must come after the header #includes above, to avoid poisoning system headers.
// This is only enabled for rocblas-test and rocblas-bench.
#if defined(GOOGLE_TEST) || defined(ROCBLAS_BENCH)
#undef stdout
#undef stderr
// All stdio and std::ostream functions related to stdout and stderr are poisoned, as are functions
// which can create buffer overflows
#pragma GCC poison cout cerr clog stdout stderr gets puts putchar fprintf printf sprintf vfprintf \
    vprintf vsprintf
#endif

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class rocblas_local_handle
{
    rocblas_handle handle;

public:
    rocblas_local_handle()
    {
        rocblas_create_handle(&handle);
    }
    ~rocblas_local_handle()
    {
        rocblas_destroy_handle(handle);
    }

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&()
    {
        return handle;
    }
    operator const rocblas_handle&() const
    {
        return handle;
    }
};

/* ============================================================================================ */
/*  device query and print out their ID and name */
rocblas_int query_device_property();

/*  set current device to device_id */
void set_device(rocblas_int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocblas sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/* ============================================================================================ */
// Return path of this executable
std::string rocblas_exepath();

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void rocblas_print_matrix(
    std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
        {
            std::cout << "matrix  col " << i << ", row " << j
                      << ", CPU result=" << CPU_result[j + i * lda]
                      << ", GPU result=" << GPU_result[j + i * lda] << "\n";
        }
}

#endif
