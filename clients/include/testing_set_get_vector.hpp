/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;

template <typename T>
rocblas_status testing_set_get_vector(Arguments argus)
{
    rocblas_int M         = argus.M;
    rocblas_int incx      = argus.incx;
    rocblas_int incy      = argus.incy;
    rocblas_int incb      = argus.incb;
    rocblas_int safe_size = 100;

    rocblas_status status     = rocblas_status_success;
    rocblas_status status_set = rocblas_status_success;
    rocblas_status status_get = rocblas_status_success;

    rocblas_local_handle handle;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || incb <= 0 || nullptr == handle)
    {
        host_vector<T> hx(safe_size);
        host_vector<T> hy(safe_size);

        device_vector<T> db(safe_size);
        if(!db)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status_set = rocblas_set_vector(M, sizeof(T), hx, incx, db, incb);
        status_get = rocblas_get_vector(M, sizeof(T), db, incb, hy, incy);

        if(nullptr == handle)
        {
            verify_rocblas_status_invalid_handle(status_set);
            verify_rocblas_status_invalid_handle(status_get);
        }
        else
        {
            set_get_vector_arg_check(status_set, M, incx, incy, incb);
            set_get_vector_arg_check(status_get, M, incx, incy, incb);
        }

        if(status_set != rocblas_status_success)
        {
            return status_set;
        }
        else if(status_get != rocblas_status_success)
        {
            return status_get;
        }
        else
        {
            return rocblas_status_success;
        }
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M * incx);
    host_vector<T> hy(M * incy);
    host_vector<T> hb(M * incb);
    host_vector<T> hy_gold(M * incy);

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    // allocate memory on device
    device_vector<T> db(M * incb);
    if(!db)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, M, incx);
    rocblas_init<T>(hy, 1, M, incy);
    rocblas_init<T>(hb, 1, M, incb);
    hy_gold = hy;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS
        rocblas_init<T>(hy, 1, M, incy);
        rocblas_init<T>(hb, 1, M, incb);
        CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * incb * M, hipMemcpyHostToDevice));

        status_set = rocblas_set_vector(M, sizeof(T), hx, incx, db, incb);
        status_get = rocblas_get_vector(M, sizeof(T), db, incb, hy, incy);

        cpu_time_used = get_time_us();

        // reference calculation
        for(int i = 0; i < M; i++)
        {
            hy_gold[i * incy] = hx[i * incx];
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cpu_bandwidth = (M * sizeof(T)) / cpu_time_used / 1e3;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, incy, hy, hy_gold);
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, M, incy, hy, hy_gold);
        }
    }

    if(argus.timing)
    {
        int number_timing_iterations = 1;
        gpu_time_used                = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_set_vector(M, sizeof(T), hx, incx, db, incb);
            rocblas_get_vector(M, sizeof(T), db, incb, hy, incy);
        }

        gpu_time_used     = get_time_us() - gpu_time_used;
        rocblas_bandwidth = (M * sizeof(T)) / gpu_time_used / 1e3 / number_timing_iterations;

        cout << "M,incx,incy,incb,rocblas-GB/s";

        if(argus.norm_check && cpu_bandwidth != std::numeric_limits<T>::infinity())
            cout << ",CPU-GB/s";

        cout << endl;

        cout << M << "," << incx << "," << incy << "," << incb << "," << rocblas_bandwidth;

        if(argus.norm_check && cpu_bandwidth != std::numeric_limits<T>::infinity())
            cout << "," << cpu_bandwidth;

        cout << endl;
    }

    return rocblas_status_success;
}
