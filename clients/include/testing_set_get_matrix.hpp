/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;

template <typename T>
rocblas_status testing_set_get_matrix(Arguments argus)
{
    rocblas_int rows      = argus.rows;
    rocblas_int cols      = argus.cols;
    rocblas_int lda       = argus.lda;
    rocblas_int ldb       = argus.ldb;
    rocblas_int ldc       = argus.ldc;
    rocblas_int safe_size = 100; // arbritrarily set to 100

    rocblas_status status_set = rocblas_status_success;
    rocblas_status status_get = rocblas_status_success;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(rows < 0 || lda <= 0 || lda < rows || cols < 0 || ldb <= 0 || ldb < rows || ldc <= 0 ||
       ldc < rows || nullptr == handle)
    {
        vector<T> ha(safe_size);
        vector<T> hb(safe_size);
        vector<T> hc(safe_size);

        auto dc_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dc = (T*)dc_managed.get();
        if(!dc)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status_set =
            rocblas_set_matrix(rows, cols, sizeof(T), (void*)ha.data(), lda, (void*)dc, ldc);
        status_get =
            rocblas_get_matrix(rows, cols, sizeof(T), (void*)dc, ldc, (void*)hb.data(), ldb);

        if(nullptr == handle)
        {
            verify_rocblas_status_invalid_handle(status_set);
            verify_rocblas_status_invalid_handle(status_get);
        }
        else
        {
            set_get_matrix_arg_check(status_set, rows, cols, lda, ldb, ldc);
            set_get_matrix_arg_check(status_get, rows, cols, lda, ldb, ldc);
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
    vector<T> ha(cols * lda);
    vector<T> hb(cols * ldb);
    vector<T> hc(cols * ldc);
    vector<T> hb_gold(cols * ldb);

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    // allocate memory on device
    auto dc_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * cols * ldc),
                                         rocblas_test::device_free};
    T* dc = (T*)dc_managed.get();
    if(!dc)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(ha, rows, cols, lda);
    rocblas_init<T>(hb, rows, cols, ldb);
    rocblas_init<T>(hc, rows, cols, ldc);
    hb_gold = hb;

    if(argus.unit_check || argus.norm_check)
    {
        // ROCBLAS
        rocblas_init<T>(hb, rows, cols, ldb);
        rocblas_init<T>(hc, rows, cols, ldc);
        CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * ldc * cols, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(
            rocblas_set_matrix(rows, cols, sizeof(T), (void*)ha.data(), lda, (void*)dc, ldc));
        CHECK_ROCBLAS_ERROR(
            rocblas_get_matrix(rows, cols, sizeof(T), (void*)dc, ldc, (void*)hb.data(), ldb));

        // reference calculation
        cpu_time_used = get_time_us();
        for(int i1 = 0; i1 < rows; i1++)
        {
            for(int i2 = 0; i2 < cols; i2++)
            {
                hb_gold[i1 + i2 * ldb] = ha[i1 + i2 * lda];
            }
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cpu_bandwidth = (rows * cols * sizeof(T)) / cpu_time_used / 1e3;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(rows, cols, ldb, hb.data(), hb_gold.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', rows, cols, ldb, hb.data(), hb_gold.data());
        }
    }

    if(argus.timing)
    {
        int number_timing_iterations = 1;
        gpu_time_used                = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_set_matrix(rows, cols, sizeof(T), (void*)ha.data(), lda, (void*)dc, ldc);
            rocblas_get_matrix(rows, cols, sizeof(T), (void*)dc, ldc, (void*)hb.data(), ldb);
        }

        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_bandwidth =
            (rows * cols * sizeof(T)) / gpu_time_used / 1e3 / number_timing_iterations;

        cout << "rows,cols,lda,ldb,rocblas-GB/s";

        if(argus.norm_check)
            cout << ",CPU-GB/s";

        cout << endl;

        cout << rows << "," << cols << "," << lda << "," << ldb << "," << rocblas_bandwidth;

        if(argus.norm_check)
            cout << "," << cpu_bandwidth;

        cout << endl;
    }

    return rocblas_status_success;
}
