/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_set_get_matrix(const Arguments& arg)
{
    rocblas_int rows = arg.M;
    rocblas_int cols = arg.N;
    rocblas_int lda  = arg.lda;
    rocblas_int ldb  = arg.ldb;
    rocblas_int ldc  = arg.ldc;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(rows < 0 || lda <= 0 || lda < rows || cols < 0 || ldb <= 0 || ldb < rows || ldc <= 0
       || ldc < rows)
    {
        static const size_t safe_size = 100; // arbritrarily set to 100

        host_vector<T> ha(safe_size);
        host_vector<T> hb(safe_size);
        host_vector<T> hc(safe_size);

        device_vector<T> dc(safe_size);
        CHECK_DEVICE_ALLOCATION(dc.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_set_matrix(rows, cols, sizeof(T), ha, lda, dc, ldc),
                              rocblas_status_invalid_size);
        EXPECT_ROCBLAS_STATUS(rocblas_get_matrix(rows, cols, sizeof(T), dc, ldc, hb, ldb),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> ha(cols * size_t(lda));
    host_vector<T> hb(cols * size_t(ldb));
    host_vector<T> hc(cols * size_t(ldc));
    host_vector<T> hb_gold(cols * size_t(ldb));

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    // allocate memory on device
    device_vector<T> dc(cols * size_t(ldc));
    CHECK_DEVICE_ALLOCATION(dc.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(ha, rows, cols, lda);
    rocblas_init<T>(hb, rows, cols, ldb);
    rocblas_init<T>(hc, rows, cols, ldc);
    hb_gold = hb;

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        rocblas_init<T>(hb, rows, cols, ldb);
        rocblas_init<T>(hc, rows, cols, ldc);
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(T) * ldc * cols, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_set_matrix(rows, cols, sizeof(T), ha, lda, dc, ldc));
        CHECK_ROCBLAS_ERROR(rocblas_get_matrix(rows, cols, sizeof(T), dc, ldc, hb, ldb));

        // reference calculation
        cpu_time_used = get_time_us_no_sync();
        for(int i1 = 0; i1 < rows; i1++)
            for(int i2 = 0; i2 < cols; i2++)
                hb_gold[i1 + i2 * ldb] = ha[i1 + i2 * lda];

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        cpu_bandwidth = (rows * cols * sizeof(T)) / cpu_time_used / 1e3;

        if(arg.unit_check)
        {
            unit_check_general<T>(rows, cols, ldb, hb, hb_gold);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', rows, cols, ldb, hb, hb_gold);
        }
    }

    if(arg.timing)
    {
        int number_timing_iterations = arg.iters;
        gpu_time_used                = get_time_us_sync_device(); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_set_matrix(rows, cols, sizeof(T), ha, lda, dc, ldc);
            rocblas_get_matrix(rows, cols, sizeof(T), dc, ldc, hb, ldb);
        }

        gpu_time_used = get_time_us_sync_device() - gpu_time_used;
        rocblas_bandwidth
            = (rows * cols * sizeof(T)) / gpu_time_used / 1e3 / number_timing_iterations;

        rocblas_cout << "rows,cols,lda,ldb,rocblas-GB/s";

        if(arg.norm_check)
            rocblas_cout << ",CPU-GB/s,Frobenius_norm_error";

        rocblas_cout << std::endl;

        rocblas_cout << rows << "," << cols << "," << lda << "," << ldb << "," << rocblas_bandwidth;

        if(arg.norm_check)
            rocblas_cout << "," << cpu_bandwidth << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}
