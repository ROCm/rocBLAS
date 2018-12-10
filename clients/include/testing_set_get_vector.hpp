/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "flops.hpp"

template <typename T>
void testing_set_get_vector(const Arguments& arg)
{
    rocblas_int M    = arg.M;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;
    rocblas_int incb = arg.incb;
    rocblas_local_handle handle;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || incb <= 0)
    {
        static const size_t safe_size = 100;

        host_vector<T> hx(safe_size);
        host_vector<T> hy(safe_size);
        device_vector<T> db(safe_size);
        if(!db)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }
        EXPECT_ROCBLAS_STATUS(rocblas_set_vector(M, sizeof(T), hx, incx, db, incb),
                              rocblas_status_invalid_size);
        EXPECT_ROCBLAS_STATUS(rocblas_get_vector(M, sizeof(T), db, incb, hy, incy),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M * static_cast<size_t>(incx));
    host_vector<T> hy(M * static_cast<size_t>(incy));
    host_vector<T> hb(M * static_cast<size_t>(incb));
    host_vector<T> hy_gold(M * static_cast<size_t>(incy));

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    // allocate memory on device
    device_vector<T> db(M * static_cast<size_t>(incb));
    if(!db)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, M, incx);
    rocblas_init<T>(hy, 1, M, incy);
    rocblas_init<T>(hb, 1, M, incb);
    hy_gold = hy;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        rocblas_init<T>(hy, 1, M, incy);
        rocblas_init<T>(hb, 1, M, incb);
        CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * incb * M, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_vector(M, sizeof(T), hx, incx, db, incb));
        CHECK_ROCBLAS_ERROR(rocblas_get_vector(M, sizeof(T), db, incb, hy, incy));

        cpu_time_used = get_time_us();

        // reference calculation
        for(int i = 0; i < M; i++)
        {
            hy_gold[i * incy] = hx[i * incx];
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cpu_bandwidth = (M * sizeof(T)) / cpu_time_used / 1e3;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, incy, hy, hy_gold);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, M, incy, hy, hy_gold);
        }
    }

    if(arg.timing)
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

        std::cout << "M,incx,incy,incb,rocblas-GB/s";

        if(arg.norm_check && cpu_bandwidth != std::numeric_limits<T>::infinity())
            std::cout << ",CPU-GB/s";

        std::cout << std::endl;

        std::cout << M << "," << incx << "," << incy << "," << incb << "," << rocblas_bandwidth;

        if(arg.norm_check && cpu_bandwidth != std::numeric_limits<T>::infinity())
            std::cout << "," << cpu_bandwidth;

        std::cout << std::endl;
    }
}
