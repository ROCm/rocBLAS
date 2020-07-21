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
void testing_set_get_vector(const Arguments& arg)
{
    rocblas_int          M    = arg.M;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_int          incb = arg.incb;
    rocblas_local_handle handle(arg.atomics_mode);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || incb <= 0)
    {
        static const size_t safe_size = 100;

        host_vector<T>   hx(safe_size);
        host_vector<T>   hy(safe_size);
        device_vector<T> db(safe_size);
        CHECK_DEVICE_ALLOCATION(db.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_set_vector(M, sizeof(T), hx, incx, db, incb),
                              rocblas_status_invalid_size);
        EXPECT_ROCBLAS_STATUS(rocblas_get_vector(M, sizeof(T), db, incb, hy, incy),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M * size_t(incx));
    host_vector<T> hy(M * size_t(incy));
    host_vector<T> hb(M * size_t(incb));
    host_vector<T> hy_gold(M * size_t(incy));

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    // allocate memory on device
    device_vector<T> db(M * size_t(incb));
    CHECK_DEVICE_ALLOCATION(db.memcheck());

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

        cpu_time_used = get_time_us_no_sync();

        // reference calculation
        for(int i = 0; i < M; i++)
        {
            hy_gold[i * incy] = hx[i * incx];
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;
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
        int         number_timing_iterations = arg.iters;
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_set_vector(M, sizeof(T), hx, incx, db, incb);
            rocblas_get_vector(M, sizeof(T), db, incb, hy, incy);
        }

        gpu_time_used     = get_time_us_sync(stream) - gpu_time_used;
        rocblas_bandwidth = (M * sizeof(T)) / gpu_time_used / 1e3 / number_timing_iterations;

        rocblas_cout << "M,incx,incy,incb,rocblas-GB/s";

        if(arg.norm_check && cpu_bandwidth != std::numeric_limits<T>::infinity())
            rocblas_cout << ",CPU-GB/s";

        rocblas_cout << std::endl;

        rocblas_cout << M << "," << incx << "," << incy << "," << incb << "," << rocblas_bandwidth;

        if(arg.norm_check && cpu_bandwidth != std::numeric_limits<T>::infinity())
            rocblas_cout << "," << cpu_bandwidth;

        rocblas_cout << std::endl;
    }
}
