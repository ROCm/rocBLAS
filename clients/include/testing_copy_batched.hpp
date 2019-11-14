/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
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
void testing_copy_batched_bad_arg(const Arguments& arg)
{
    rocblas_int       N           = 100;
    rocblas_int       incx        = 1;
    rocblas_int       incy        = 1;
    const rocblas_int batch_count = 5;

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);
    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_copy_batched<T>(handle, N, nullptr, incx, dy, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_copy_batched<T>(handle, N, dx, incx, nullptr, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_copy_batched<T>(nullptr, N, dx, incx, dy, incy, batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_copy_batched(const Arguments& arg)
{
    rocblas_int          N    = arg.N;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_local_handle handle;
    rocblas_int          batch_count = arg.batch_count;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<T*, 0, T> dx(1);
        device_vector<T*, 0, T> dy(1);
        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }
        EXPECT_ROCBLAS_STATUS(rocblas_copy_batched<T>(handle, N, dx, incx, dy, incy, batch_count),
                              N > 0 && batch_count < 0 ? rocblas_status_invalid_size
                                                       : rocblas_status_success);
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    //Device-arrays of pointers to device memory
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);

    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hy[batch_count];
    host_vector<T> hy_gold[batch_count];
    host_vector<T> hx[batch_count];

    // Host-arrays of pointers to device memory
    // (intermediate arrays used for the transfers)
    device_batch_vector<T> x(batch_count, size_x);
    device_batch_vector<T> y(batch_count, size_y);

    for(int b = 0; b < batch_count; ++b)
    {
        hx[b]      = host_vector<T>(size_x);
        hy[b]      = host_vector<T>(size_y);
        hy_gold[b] = host_vector<T>(size_y);
    }

    int last = batch_count - 1;
    if(batch_count && ((!y[last] && size_y) || (!x[last] && size_x)))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();

    for(int b = 0; b < batch_count; ++b)
    {
        rocblas_init<T>(hx[b], 1, N, abs_incx);
        rocblas_init<T>(hy[b], 1, N, abs_incy);

        // copy_batched vector is easy in STL; hy_gold = hx: save a copy_batched in hy_gold which will be output of CPU
        // BLAS
        hy_gold[b] = hy[b];
    }

    // copy data from CPU to device
    // 1. Use intermediate arrays to access device memory from host
    for(int b = 0; b < batch_count; ++b)
    {
        CHECK_HIP_ERROR(hipMemcpy(x[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(y[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
    }

    // 2. Copy intermediate arrays into device arrays
    CHECK_HIP_ERROR(hipMemcpy(dx, x, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, y, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_copy_batched<T>(handle, N, dx, incx, dy, incy, batch_count));

        for(int b = 0; b < batch_count; ++b)
        {
            hipMemcpy(hy[b], y[b], sizeof(T) * size_y, hipMemcpyDeviceToHost);
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_copy<T>(N, hx[b], incx, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, hy_gold, hy);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, batch_count, hy_gold, hy);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_copy_batched<T>(handle, N, dx, incx, dy, incy, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_copy_batched<T>(handle, N, dx, incx, dy, incy, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,batch_count,rocblas-us";

        if(arg.norm_check)
            std::cout << ",CPU-us,error";

        std::cout << std::endl;

        std::cout << N << "," << incx << "," << incy << "," << batch_count << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error;

        std::cout << std::endl;
    }
}
