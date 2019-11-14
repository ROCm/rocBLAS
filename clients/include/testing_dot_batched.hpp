/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, bool CONJ = false>
void testing_dot_batched_bad_arg(const Arguments& arg)
{
    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int stride_y    = incy * N;
    rocblas_int batch_count = 5;

    rocblas_local_handle    handle;
    device_vector<T*, 0, T> dx(1);
    device_vector<T*, 0, T> dy(1);
    device_vector<T>        d_rocblas_result(batch_count);
    if(!dx || !dy || !d_rocblas_result)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                  N,
                                                                  nullptr,
                                                                  incx,
                                                                  dy,
                                                                  incy,
                                                                  batch_count,
                                                                  d_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                  N,
                                                                  dx,
                                                                  incx,
                                                                  nullptr,
                                                                  incy,
                                                                  batch_count,
                                                                  d_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc_batched<T>
              : rocblas_dot_batched<T>)(handle, N, dx, incx, dy, incy, batch_count, nullptr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(nullptr,
                                                                  N,
                                                                  dx,
                                                                  incx,
                                                                  dy,
                                                                  incy,
                                                                  batch_count,
                                                                  d_rocblas_result),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_dotc_batched_bad_arg(const Arguments& arg)
{
    testing_dot_batched_bad_arg<T, true>(arg);
}

template <typename T, bool CONJ = false>
void testing_dot_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    double               rocblas_error_1 = 0;
    double               rocblas_error_2 = 0;
    rocblas_local_handle handle;

    // check to prevent undefined memmory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        static const size_t     safe_size = 100; // arbitrarily set to 100
        device_vector<T*, 0, T> dx(safe_size);
        device_vector<T*, 0, T> dy(safe_size);
        device_vector<T>        d_rocblas_result(std::max(batch_count, 1));
        if(!dx || !dy || !d_rocblas_result)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        if(batch_count < 0)
        {
            EXPECT_ROCBLAS_STATUS(
                (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                          N,
                                                                          dx,
                                                                          incx,
                                                                          dy,
                                                                          incy,
                                                                          batch_count,
                                                                          d_rocblas_result),
                rocblas_status_invalid_size);
        }
        else
        {
            CHECK_ROCBLAS_ERROR(
                (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                          N,
                                                                          dx,
                                                                          incx,
                                                                          dy,
                                                                          incy,
                                                                          batch_count,
                                                                          d_rocblas_result));
        }
        return;
    }

    host_vector<T> cpu_result(batch_count);
    host_vector<T> rocblas_result_1(batch_count);
    host_vector<T> rocblas_result_2(batch_count);
    rocblas_int    abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int    abs_incy = incy >= 0 ? incy : -incy;
    size_t         size_x   = N * size_t(abs_incx);
    size_t         size_y   = N * size_t(abs_incy);

    //Device-arrays of pointers to device memory
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);
    device_vector<T>        d_rocblas_result_2(batch_count);
    if(!dx || !dy || !d_rocblas_result_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hy[batch_count];
    host_vector<T> hx[batch_count];

    // Host-arrays of pointers to device memory
    // (intermediate arrays used for the transfers)
    device_batch_vector<T> x(batch_count, size_x);
    device_batch_vector<T> y(batch_count, size_y);

    for(int b = 0; b < batch_count; ++b)
    {
        hx[b] = host_vector<T>(size_x);
        hy[b] = host_vector<T>(size_y);
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
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                      N,
                                                                      dx,
                                                                      incx,
                                                                      dy,
                                                                      incy,
                                                                      batch_count,
                                                                      rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                      N,
                                                                      dx,
                                                                      incx,
                                                                      dy,
                                                                      incy,
                                                                      batch_count,
                                                                      d_rocblas_result_2));

        CHECK_HIP_ERROR(hipMemcpy(
            rocblas_result_2, d_rocblas_result_2, sizeof(T) * batch_count, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx[b], incx, hy[b], incy, &cpu_result[b]);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * dot_gflop_count<CONJ, T>(N) / cpu_time_used * 1e6 * 1;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, 1, batch_count, 1, 1, cpu_result, rocblas_result_1);
            unit_check_general<T>(1, 1, batch_count, 1, 1, cpu_result, rocblas_result_2);
        }

        if(arg.norm_check)
        {
            std::cout << "cpu=" << cpu_result << ", gpu_host_ptr=" << rocblas_result_1
                      << ", gpu_device_ptr=" << rocblas_result_2 << "\n";

            for(int b = 0; b < batch_count; ++b)
            {
                rocblas_error_1
                    += rocblas_abs((cpu_result[b] - rocblas_result_1[b]) / cpu_result[b]);
                rocblas_error_2
                    += rocblas_abs((cpu_result[b] - rocblas_result_2[b]) / cpu_result[b]);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                      N,
                                                                      dx,
                                                                      incx,
                                                                      dy,
                                                                      incy,
                                                                      batch_count,
                                                                      rocblas_result_1);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            (CONJ ? rocblas_dotc_batched<T> : rocblas_dot_batched<T>)(handle,
                                                                      N,
                                                                      dx,
                                                                      incx,
                                                                      dy,
                                                                      incy,
                                                                      batch_count,
                                                                      rocblas_result_1);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * dot_gflop_count<CONJ, T>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = batch_count * (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        std::cout << "N,incx,incy,batch_count,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << batch_count << "," << rocblas_gflops
                  << "," << rocblas_bandwidth << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}

template <typename T>
void testing_dotc_batched(const Arguments& arg)
{
    testing_dot_batched<T, true>(arg);
}
