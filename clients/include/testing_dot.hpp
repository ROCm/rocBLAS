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
void testing_dot_bad_arg(const Arguments& arg)
{
    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle;
    device_vector<T>     dx(safe_size);
    device_vector<T>     dy(safe_size);
    device_vector<T>     d_rocblas_result(1);
    if(!dx || !dy || !d_rocblas_result)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc<T>
              : rocblas_dot<T>)(handle, N, nullptr, incx, dy, incy, d_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc<T>
              : rocblas_dot<T>)(handle, N, dx, incx, nullptr, incy, d_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc<T> : rocblas_dot<T>)(handle, N, dx, incx, dy, incy, nullptr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (CONJ ? rocblas_dotc<T> : rocblas_dot<T>)(nullptr, N, dx, incx, dy, incy, d_rocblas_result),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_dotc_bad_arg(const Arguments& arg)
{
    testing_dot_bad_arg<T, true>(arg);
}

template <typename T, bool CONJ = false>
void testing_dot(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    T cpu_result;
    T rocblas_result_1;
    T rocblas_result_2;

    double               rocblas_error_1;
    double               rocblas_error_2;
    rocblas_local_handle handle;

    // check to prevent undefined memmory allocation error
    if(N <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        device_vector<T>    dy(safe_size);
        device_vector<T>    d_rocblas_result(1);
        if(!dx || !dy || !d_rocblas_result)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (CONJ ? rocblas_dotc<T>
                  : rocblas_dot<T>)(handle, N, dx, incx, dy, incy, d_rocblas_result));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> d_rocblas_result_2(1);
    if(!dx || !dy || !d_rocblas_result_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            (CONJ ? rocblas_dotc<T>
                  : rocblas_dot<T>)(handle, N, dx, incx, dy, incy, &rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (CONJ ? rocblas_dotc<T>
                  : rocblas_dot<T>)(handle, N, dx, incx, dy, incy, d_rocblas_result_2));
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result_2, d_rocblas_result_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx, incx, hy, incy, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = dot_gflop_count<CONJ, T>(N) / cpu_time_used * 1e6 * 1;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        if(arg.norm_check)
        {
            std::cout << "cpu=" << cpu_result << ", gpu_host_ptr=" << rocblas_result_1
                      << ", gpu_device_ptr=" << rocblas_result_2 << "\n";

            rocblas_error_1 = double(rocblas_abs((cpu_result - rocblas_result_1) / cpu_result));
            rocblas_error_2 = double(rocblas_abs((cpu_result - rocblas_result_2) / cpu_result));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            (CONJ ? rocblas_dotc<T>
                  : rocblas_dot<T>)(handle, N, dx, incx, dy, incy, &rocblas_result_1);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            (CONJ ? rocblas_dotc<T>
                  : rocblas_dot<T>)(handle, N, dx, incx, dy, incy, &rocblas_result_1);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = dot_gflop_count<CONJ, T>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        std::cout << "N,incx,incy,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << rocblas_gflops << ","
                  << rocblas_bandwidth << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}

template <typename T>
void testing_dotc(const Arguments& arg)
{
    testing_dot<T, true>(arg);
}
