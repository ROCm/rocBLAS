/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "bytes.hpp"
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
    const bool FORTRAN        = arg.fortran;
    auto       rocblas_dot_fn = FORTRAN ? (CONJ ? rocblas_dotc<T, true> : rocblas_dot<T, true>)
                                  : (CONJ ? rocblas_dotc<T, false> : rocblas_dot<T, false>);

    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle;
    device_vector<T>     dx(safe_size);
    device_vector<T>     dy(safe_size);
    device_vector<T>     d_rocblas_result(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS((rocblas_dot_fn)(handle, N, nullptr, incx, dy, incy, d_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_dot_fn)(handle, N, dx, incx, nullptr, incy, d_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_dot_fn)(handle, N, dx, incx, dy, incy, nullptr),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_dot_fn)(nullptr, N, dx, incx, dy, incy, d_rocblas_result),
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
    const bool FORTRAN        = arg.fortran;
    auto       rocblas_dot_fn = FORTRAN ? (CONJ ? rocblas_dotc<T, true> : rocblas_dot<T, true>)
                                  : (CONJ ? rocblas_dotc<T, false> : rocblas_dot<T, false>);

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
        device_vector<T> d_rocblas_result(1);
        CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_dot_fn)(handle, N, nullptr, incx, nullptr, incy, d_rocblas_result));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> d_rocblas_result_2(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

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

    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
    auto dy_ptr = (arg.algo) ? (T*)(dx) : (T*)(dy);
    auto hy_ptr = (arg.algo) ? &hx[0] : &hy[0];
    if(arg.algo)
        incy = incx;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_dot_fn)(handle, N, dx, incx, dy_ptr, incy, &rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_dot_fn)(handle, N, dx, incx, dy_ptr, incy, d_rocblas_result_2));
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result_2, d_rocblas_result_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx, incx, hy_ptr, incy, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << cpu_result << ", gpu_host_ptr=" << rocblas_result_1
                         << ", gpu_device_ptr=" << rocblas_result_2 << std::endl;

            rocblas_error_1 = double(rocblas_abs((cpu_result - rocblas_result_1) / cpu_result));
            rocblas_error_2 = double(rocblas_abs((cpu_result - rocblas_result_2) / cpu_result));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            (rocblas_dot_fn)(handle, N, dx, incx, dy_ptr, incy, d_rocblas_result_2);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            (rocblas_dot_fn)(handle, N, dx, incx, dy_ptr, incy, d_rocblas_result_2);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_algo>{}.log_args<T>(rocblas_cout,
                                                                 arg,
                                                                 gpu_time_used,
                                                                 dot_gflop_count<CONJ, T>(N),
                                                                 dot_gbyte_count<T>(N),
                                                                 cpu_time_used,
                                                                 rocblas_error_1,
                                                                 rocblas_error_2);
    }
}

template <typename T>
void testing_dotc(const Arguments& arg)
{
    testing_dot<T, true>(arg);
}
