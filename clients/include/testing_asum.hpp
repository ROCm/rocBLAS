/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_asum_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_asum_fn = FORTRAN ? rocblas_asum<T, true> : rocblas_asum<T, false>;

    rocblas_int         N                = 100;
    rocblas_int         incx             = 1;
    static const size_t safe_size        = 100;
    real_t<T>           rocblas_result   = 10;
    real_t<T>*          h_rocblas_result = &rocblas_result;

    rocblas_local_handle handle;
    device_vector<T>     dx(safe_size);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_asum_fn(handle, N, nullptr, incx, h_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_asum_fn(handle, N, dx, incx, nullptr),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_asum_fn(nullptr, N, dx, incx, h_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_asum(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_asum_fn = FORTRAN ? rocblas_asum<T, true> : rocblas_asum<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;

    real_t<T>            rocblas_result_1;
    real_t<T>            rocblas_result_2;
    real_t<T>            cpu_result;
    double               rocblas_error_1;
    double               rocblas_error_2;
    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        CHECK_DEVICE_ALLOCATION(dx.memcheck());

        device_vector<real_t<T>> dr(1);
        CHECK_DEVICE_ALLOCATION(dr.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, dr));
        return;
    }

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    device_vector<real_t<T>> dr(1);
    CHECK_DEVICE_ALLOCATION(dr.memcheck());

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    CHECK_HIP_ERROR(hx.memcheck());

    // Initial Data on CPU
    rocblas_init(hx);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, &rocblas_result_1));

        // GPU BLAS rocblas_pointer_mode_device
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, dr));
        CHECK_HIP_ERROR(hipMemcpy(&rocblas_result_2, dr, sizeof(real_t<T>), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_asum<T>(N, hx, incx, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<real_t<T>, real_t<T>>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<real_t<T>, real_t<T>>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << std::scientific << cpu_result
                         << ", gpu_host_ptr=" << rocblas_result_1
                         << ", gpu_dev_ptr=" << rocblas_result_2 << std::endl;

            rocblas_error_1 = std::abs((cpu_result - rocblas_result_1) / cpu_result);
            rocblas_error_2 = std::abs((cpu_result - rocblas_result_2) / cpu_result);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum_fn(handle, N, dx, incx, dr);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum_fn(handle, N, dx, incx, dr);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                 arg,
                                                 gpu_time_used,
                                                 asum_gflop_count<T>(N),
                                                 asum_gflop_count<T>(N),
                                                 cpu_time_used,
                                                 rocblas_error_1,
                                                 rocblas_error_2);
    }
}
