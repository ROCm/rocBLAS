/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
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

template <typename T>
void testing_nrm2_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_nrm2_strided_batched_fn = arg.fortran ? rocblas_nrm2_strided_batched<T, true>
                                                       : rocblas_nrm2_strided_batched<T, false>;

    rocblas_int         N           = 100;
    rocblas_int         incx        = 1;
    rocblas_stride      stridex     = 1;
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 100;

    rocblas_local_handle handle{arg};

    device_vector<T>         dx(safe_size);
    device_vector<real_t<T>> d_rocblas_result(batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_strided_batched_fn(
                              handle, N, nullptr, incx, stridex, batch_count, d_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_nrm2_strided_batched_fn(handle, N, dx, incx, stridex, batch_count, nullptr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_strided_batched_fn(
                              nullptr, N, dx, incx, stridex, batch_count, d_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_nrm2_strided_batched(const Arguments& arg)
{
    auto rocblas_nrm2_strided_batched_fn = arg.fortran ? rocblas_nrm2_strided_batched<T, true>
                                                       : rocblas_nrm2_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_stride stridex     = arg.stride_x;
    rocblas_int    batch_count = arg.batch_count;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        device_vector<real_t<T>> d_rocblas_result_0(std::max(1, batch_count));
        host_vector<real_t<T>>   h_rocblas_result_0(std::max(1, batch_count));
        CHECK_HIP_ERROR(d_rocblas_result_0.memcheck());
        CHECK_HIP_ERROR(h_rocblas_result_0.memcheck());

        rocblas_init_nan(h_rocblas_result_0, 1, std::max(1, batch_count), 1);
        CHECK_HIP_ERROR(hipMemcpy(d_rocblas_result_0,
                                  h_rocblas_result_0,
                                  sizeof(real_t<T>) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_fn(
            handle, N, nullptr, incx, stridex, batch_count, d_rocblas_result_0));

        if(batch_count > 0)
        {
            host_vector<real_t<T>> cpu_0(batch_count);
            host_vector<real_t<T>> gpu_0(batch_count);
            CHECK_HIP_ERROR(cpu_0.memcheck());
            CHECK_HIP_ERROR(gpu_0.memcheck());

            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_rocblas_result_0, sizeof(real_t<T>) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<real_t<T>>(1, batch_count, 1, cpu_0, gpu_0);
        }

        return;
    }

    real_t<T> rocblas_result_1[batch_count];
    real_t<T> rocblas_result_2[batch_count];
    real_t<T> cpu_result[batch_count];

    size_t size_x = (size_t)stridex;

    // allocate memory on device
    device_vector<T>         dx(batch_count * size_x);
    device_vector<real_t<T>> d_rocblas_result_2(batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(batch_count * size_x);

    // Initialize data on host memory
    rocblas_init_vector(
        hx, arg, N, incx, stridex, batch_count, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x * batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_fn(
            handle, N, dx, incx, stridex, batch_count, rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_fn(
            handle, N, dx, incx, stridex, batch_count, d_rocblas_result_2));
        CHECK_HIP_ERROR(hipMemcpy(rocblas_result_2,
                                  d_rocblas_result_2,
                                  batch_count * sizeof(real_t<T>),
                                  hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int i = 0; i < batch_count; i++)
            cblas_nrm2<T>(N, hx + i * stridex, incx, cpu_result + i);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        real_t<T> abs_result = cpu_result[0] > 0 ? cpu_result[0] : -cpu_result[0];
        real_t<T> abs_error;
        if(abs_result > 0)
        {
            abs_error = std::numeric_limits<real_t<T>>::epsilon() * N * abs_result;
        }
        else
        {
            abs_error = std::numeric_limits<real_t<T>>::epsilon() * N;
        }
        real_t<T> tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
            //  If test fails, try decreasing n or increasing tolerance.
        abs_error *= tolerance;

        if(!rocblas_isnan(arg.alpha))
        {
            if(arg.unit_check)
            {
                near_check_general<real_t<T>, real_t<T>>(
                    batch_count, 1, 1, cpu_result, rocblas_result_1, abs_error);
                near_check_general<real_t<T>, real_t<T>>(
                    batch_count, 1, 1, cpu_result, rocblas_result_2, abs_error);
            }
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << cpu_result[0] << ", gpu_host_ptr=" << rocblas_result_1[0]
                         << ", gpu_dev_ptr=" << rocblas_result_2[0] << "\n";
            rocblas_error_1 = std::abs((cpu_result[0] - rocblas_result_1[0]) / cpu_result[0]);
            rocblas_error_2 = std::abs((cpu_result[0] - rocblas_result_2[0]) / cpu_result[0]);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_nrm2_strided_batched_fn(
                handle, N, dx, incx, stridex, batch_count, d_rocblas_result_2);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_nrm2_strided_batched_fn(
                handle, N, dx, incx, stridex, batch_count, d_rocblas_result_2);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_batch_count>{}.log_args<T>(rocblas_cout,
                                                                            arg,
                                                                            gpu_time_used,
                                                                            nrm2_gflop_count<T>(N),
                                                                            nrm2_gbyte_count<T>(N),
                                                                            cpu_time_used,
                                                                            rocblas_error_1,
                                                                            rocblas_error_2);
    }
}
