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
void testing_asum_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_asum_batched_fn
        = FORTRAN ? rocblas_asum_batched<T, true> : rocblas_asum_batched<T, false>;

    rocblas_int         N                = 100;
    rocblas_int         incx             = 1;
    rocblas_int         batch_count      = 5;
    static const size_t safe_size        = 100;
    real_t<T>           rocblas_result   = 10;
    real_t<T>*          h_rocblas_result = &rocblas_result;

    rocblas_local_handle handle;

    device_batch_vector<T> dx(N, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    EXPECT_ROCBLAS_STATUS(
        rocblas_asum_batched_fn(handle, N, nullptr, incx, batch_count, h_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_asum_batched_fn(handle, N, dx.ptr_on_device(), incx, batch_count, nullptr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_asum_batched_fn(
                              nullptr, N, dx.ptr_on_device(), incx, batch_count, h_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_asum_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_asum_batched_fn
        = FORTRAN ? rocblas_asum_batched<T, true> : rocblas_asum_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int batch_count = arg.batch_count;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        host_vector<real_t<T>> res(std::max(1, std::abs(batch_count)));
        CHECK_HIP_ERROR(res.memcheck());
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_asum_batched_fn(handle, N, nullptr, incx, batch_count, res),
                              rocblas_status_success);

        return;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice

    // allocate memory
    device_batch_vector<T> dx(N, incx, batch_count);
    host_batch_vector<T>   hx(N, incx, batch_count);

    device_vector<real_t<T>> dr(batch_count);
    host_vector<real_t<T>>   hr1(batch_count);
    host_vector<real_t<T>>   hr(batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(hx.memcheck());

    CHECK_DEVICE_ALLOCATION(dr.memcheck());
    CHECK_HIP_ERROR(hr1.memcheck());
    CHECK_HIP_ERROR(hr.memcheck());

    //
    // Initialize memory on host.
    //
    rocblas_init(hx);

    //
    // Transfer from host to device.
    //
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            rocblas_asum_batched_fn(handle, N, dx.ptr_on_device(), incx, batch_count, hr1));

        // GPU BLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_asum_batched_fn(handle, N, dx.ptr_on_device(), incx, batch_count, dr));

        //
        // Transfer from device to host.
        //
        CHECK_HIP_ERROR(hr.transfer_from(dr));

        real_t<T> cpu_result[batch_count];
        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_asum<T>(N, hx[i], incx, cpu_result + i);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<real_t<T>, real_t<T>>(1, batch_count, 1, cpu_result, hr1);
            unit_check_general<real_t<T>, real_t<T>>(1, batch_count, 1, cpu_result, hr);
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << std::scientific << cpu_result[0]
                         << ", gpu_host_ptr=" << hr1[0] << ", gpu_dev_ptr=" << hr[0] << std::endl;

            rocblas_error_1 = std::abs((cpu_result[0] - hr1[0]) / cpu_result[0]);
            rocblas_error_2 = std::abs((cpu_result[0] - hr[0]) / cpu_result[0]);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum_batched_fn(handle, N, dx.ptr_on_device(), incx, batch_count, dr);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum_batched_fn(handle, N, dx.ptr_on_device(), incx, batch_count, dr);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_batch_count>{}.log_args<T>(rocblas_cout,
                                                                arg,
                                                                gpu_time_used,
                                                                asum_gflop_count<T>(N),
                                                                asum_gbyte_count<T>(N),
                                                                cpu_time_used,
                                                                rocblas_error_1,
                                                                rocblas_error_2);
    }
}
