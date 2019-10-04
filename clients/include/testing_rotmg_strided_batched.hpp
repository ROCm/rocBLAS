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

template <typename T, typename U = T>
void testing_rotmg_strided_batched_bad_arg(const Arguments& arg)
{
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 5;

    rocblas_local_handle handle;
    device_vector<T>     d1(safe_size);
    device_vector<T>     d2(safe_size);
    device_vector<T>     x1(safe_size);
    device_vector<T>     y1(safe_size);
    device_vector<T>     param(safe_size);

    if(!d1 || !d2 || !x1 || !y1 || !param)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(
                              nullptr, d1, 0, d2, 0, x1, 0, y1, 0, param, 0, batch_count)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(
                              handle, nullptr, 0, d2, 0, x1, 0, y1, 0, param, 0, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(
                              handle, d1, 0, nullptr, 0, x1, 0, y1, 0, param, 0, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(
                              handle, d1, 0, d2, 0, nullptr, 0, y1, 0, param, 0, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(
                              handle, d1, 0, d2, 0, x1, 0, nullptr, 0, param, 0, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(
                              handle, d1, 0, d2, 0, x1, 0, y1, 0, nullptr, 0, batch_count)),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotmg_strided_batched(const Arguments& arg)
{
    const int            TEST_COUNT   = 100;
    rocblas_int          batch_count  = arg.batch_count;
    rocblas_int          stride_d1    = arg.stride_a;
    rocblas_int          stride_d2    = arg.stride_b;
    rocblas_int          stride_x1    = arg.stride_x;
    rocblas_int          stride_y1    = arg.stride_y;
    rocblas_int          stride_param = arg.stride_c;
    rocblas_local_handle handle;

    double gpu_time_used, cpu_time_used;
    double norm_error_host = 0.0, norm_error_device = 0.0;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        size_t           safe_size = 1;
        device_vector<T> d1(safe_size);
        device_vector<T> d2(safe_size);
        device_vector<T> x1(safe_size);
        device_vector<T> y1(safe_size);
        device_vector<T> params(safe_size);

        if(!d1 || !d2 || !x1 || !y1 || !params)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS((rocblas_rotmg_strided_batched<T>(handle,
                                                                    d1,
                                                                    stride_d1,
                                                                    d2,
                                                                    stride_d2,
                                                                    x1,
                                                                    stride_x1,
                                                                    y1,
                                                                    stride_y1,
                                                                    params,
                                                                    stride_param,
                                                                    batch_count)),
                                  rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR((rocblas_rotmg_strided_batched<T>(handle,
                                                                  d1,
                                                                  stride_d1,
                                                                  d2,
                                                                  stride_d2,
                                                                  x1,
                                                                  stride_x1,
                                                                  y1,
                                                                  stride_y1,
                                                                  params,
                                                                  stride_param,
                                                                  batch_count)));

        return;
    }

    size_t size_d1    = batch_count * stride_d1;
    size_t size_d2    = batch_count * stride_d2;
    size_t size_x1    = batch_count * stride_x1;
    size_t size_y1    = batch_count * stride_y1;
    size_t size_param = batch_count * stride_param;

    // Initial Data on CPU
    host_vector<T> hd1(size_d1);
    host_vector<T> hd2(size_d2);
    host_vector<T> hx1(size_x1);
    host_vector<T> hy1(size_y1);
    host_vector<T> hparams(size_param);

    for(int i = 0; i < TEST_COUNT; i++)
    {
        rocblas_seedrand();
        rocblas_init<T>(hparams, 1, 5, 1, stride_param, batch_count);
        rocblas_init<T>(hd1, 1, 1, 1, stride_d1, batch_count);
        rocblas_init<T>(hd2, 1, 1, 1, stride_d2, batch_count);
        rocblas_init<T>(hx1, 1, 1, 1, stride_x1, batch_count);
        rocblas_init<T>(hy1, 1, 1, 1, stride_y1, batch_count);

        host_vector<T> cparams = hparams;
        host_vector<T> cd1     = hd1;
        host_vector<T> cd2     = hd2;
        host_vector<T> cx1     = hx1;
        host_vector<T> cy1     = hy1;

        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotmg<T>(cd1 + b * stride_d1,
                           cd2 + b * stride_d2,
                           cx1 + b * stride_x1,
                           cy1 + b * stride_y1,
                           cparams + b * stride_param);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> rd1     = hd1;
            host_vector<T> rd2     = hd2;
            host_vector<T> rx1     = hx1;
            host_vector<T> ry1     = hy1;
            host_vector<T> rparams = hparams;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            CHECK_ROCBLAS_ERROR((rocblas_rotmg_strided_batched<T>(handle,
                                                                  rd1,
                                                                  stride_d1,
                                                                  rd2,
                                                                  stride_d2,
                                                                  rx1,
                                                                  stride_x1,
                                                                  ry1,
                                                                  stride_y1,
                                                                  rparams,
                                                                  stride_param,
                                                                  batch_count)));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, batch_count, 1, stride_d1, rd1, cd1);
                unit_check_general<T>(1, 1, batch_count, 1, stride_d2, rd2, cd2);
                unit_check_general<T>(1, 1, batch_count, 1, stride_x1, rx1, cx1);
                unit_check_general<T>(1, 1, batch_count, 1, stride_y1, ry1, cy1);
                unit_check_general<T>(1, 5, batch_count, 1, stride_param, rparams, cparams);
            }

            if(arg.norm_check)
            {
                norm_error_host
                    = norm_check_general<T>('F', 1, 1, 1, stride_d1, batch_count, rd1, cd1);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_d2, batch_count, rd2, cd2);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_x1, batch_count, rx1, cx1);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_y1, batch_count, ry1, cy1);
                norm_error_host += norm_check_general<T>(
                    'F', 1, 5, 1, stride_param, batch_count, rparams, cparams);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_vector<T> dd1(size_d1);
            device_vector<T> dd2(size_d2);
            device_vector<T> dx1(size_x1);
            device_vector<T> dy1(size_y1);
            device_vector<T> dparams(size_param);

            CHECK_HIP_ERROR(hipMemcpy(dd1, hd1, sizeof(T) * size_d1, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dd2, hd2, sizeof(T) * size_d2, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dx1, hx1, sizeof(T) * size_x1, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy1, hy1, sizeof(T) * size_y1, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(dparams, hparams, sizeof(T) * size_param, hipMemcpyHostToDevice));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotmg_strided_batched<T>(handle,
                                                                  dd1,
                                                                  stride_d1,
                                                                  dd2,
                                                                  stride_d2,
                                                                  dx1,
                                                                  stride_x1,
                                                                  dy1,
                                                                  stride_y1,
                                                                  dparams,
                                                                  stride_param,
                                                                  batch_count)));

            host_vector<T> rd1(size_d1);
            host_vector<T> rd2(size_d2);
            host_vector<T> rx1(size_x1);
            host_vector<T> ry1(size_y1);
            host_vector<T> rparams(size_param);

            CHECK_HIP_ERROR(hipMemcpy(rd1, dd1, sizeof(T) * size_d1, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rd2, dd2, sizeof(T) * size_d2, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rx1, dx1, sizeof(T) * size_x1, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry1, dy1, sizeof(T) * size_y1, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(
                hipMemcpy(rparams, dparams, sizeof(T) * size_param, hipMemcpyDeviceToHost));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, batch_count, 1, stride_d1, rd1, cd1);
                unit_check_general<T>(1, 1, batch_count, 1, stride_d2, rd2, cd2);
                unit_check_general<T>(1, 1, batch_count, 1, stride_x1, rx1, cx1);
                unit_check_general<T>(1, 1, batch_count, 1, stride_y1, ry1, cy1);
                unit_check_general<T>(1, 5, batch_count, 1, stride_param, rparams, cparams);
            }

            if(arg.norm_check)
            {
                norm_error_device
                    = norm_check_general<T>('F', 1, 1, 1, stride_d1, batch_count, rd1, cd1);
                norm_error_device
                    += norm_check_general<T>('F', 1, 1, 1, stride_d2, batch_count, rd2, cd2);
                norm_error_device
                    += norm_check_general<T>('F', 1, 1, 1, stride_x1, batch_count, rx1, cx1);
                norm_error_device
                    += norm_check_general<T>('F', 1, 1, 1, stride_y1, batch_count, ry1, cy1);
                norm_error_host += norm_check_general<T>(
                    'F', 1, 5, 1, stride_param, batch_count, rparams, cparams);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_vector<T> dd1(size_d1);
        device_vector<T> dd2(size_d2);
        device_vector<T> dx1(size_x1);
        device_vector<T> dy1(size_y1);
        device_vector<T> dparams(size_param);

        CHECK_HIP_ERROR(hipMemcpy(dd1, hd1, sizeof(T) * size_d1, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dd2, hd2, sizeof(T) * size_d2, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dx1, hx1, sizeof(T) * size_x1, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy1, hy1, sizeof(T) * size_y1, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dparams, hparams, sizeof(T) * size_param, hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotmg_strided_batched<T>(handle,
                                             dd1,
                                             stride_d1,
                                             dd2,
                                             stride_d2,
                                             dx1,
                                             stride_x1,
                                             dy1,
                                             stride_y1,
                                             dparams,
                                             stride_param,
                                             batch_count);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotmg_strided_batched<T>(handle,
                                             dd1,
                                             stride_d1,
                                             dd2,
                                             stride_d2,
                                             dx1,
                                             stride_x1,
                                             dy1,
                                             stride_y1,
                                             dparams,
                                             stride_param,
                                             batch_count);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            std::cout << ",norm_error_host_ptr,norm_error_device";
        std::cout << std::endl;

        std::cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            std::cout << ',' << norm_error_host << ',' << norm_error_device;
        std::cout << std::endl;
    }
}
