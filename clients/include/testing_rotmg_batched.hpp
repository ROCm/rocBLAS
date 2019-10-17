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
void testing_rotmg_batched_bad_arg(const Arguments& arg)
{
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 5;

    rocblas_local_handle    handle;
    device_vector<T*, 0, T> d1(batch_count);
    device_vector<T*, 0, T> d2(batch_count);
    device_vector<T*, 0, T> x1(batch_count);
    device_vector<T*, 0, T> y1(batch_count);
    device_vector<T*, 0, T> param(batch_count);

    if(!d1 || !d2 || !x1 || !y1 || !param)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_batched<T>(nullptr, d1, d2, x1, y1, param, batch_count)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotmg_batched<T>(handle, nullptr, d2, x1, y1, param, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotmg_batched<T>(handle, d1, nullptr, x1, y1, param, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotmg_batched<T>(handle, d1, d2, nullptr, y1, param, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotmg_batched<T>(handle, d1, d2, x1, nullptr, param, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotmg_batched<T>(handle, d1, d2, x1, y1, nullptr, batch_count)),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotmg_batched(const Arguments& arg)
{
    const int            TEST_COUNT  = 100;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_local_handle handle;

    double gpu_time_used, cpu_time_used;
    double norm_error_host = 0.0, norm_error_device = 0.0;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        size_t                  safe_size = 1;
        device_vector<T*, 0, T> d1(safe_size);
        device_vector<T*, 0, T> d2(safe_size);
        device_vector<T*, 0, T> x1(safe_size);
        device_vector<T*, 0, T> y1(safe_size);
        device_vector<T*, 0, T> params(safe_size);

        if(!d1 || !d2 || !x1 || !y1 || !params)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS(
                (rocblas_rotmg_batched<T>(handle, d1, d2, x1, y1, params, batch_count)),
                rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR(
                (rocblas_rotmg_batched<T>(handle, d1, d2, x1, y1, params, batch_count)));

        return;
    }

    // Initial Data on CPU
    host_vector<T> hd1[batch_count];
    host_vector<T> hd2[batch_count];
    host_vector<T> hx1[batch_count];
    host_vector<T> hy1[batch_count];
    host_vector<T> hparams[batch_count];

    device_batch_vector<T> bd1(batch_count, 1);
    device_batch_vector<T> bd2(batch_count, 1);
    device_batch_vector<T> bx1(batch_count, 1);
    device_batch_vector<T> by1(batch_count, 1);
    device_batch_vector<T> bparams(batch_count, 5);

    for(int b = 0; b < batch_count; b++)
    {
        hd1[b]     = host_vector<T>(1);
        hd2[b]     = host_vector<T>(1);
        hx1[b]     = host_vector<T>(1);
        hy1[b]     = host_vector<T>(1);
        hparams[b] = host_vector<T>(5);
    }

    for(int i = 0; i < TEST_COUNT; i++)
    {
        host_vector<T> cd1[batch_count];
        host_vector<T> cd2[batch_count];
        host_vector<T> cx1[batch_count];
        host_vector<T> cy1[batch_count];
        host_vector<T> cparams[batch_count];

        rocblas_seedrand();

        for(int b = 0; b < batch_count; b++)
        {
            rocblas_init<T>(hd1[b], 1, 1, 1);
            rocblas_init<T>(hd2[b], 1, 1, 1);
            rocblas_init<T>(hx1[b], 1, 1, 1);
            rocblas_init<T>(hy1[b], 1, 1, 1);
            rocblas_init<T>(hparams[b], 1, 5, 1);
            cd1[b]     = hd1[b];
            cd2[b]     = hd2[b];
            cx1[b]     = hx1[b];
            cy1[b]     = hy1[b];
            cparams[b] = hparams[b];
        }

        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotmg<T>(cd1[b], cd2[b], cx1[b], cy1[b], cparams[b]);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> rd1[batch_count];
            host_vector<T> rd2[batch_count];
            host_vector<T> rx1[batch_count];
            host_vector<T> ry1[batch_count];
            host_vector<T> rparams[batch_count];
            T*             rd1_in[batch_count];
            T*             rd2_in[batch_count];
            T*             rx1_in[batch_count];
            T*             ry1_in[batch_count];
            T*             rparams_in[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                rd1_in[b] = rd1[b] = hd1[b];
                rd2_in[b] = rd2[b] = hd2[b];
                rx1_in[b] = rx1[b] = hx1[b];
                ry1_in[b] = ry1[b] = hy1[b];
                rparams_in[b] = rparams[b] = hparams[b];
            }

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            CHECK_ROCBLAS_ERROR((rocblas_rotmg_batched<T>(
                handle, rd1_in, rd2_in, rx1_in, ry1_in, rparams_in, batch_count)));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, batch_count, 1, rd1, cd1);
                unit_check_general<T>(1, 1, batch_count, 1, rd2, cd2);
                unit_check_general<T>(1, 1, batch_count, 1, rx1, cx1);
                unit_check_general<T>(1, 1, batch_count, 1, ry1, cy1);
                unit_check_general<T>(1, 5, batch_count, 1, rparams, cparams);
            }

            if(arg.norm_check)
            {
                norm_error_host = norm_check_general<T>('F', 1, 1, batch_count, 1, rd1, cd1);
                norm_error_host += norm_check_general<T>('F', 1, 1, batch_count, 1, rd2, cd2);
                norm_error_host += norm_check_general<T>('F', 1, 1, batch_count, 1, rx1, cx1);
                norm_error_host += norm_check_general<T>('F', 1, 1, batch_count, 1, ry1, cy1);
                norm_error_host
                    += norm_check_general<T>('F', 1, 5, batch_count, 1, rparams, cparams);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_vector<T*, 0, T> dd1(batch_count);
            device_vector<T*, 0, T> dd2(batch_count);
            device_vector<T*, 0, T> dx1(batch_count);
            device_vector<T*, 0, T> dy1(batch_count);
            device_vector<T*, 0, T> dparams(batch_count);
            device_batch_vector<T>  bd1(batch_count, 1);
            device_batch_vector<T>  bd2(batch_count, 1);
            device_batch_vector<T>  bx1(batch_count, 1);
            device_batch_vector<T>  by1(batch_count, 1);
            device_batch_vector<T>  bparams(batch_count, 5);

            for(int b = 0; b < batch_count; b++)
            {
                CHECK_HIP_ERROR(hipMemcpy(bd1[b], hd1[b], sizeof(T), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(bd2[b], hd2[b], sizeof(T), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(bx1[b], hx1[b], sizeof(T), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(by1[b], hy1[b], sizeof(T), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(
                    hipMemcpy(bparams[b], hparams[b], sizeof(T) * 5, hipMemcpyHostToDevice));
            }
            CHECK_HIP_ERROR(hipMemcpy(dd1, bd1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dd2, bd2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dx1, bx1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy1, by1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(dparams, bparams, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(
                (rocblas_rotmg_batched<T>(handle, dd1, dd2, dx1, dy1, dparams, batch_count)));

            host_vector<T> rd1[batch_count];
            host_vector<T> rd2[batch_count];
            host_vector<T> rx1[batch_count];
            host_vector<T> ry1[batch_count];
            host_vector<T> rparams[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                rd1[b]     = host_vector<T>(1);
                rd2[b]     = host_vector<T>(1);
                rx1[b]     = host_vector<T>(1);
                ry1[b]     = host_vector<T>(1);
                rparams[b] = host_vector<T>(5);
                CHECK_HIP_ERROR(hipMemcpy(rd1[b], bd1[b], sizeof(T), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(rd2[b], bd2[b], sizeof(T), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(rx1[b], bx1[b], sizeof(T), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry1[b], by1[b], sizeof(T), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(
                    hipMemcpy(rparams[b], bparams[b], sizeof(T) * 5, hipMemcpyDeviceToHost));
            }

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, batch_count, 1, rd1, cd1);
                unit_check_general<T>(1, 1, batch_count, 1, rd2, cd2);
                unit_check_general<T>(1, 1, batch_count, 1, rx1, cx1);
                unit_check_general<T>(1, 1, batch_count, 1, ry1, cy1);
                unit_check_general<T>(1, 5, batch_count, 1, rparams, cparams);
            }

            if(arg.norm_check)
            {
                norm_error_device = norm_check_general<T>('F', 1, 1, batch_count, 1, rd1, cx1);
                norm_error_device += norm_check_general<T>('F', 1, 1, batch_count, 1, rd2, cd2);
                norm_error_device += norm_check_general<T>('F', 1, 1, batch_count, 1, rx1, cx1);
                norm_error_device += norm_check_general<T>('F', 1, 1, batch_count, 1, ry1, cy1);
                norm_error_device
                    += norm_check_general<T>('F', 1, 5, batch_count, 1, rparams, cparams);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_vector<T*, 0, T> dd1(batch_count);
        device_vector<T*, 0, T> dd2(batch_count);
        device_vector<T*, 0, T> dx1(batch_count);
        device_vector<T*, 0, T> dy1(batch_count);
        device_vector<T*, 0, T> dparams(batch_count);
        device_batch_vector<T>  bd1(batch_count, 1);
        device_batch_vector<T>  bd2(batch_count, 1);
        device_batch_vector<T>  bx1(batch_count, 1);
        device_batch_vector<T>  by1(batch_count, 1);
        device_batch_vector<T>  bparams(batch_count, 5);

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(hipMemcpy(bd1[b], hd1[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bd2[b], hd2[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bx1[b], hx1[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(by1[b], hy1[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(bparams[b], hparams[b], sizeof(T) * 5, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dd1, bd1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dd2, bd2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dx1, bx1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy1, by1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dparams, bparams, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotmg_batched<T>(handle, dd1, dd2, dx1, dy1, dparams, batch_count);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotmg_batched<T>(handle, dd1, dd2, dx1, dy1, dparams, batch_count);
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
