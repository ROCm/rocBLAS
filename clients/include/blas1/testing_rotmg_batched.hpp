/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_rotgm_batched_fn
        = FORTRAN ? rocblas_rotmg_batched<T, true> : rocblas_rotmg_batched<T, false>;

    rocblas_int batch_count = 5;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_batch_vector<T> d1(1, 1, batch_count);
    device_batch_vector<T> d2(1, 1, batch_count);
    device_batch_vector<T> x1(1, 1, batch_count);
    device_batch_vector<T> y1(1, 1, batch_count);
    device_batch_vector<T> param(5, 1, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(d1.memcheck());
    CHECK_DEVICE_ALLOCATION(d2.memcheck());
    CHECK_DEVICE_ALLOCATION(x1.memcheck());
    CHECK_DEVICE_ALLOCATION(y1.memcheck());
    CHECK_DEVICE_ALLOCATION(param.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(nullptr,
                                                   d1.ptr_on_device(),
                                                   d2.ptr_on_device(),
                                                   x1.ptr_on_device(),
                                                   y1.ptr_on_device(),
                                                   param.ptr_on_device(),
                                                   batch_count),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(handle,
                                                   nullptr,
                                                   d2.ptr_on_device(),
                                                   x1.ptr_on_device(),
                                                   y1.ptr_on_device(),
                                                   param.ptr_on_device(),
                                                   batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(handle,
                                                   d1.ptr_on_device(),
                                                   nullptr,
                                                   x1.ptr_on_device(),
                                                   y1.ptr_on_device(),
                                                   param.ptr_on_device(),
                                                   batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(handle,
                                                   d1.ptr_on_device(),
                                                   d2.ptr_on_device(),
                                                   nullptr,
                                                   y1.ptr_on_device(),
                                                   param.ptr_on_device(),
                                                   batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(handle,
                                                   d1.ptr_on_device(),
                                                   d2.ptr_on_device(),
                                                   x1.ptr_on_device(),
                                                   nullptr,
                                                   param.ptr_on_device(),
                                                   batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(handle,
                                                   d1.ptr_on_device(),
                                                   d2.ptr_on_device(),
                                                   x1.ptr_on_device(),
                                                   y1.ptr_on_device(),
                                                   nullptr,
                                                   batch_count),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotmg_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_rotgm_batched_fn
        = FORTRAN ? rocblas_rotmg_batched<T, true> : rocblas_rotmg_batched<T, false>;

    const int            TEST_COUNT  = 100;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_local_handle handle{arg};

    double gpu_time_used, cpu_time_used;
    double norm_error_host = 0.0, norm_error_device = 0.0;
    T      rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS(rocblas_rotgm_batched_fn(
                                  handle, nullptr, nullptr, nullptr, nullptr, nullptr, batch_count),
                              rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hd1), `d` is in GPU (device) memory (eg dd1).
    // Allocate host memory
    host_batch_vector<T> hd1(1, 1, batch_count);
    host_batch_vector<T> hd2(1, 1, batch_count);
    host_batch_vector<T> hx(1, 1, batch_count);
    host_batch_vector<T> hy(1, 1, batch_count);
    host_batch_vector<T> hparams(5, 1, batch_count);

    for(int i = 0; i < TEST_COUNT; i++)
    {
        host_batch_vector<T> hd1_gold(1, 1, batch_count);
        host_batch_vector<T> hd2_gold(1, 1, batch_count);
        host_batch_vector<T> hx_gold(1, 1, batch_count);
        host_batch_vector<T> hy_gold(1, 1, batch_count);
        host_batch_vector<T> hparams_gold(5, 1, batch_count);

        // Initialize data on host memory
        rocblas_init_vector(hd1, arg, rocblas_client_alpha_sets_nan, true);
        rocblas_init_vector(hd2, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(hparams, arg, rocblas_client_alpha_sets_nan, false);

        hd1_gold.copy_from(hd1);
        hd2_gold.copy_from(hd2);
        hx_gold.copy_from(hx);
        hy_gold.copy_from(hy);
        hparams_gold.copy_from(hparams);

        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotmg<T>(hd1_gold[b], hd2_gold[b], hx_gold[b], hy_gold[b], hparams_gold[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_batch_vector<T> rd1(1, 1, batch_count);
            host_batch_vector<T> rd2(1, 1, batch_count);
            host_batch_vector<T> rx(1, 1, batch_count);
            host_batch_vector<T> ry(1, 1, batch_count);
            host_batch_vector<T> rparams(5, 1, batch_count);

            rd1.copy_from(hd1);
            rd2.copy_from(hd2);
            rx.copy_from(hx);
            ry.copy_from(hy);
            rparams.copy_from(hparams);

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            CHECK_ROCBLAS_ERROR(
                rocblas_rotgm_batched_fn(handle, rd1, rd2, rx, ry, rparams, batch_count));

            if(arg.unit_check)
            {
                near_check_general<T>(1, 1, 1, rd1, hd1_gold, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, rd2, hd2_gold, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, rx, hx_gold, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, ry, hy_gold, batch_count, rel_error);
                near_check_general<T>(1, 5, 1, rparams, hparams_gold, batch_count, rel_error);
            }

            if(arg.norm_check)
            {
                norm_error_host = norm_check_general<T>('F', 1, 1, 1, rd1, hd1_gold, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, rd2, hd2_gold, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, rx, hx_gold, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, ry, hy_gold, batch_count);
                norm_error_host
                    += norm_check_general<T>('F', 1, 5, 1, rparams, hparams_gold, batch_count);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            // Allocate device memory
            device_batch_vector<T> dd1(1, 1, batch_count);
            device_batch_vector<T> dd2(1, 1, batch_count);
            device_batch_vector<T> dx(1, 1, batch_count);
            device_batch_vector<T> dy(1, 1, batch_count);
            device_batch_vector<T> dparams(5, 1, batch_count);

            // Check device memory allocation
            CHECK_DEVICE_ALLOCATION(dd1.memcheck());
            CHECK_DEVICE_ALLOCATION(dd2.memcheck());
            CHECK_DEVICE_ALLOCATION(dx.memcheck());
            CHECK_DEVICE_ALLOCATION(dy.memcheck());
            CHECK_DEVICE_ALLOCATION(dparams.memcheck());

            CHECK_HIP_ERROR(dd1.transfer_from(hd1));
            CHECK_HIP_ERROR(dd2.transfer_from(hd2));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(dparams.transfer_from(hparams));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(rocblas_rotgm_batched_fn(handle,
                                                         dd1.ptr_on_device(),
                                                         dd2.ptr_on_device(),
                                                         dx.ptr_on_device(),
                                                         dy.ptr_on_device(),
                                                         dparams.ptr_on_device(),
                                                         batch_count));

            host_batch_vector<T> rd1(1, 1, batch_count);
            host_batch_vector<T> rd2(1, 1, batch_count);
            host_batch_vector<T> rx(1, 1, batch_count);
            host_batch_vector<T> ry(1, 1, batch_count);
            host_batch_vector<T> rparams(5, 1, batch_count);

            CHECK_HIP_ERROR(rd1.transfer_from(dd1));
            CHECK_HIP_ERROR(rd2.transfer_from(dd2));
            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));
            CHECK_HIP_ERROR(rparams.transfer_from(dparams));

            if(arg.unit_check)
            {
                near_check_general<T>(1, 1, 1, rd1, hd1_gold, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, rd2, hd2_gold, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, rx, hx_gold, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, ry, hy_gold, batch_count, rel_error);
                near_check_general<T>(1, 5, 1, rparams, hparams_gold, batch_count, rel_error);
            }

            if(arg.norm_check)
            {
                norm_error_device = norm_check_general<T>('F', 1, 1, 1, rd1, hx_gold, batch_count);
                norm_error_device
                    += norm_check_general<T>('F', 1, 1, 1, rd2, hd2_gold, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, rx, hx_gold, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, ry, hy_gold, batch_count);
                norm_error_device
                    += norm_check_general<T>('F', 1, 5, 1, rparams, hparams_gold, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_batch_vector<T> dd1(1, 1, batch_count);
        device_batch_vector<T> dd2(1, 1, batch_count);
        device_batch_vector<T> dx(1, 1, batch_count);
        device_batch_vector<T> dy(1, 1, batch_count);
        device_batch_vector<T> dparams(5, 1, batch_count);

        CHECK_DEVICE_ALLOCATION(dd1.memcheck());
        CHECK_DEVICE_ALLOCATION(dd2.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());
        CHECK_DEVICE_ALLOCATION(dparams.memcheck());

        CHECK_HIP_ERROR(dd1.transfer_from(hd1));
        CHECK_HIP_ERROR(dd2.transfer_from(hd2));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dparams.transfer_from(hparams));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotgm_batched_fn(handle,
                                     dd1.ptr_on_device(),
                                     dd2.ptr_on_device(),
                                     dx.ptr_on_device(),
                                     dy.ptr_on_device(),
                                     dparams.ptr_on_device(),
                                     batch_count);
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotgm_batched_fn(handle,
                                     dd1.ptr_on_device(),
                                     dd2.ptr_on_device(),
                                     dx.ptr_on_device(),
                                     dy.ptr_on_device(),
                                     dparams.ptr_on_device(),
                                     batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_batch_count>{}.log_args<T>(rocblas_cout,
                                                   arg,
                                                   gpu_time_used,
                                                   ArgumentLogging::NA_value,
                                                   ArgumentLogging::NA_value,
                                                   cpu_time_used,
                                                   norm_error_host,
                                                   norm_error_device);
    }
}
