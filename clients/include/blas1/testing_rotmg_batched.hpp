/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
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

    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 5;

    rocblas_local_handle   handle{arg};
    device_batch_vector<T> d1(safe_size, 1, batch_count);
    device_batch_vector<T> d2(safe_size, 1, batch_count);
    device_batch_vector<T> x1(safe_size, 1, batch_count);
    device_batch_vector<T> y1(safe_size, 1, batch_count);
    device_batch_vector<T> param(safe_size, 1, batch_count);
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

    // Initial Data on CPU
    host_batch_vector<T> hd1(1, 1, batch_count);
    host_batch_vector<T> hd2(1, 1, batch_count);
    host_batch_vector<T> hx1(1, 1, batch_count);
    host_batch_vector<T> hy1(1, 1, batch_count);
    host_batch_vector<T> hparams(5, 1, batch_count);

    for(int i = 0; i < TEST_COUNT; i++)
    {
        host_batch_vector<T> cd1(1, 1, batch_count);
        host_batch_vector<T> cd2(1, 1, batch_count);
        host_batch_vector<T> cx1(1, 1, batch_count);
        host_batch_vector<T> cy1(1, 1, batch_count);
        host_batch_vector<T> cparams(5, 1, batch_count);

        // Initialize data on host memory
        rocblas_init_vector(hd1, arg, true);
        rocblas_init_vector(hd2, arg, false);
        rocblas_init_vector(hx1, arg, false);
        rocblas_init_vector(hy1, arg, false);
        rocblas_init_vector(hparams, arg, false);

        cd1.copy_from(hd1);
        cd2.copy_from(hd2);
        cx1.copy_from(hx1);
        cy1.copy_from(hy1);
        cparams.copy_from(hparams);

        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotmg<T>(cd1[b], cd2[b], cx1[b], cy1[b], cparams[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_batch_vector<T> rd1(1, 1, batch_count);
            host_batch_vector<T> rd2(1, 1, batch_count);
            host_batch_vector<T> rx1(1, 1, batch_count);
            host_batch_vector<T> ry1(1, 1, batch_count);
            host_batch_vector<T> rparams(5, 1, batch_count);

            rd1.copy_from(hd1);
            rd2.copy_from(hd2);
            rx1.copy_from(hx1);
            ry1.copy_from(hy1);
            rparams.copy_from(hparams);

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            CHECK_ROCBLAS_ERROR(
                rocblas_rotgm_batched_fn(handle, rd1, rd2, rx1, ry1, rparams, batch_count));

            //when (input vectors are initialized with NaN's) the resultant output vector for both the cblas and rocBLAS are NAn's.  The `near_check_general` function compares the output of both the results (i.e., Nan's) and
            //throws an error. That is the reason why it is enclosed in an `if(!rocblas_isnan(arg.alpha))` loop to skip the check.

            if(!rocblas_isnan(arg.alpha))
            {
                if(arg.unit_check)
                {
                    near_check_general<T>(1, 1, 1, rd1, cd1, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rd2, cd2, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rx1, cx1, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, ry1, cy1, batch_count, rel_error);
                    near_check_general<T>(1, 5, 1, rparams, cparams, batch_count, rel_error);
                }
            }

            if(arg.norm_check)
            {
                norm_error_host = norm_check_general<T>('F', 1, 1, 1, rd1, cd1, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, rd2, cd2, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, rx1, cx1, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, ry1, cy1, batch_count);
                norm_error_host
                    += norm_check_general<T>('F', 1, 5, 1, rparams, cparams, batch_count);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_batch_vector<T> dd1(1, 1, batch_count);
            device_batch_vector<T> dd2(1, 1, batch_count);
            device_batch_vector<T> dx1(1, 1, batch_count);
            device_batch_vector<T> dy1(1, 1, batch_count);
            device_batch_vector<T> dparams(5, 1, batch_count);
            CHECK_DEVICE_ALLOCATION(dd1.memcheck());
            CHECK_DEVICE_ALLOCATION(dd2.memcheck());
            CHECK_DEVICE_ALLOCATION(dx1.memcheck());
            CHECK_DEVICE_ALLOCATION(dy1.memcheck());
            CHECK_DEVICE_ALLOCATION(dparams.memcheck());

            CHECK_HIP_ERROR(dd1.transfer_from(hd1));
            CHECK_HIP_ERROR(dd2.transfer_from(hd2));
            CHECK_HIP_ERROR(dx1.transfer_from(hx1));
            CHECK_HIP_ERROR(dy1.transfer_from(hy1));
            CHECK_HIP_ERROR(dparams.transfer_from(hparams));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(rocblas_rotgm_batched_fn(handle,
                                                         dd1.ptr_on_device(),
                                                         dd2.ptr_on_device(),
                                                         dx1.ptr_on_device(),
                                                         dy1.ptr_on_device(),
                                                         dparams.ptr_on_device(),
                                                         batch_count));

            host_batch_vector<T> rd1(1, 1, batch_count);
            host_batch_vector<T> rd2(1, 1, batch_count);
            host_batch_vector<T> rx1(1, 1, batch_count);
            host_batch_vector<T> ry1(1, 1, batch_count);
            host_batch_vector<T> rparams(5, 1, batch_count);
            CHECK_HIP_ERROR(rd1.transfer_from(dd1));
            CHECK_HIP_ERROR(rd2.transfer_from(dd2));
            CHECK_HIP_ERROR(rx1.transfer_from(dx1));
            CHECK_HIP_ERROR(ry1.transfer_from(dy1));
            CHECK_HIP_ERROR(rparams.transfer_from(dparams));

            if(!rocblas_isnan(arg.alpha))
            {
                if(arg.unit_check)
                {
                    near_check_general<T>(1, 1, 1, rd1, cd1, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rd2, cd2, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rx1, cx1, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, ry1, cy1, batch_count, rel_error);
                    near_check_general<T>(1, 5, 1, rparams, cparams, batch_count, rel_error);
                }
            }

            if(arg.norm_check)
            {
                norm_error_device = norm_check_general<T>('F', 1, 1, 1, rd1, cx1, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, rd2, cd2, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, rx1, cx1, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, ry1, cy1, batch_count);
                norm_error_device
                    += norm_check_general<T>('F', 1, 5, 1, rparams, cparams, batch_count);
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
        device_batch_vector<T> dx1(1, 1, batch_count);
        device_batch_vector<T> dy1(1, 1, batch_count);
        device_batch_vector<T> dparams(5, 1, batch_count);
        CHECK_DEVICE_ALLOCATION(dd1.memcheck());
        CHECK_DEVICE_ALLOCATION(dd2.memcheck());
        CHECK_DEVICE_ALLOCATION(dx1.memcheck());
        CHECK_DEVICE_ALLOCATION(dy1.memcheck());
        CHECK_DEVICE_ALLOCATION(dparams.memcheck());

        CHECK_HIP_ERROR(dd1.transfer_from(hd1));
        CHECK_HIP_ERROR(dd2.transfer_from(hd2));
        CHECK_HIP_ERROR(dx1.transfer_from(hx1));
        CHECK_HIP_ERROR(dy1.transfer_from(hy1));
        CHECK_HIP_ERROR(dparams.transfer_from(hparams));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotgm_batched_fn(handle,
                                     dd1.ptr_on_device(),
                                     dd2.ptr_on_device(),
                                     dx1.ptr_on_device(),
                                     dy1.ptr_on_device(),
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
                                     dx1.ptr_on_device(),
                                     dy1.ptr_on_device(),
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
