/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "testing_common.hpp"

template <typename T, typename U = T>
void testing_rotmg_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_rotmg_strided_batched_fn    = arg.api == FORTRAN
                                                   ? rocblas_rotmg_strided_batched<T, true>
                                                   : rocblas_rotmg_strided_batched<T, false>;
    auto rocblas_rotmg_strided_batched_fn_64 = arg.api == FORTRAN_64
                                                   ? rocblas_rotmg_strided_batched_64<T, true>
                                                   : rocblas_rotmg_strided_batched_64<T, false>;

    int64_t        batch_count  = 5;
    rocblas_stride stride_d1    = 1;
    rocblas_stride stride_d2    = 1;
    rocblas_stride stride_x     = 1;
    rocblas_stride stride_y     = 1;
    rocblas_stride stride_param = 5;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_vector<T> d1(1, 1, stride_d1, batch_count);
    device_strided_batch_vector<T> d2(1, 1, stride_d2, batch_count);
    device_strided_batch_vector<T> dx(1, 1, stride_x, batch_count);
    device_strided_batch_vector<T> dy(1, 1, stride_y, batch_count);
    device_strided_batch_vector<T> dparams(5, 1, stride_param, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(d1.memcheck());
    CHECK_DEVICE_ALLOCATION(d2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparams.memcheck());

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_rotmg_strided_batched_fn,
                (nullptr, d1, 0, d2, 0, dx, 0, dy, 0, dparams, 0, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_strided_batched_fn,
                (handle, nullptr, 0, d2, 0, dx, 0, dy, 0, dparams, 0, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_strided_batched_fn,
                (handle, d1, 0, nullptr, 0, dx, 0, dy, 0, dparams, 0, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_strided_batched_fn,
                (handle, d1, 0, d2, 0, nullptr, 0, dy, 0, dparams, 0, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_strided_batched_fn,
                (handle, d1, 0, d2, 0, dx, 0, nullptr, 0, dparams, 0, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_strided_batched_fn,
                (handle, d1, 0, d2, 0, dx, 0, dy, 0, nullptr, 0, batch_count));
}

template <typename T>
void testing_rotmg_strided_batched(const Arguments& arg)
{
    auto rocblas_rotmg_strided_batched_fn    = arg.api == FORTRAN
                                                   ? rocblas_rotmg_strided_batched<T, true>
                                                   : rocblas_rotmg_strided_batched<T, false>;
    auto rocblas_rotmg_strided_batched_fn_64 = arg.api == FORTRAN_64
                                                   ? rocblas_rotmg_strided_batched_64<T, true>
                                                   : rocblas_rotmg_strided_batched_64<T, false>;

    int64_t        batch_count  = arg.batch_count;
    rocblas_stride stride_d1    = arg.stride_a;
    rocblas_stride stride_d2    = arg.stride_b;
    rocblas_stride stride_x     = arg.stride_x;
    rocblas_stride stride_y     = arg.stride_y;
    rocblas_stride stride_param = arg.stride_c;

    rocblas_local_handle handle{arg};

    double  gpu_time_used, cpu_time_used;
    double  norm_error_host = 0.0, norm_error_device = 0.0;
    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_rotmg_strided_batched_fn,
                   (handle,
                    nullptr,
                    stride_d1,
                    nullptr,
                    stride_d2,
                    nullptr,
                    stride_x,
                    nullptr,
                    stride_y,
                    nullptr,
                    stride_param,
                    batch_count));
        return;
    }

    // Initial Data on CPU
    host_strided_batch_vector<T> hd1(1, 1, stride_d1, batch_count);
    host_strided_batch_vector<T> hd2(1, 1, stride_d2, batch_count);
    host_strided_batch_vector<T> hx(1, 1, stride_x, batch_count);
    host_strided_batch_vector<T> hy(1, 1, stride_y, batch_count);
    host_strided_batch_vector<T> hparams(5, 1, stride_param, batch_count);

    // Allocate device memory
    device_strided_batch_vector<T> dd1(1, 1, stride_d1, batch_count);
    device_strided_batch_vector<T> dd2(1, 1, stride_d2, batch_count);
    device_strided_batch_vector<T> dx(1, 1, stride_x, batch_count);
    device_strided_batch_vector<T> dy(1, 1, stride_y, batch_count);
    device_strided_batch_vector<T> dparams(5, 1, stride_param, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dd1.memcheck());
    CHECK_DEVICE_ALLOCATION(dd2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparams.memcheck());

    if(arg.unit_check || arg.norm_check)
    {
        host_strided_batch_vector<T> hd1_gold(1, 1, stride_d1, batch_count);
        host_strided_batch_vector<T> hd2_gold(1, 1, stride_d2, batch_count);
        host_strided_batch_vector<T> hx_gold(1, 1, stride_x, batch_count);
        host_strided_batch_vector<T> hy_gold(1, 1, stride_y, batch_count);
        host_strided_batch_vector<T> hparams_gold(5, 1, stride_param, batch_count);

        host_strided_batch_vector<T> rd1(1, 1, stride_d1, batch_count);
        host_strided_batch_vector<T> rd2(1, 1, stride_d2, batch_count);
        host_strided_batch_vector<T> rx(1, 1, stride_x, batch_count);
        host_strided_batch_vector<T> ry(1, 1, stride_y, batch_count);
        host_strided_batch_vector<T> rparams(5, 1, stride_param, batch_count);

        const int TEST_COUNT = 100;
        int test_count = !(arg.api & c_API_64) ? TEST_COUNT : 1; // only test 1 64bit API sizes
        for(int i = 0; i < test_count; i++)
        {
            // Initialize data on host memory
            rocblas_init_vector(hparams, arg, rocblas_client_alpha_sets_nan, true);
            rocblas_init_vector(hd1, arg, rocblas_client_alpha_sets_nan, false);
            rocblas_init_vector(hd2, arg, rocblas_client_alpha_sets_nan, false);
            rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false);
            rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);

            hd1_gold.copy_from(hd1);
            hd2_gold.copy_from(hd2);
            hx_gold.copy_from(hx);
            hy_gold.copy_from(hy);
            hparams_gold.copy_from(hparams);

            cpu_time_used = get_time_us_no_sync();
            for(size_t b = 0; b < batch_count; b++)
            {
                ref_rotmg<T>(hd1_gold[b], hd2_gold[b], hx_gold[b], hy_gold[b], hparams_gold[b]);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;

            if(arg.pointer_mode_host)
            {
                rd1.copy_from(hd1);
                rd2.copy_from(hd2);
                rx.copy_from(hx);
                ry.copy_from(hy);
                rparams.copy_from(hparams);

                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                handle.pre_test(arg);
                CHECK_ROCBLAS_ERROR((rocblas_rotmg_strided_batched_fn(handle,
                                                                      rd1,
                                                                      stride_d1,
                                                                      rd2,
                                                                      stride_d2,
                                                                      rx,
                                                                      stride_x,
                                                                      ry,
                                                                      stride_y,
                                                                      rparams,
                                                                      stride_param,
                                                                      batch_count)));
                handle.post_test(arg);

                if(arg.unit_check)
                {
                    near_check_general<T>(
                        1, 1, 1, stride_d1, rd1, hd1_gold, batch_count, rel_error);
                    near_check_general<T>(
                        1, 1, 1, stride_d2, rd2, hd2_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_x, rx, hx_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_y, ry, hy_gold, batch_count, rel_error);
                    near_check_general<T>(
                        1, 5, 1, stride_param, rparams, hparams_gold, batch_count, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_host = norm_check_general<T>(
                        'F', 1, 1, 1, stride_d1, rd1, hd1_gold, batch_count);
                    norm_error_host += norm_check_general<T>(
                        'F', 1, 1, 1, stride_d2, rd2, hd2_gold, batch_count);
                    norm_error_host
                        += norm_check_general<T>('F', 1, 1, 1, stride_x, rx, hx_gold, batch_count);
                    norm_error_host
                        += norm_check_general<T>('F', 1, 1, 1, stride_y, ry, hy_gold, batch_count);
                    norm_error_host += norm_check_general<T>(
                        'F', 1, 5, 1, stride_param, rparams, hparams_gold, batch_count);
                }
            }

            if(arg.pointer_mode_device)
            {
                // Allocate device memory

                CHECK_HIP_ERROR(dd1.transfer_from(hd1));
                CHECK_HIP_ERROR(dd2.transfer_from(hd2));
                CHECK_HIP_ERROR(dx.transfer_from(hx));
                CHECK_HIP_ERROR(dy.transfer_from(hy));
                CHECK_HIP_ERROR(dparams.transfer_from(hparams));

                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                handle.pre_test(arg);
                DAPI_CHECK(rocblas_rotmg_strided_batched_fn,
                           (handle,
                            dd1,
                            stride_d1,
                            dd2,
                            stride_d2,
                            dx,
                            stride_x,
                            dy,
                            stride_y,
                            dparams,
                            stride_param,
                            batch_count));
                handle.post_test(arg);

                CHECK_HIP_ERROR(rd1.transfer_from(dd1));
                CHECK_HIP_ERROR(rd2.transfer_from(dd2));
                CHECK_HIP_ERROR(rx.transfer_from(dx));
                CHECK_HIP_ERROR(ry.transfer_from(dy));
                CHECK_HIP_ERROR(rparams.transfer_from(dparams));

                if(arg.unit_check)
                {
                    near_check_general<T>(
                        1, 1, 1, stride_d1, rd1, hd1_gold, batch_count, rel_error);
                    near_check_general<T>(
                        1, 1, 1, stride_d2, rd2, hd2_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_x, rx, hx_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_y, ry, hy_gold, batch_count, rel_error);
                    near_check_general<T>(
                        1, 5, 1, stride_param, rparams, hparams_gold, batch_count, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_device = norm_check_general<T>(
                        'F', 1, 1, 1, stride_d1, rd1, hd1_gold, batch_count);
                    norm_error_device += norm_check_general<T>(
                        'F', 1, 1, 1, stride_d2, rd2, hd2_gold, batch_count);
                    norm_error_device
                        += norm_check_general<T>('F', 1, 1, 1, stride_x, rx, hx_gold, batch_count);
                    norm_error_device
                        += norm_check_general<T>('F', 1, 1, 1, stride_y, ry, hy_gold, batch_count);
                    norm_error_device += norm_check_general<T>(
                        'F', 1, 5, 1, stride_param, rparams, hparams_gold, batch_count);
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dd1.transfer_from(hd1));
        CHECK_HIP_ERROR(dd2.transfer_from(hd2));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dparams.transfer_from(hparams));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_rotmg_strided_batched_fn,
                          (handle,
                           dd1,
                           stride_d1,
                           dd2,
                           stride_d2,
                           dx,
                           stride_x,
                           dy,
                           stride_y,
                           dparams,
                           stride_param,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_stride_a, e_stride_b, e_stride_x, e_stride_y, e_stride_c, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         norm_error_host,
                         norm_error_device);
    }
}
