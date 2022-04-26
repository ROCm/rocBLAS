/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

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
void testing_rotm_bad_arg(const Arguments& arg)
{
    auto rocblas_rotm_fn = arg.fortran ? rocblas_rotm<T, true> : rocblas_rotm<T, false>;

    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    static const size_t safe_size = 100;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);
    device_vector<T> dparam(5, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(nullptr, N, dx, incx, dy, incy, dparam),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(handle, N, nullptr, incx, dy, incy, dparam),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(handle, N, dx, incx, nullptr, incy, dparam),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(handle, N, dx, incx, dy, incy, nullptr),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotm(const Arguments& arg)
{
    auto rocblas_rotm_fn = arg.fortran ? rocblas_rotm<T, true> : rocblas_rotm<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;
    const T rel_error          = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_rotm_fn(handle, N, nullptr, incx, nullptr, incy, nullptr));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<T> hx(N, incx ? incx : 1);
    host_vector<T> hy(N, incy ? incy : 1);
    host_vector<T> hdata(4, 1);
    host_vector<T> hparam(5, 1);

    // Allocate device memory
    device_vector<T> dx(N, incx ? incx : 1);
    device_vector<T> dy(N, incy ? incy : 1);
    device_vector<T> dparam(5, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);
    rocblas_init_vector(hdata, arg, rocblas_client_alpha_sets_nan, false);

    // generating simply one set of hparam which will not be appropriate for testing
    // that it zeros out the second element of the rotm vector parameter
    memset(hparam.data(), 0, 5 * sizeof(T));

    cblas_rotmg<T>(&hdata[0], &hdata[1], &hdata[2], &hdata[3], hparam);

    const int FLAG_COUNT        = 4;
    const T   FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};
    for(int i = 0; i < FLAG_COUNT; ++i)
    {
        hparam[0] = FLAGS[i];

        // CPU BLAS reference data
        host_vector<T> hx_gold = hx;
        host_vector<T> hy_gold = hy;
        cpu_time_used          = get_time_us_no_sync();
        cblas_rotm<T>(N, hx_gold, incx, hy_gold, incy, hparam);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check || arg.norm_check)
        {
            // Test rocblas_pointer_mode_host
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

                CHECK_HIP_ERROR(dx.transfer_from(hx));
                CHECK_HIP_ERROR(dy.transfer_from(hy));

                CHECK_ROCBLAS_ERROR(rocblas_rotm_fn(handle, N, dx, incx, dy, incy, hparam));

                host_vector<T> rx(N, incx ? incx : 1);
                host_vector<T> ry(N, incy ? incy : 1);

                CHECK_HIP_ERROR(rx.transfer_from(dx));
                CHECK_HIP_ERROR(ry.transfer_from(dy));

                if(arg.unit_check)
                {
                    near_check_general<T>(1, N, abs_incx, hx_gold, rx, rel_error);
                    near_check_general<T>(1, N, abs_incy, hy_gold, ry, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_host_x += norm_check_general<T>('F', 1, N, abs_incx, hx_gold, rx);
                    norm_error_host_y += norm_check_general<T>('F', 1, N, abs_incy, hy_gold, ry);
                }
            }

            // Test rocblas_pointer_mode_device
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

                CHECK_HIP_ERROR(dx.transfer_from(hx));
                CHECK_HIP_ERROR(dy.transfer_from(hy));
                CHECK_HIP_ERROR(dparam.transfer_from(hparam));

                CHECK_ROCBLAS_ERROR(rocblas_rotm_fn(handle, N, dx, incx, dy, incy, dparam));

                host_vector<T> rx(N, incx ? incx : 1);
                host_vector<T> ry(N, incy ? incy : 1);

                CHECK_HIP_ERROR(rx.transfer_from(dx));
                CHECK_HIP_ERROR(ry.transfer_from(dy));

                if(arg.unit_check)
                {
                    near_check_general<T>(1, N, abs_incx, hx_gold, rx, rel_error);
                    near_check_general<T>(1, N, abs_incy, hy_gold, ry, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_device_x += norm_check_general<T>('F', 1, N, abs_incx, hx_gold, rx);
                    norm_error_device_y += norm_check_general<T>('F', 1, N, abs_incy, hy_gold, ry);
                }
            }
        }
    }

    if(arg.timing)
    {
        // Initializing flag value to -1
        hparam[0]             = FLAGS[0];
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dparam.transfer_from(hparam));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotm_fn(handle, N, dx, incx, dy, incy, dparam);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotm_fn(handle, N, dx, incx, dy, incy, dparam);
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<T>(rocblas_cout,
                                                         arg,
                                                         gpu_time_used,
                                                         rotm_gflop_count<T>(N, hparam[0]),
                                                         rotm_gbyte_count<T>(N, hparam[0]),
                                                         cpu_time_used,
                                                         norm_error_host_x,
                                                         norm_error_device_x,
                                                         norm_error_host_y,
                                                         norm_error_device_y);
    }
}
