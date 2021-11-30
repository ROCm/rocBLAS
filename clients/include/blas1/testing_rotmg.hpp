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

template <typename T>
void testing_rotmg_bad_arg(const Arguments& arg)
{
    auto rocblas_rotgm_fn = arg.fortran ? rocblas_rotmg<T, true> : rocblas_rotmg<T, false>;

    static const size_t safe_size = 5;

    rocblas_local_handle handle{arg};
    device_vector<T>     d1(safe_size);
    device_vector<T>     d2(safe_size);
    device_vector<T>     x1(safe_size);
    device_vector<T>     y1(safe_size);
    device_vector<T>     param(safe_size);
    CHECK_DEVICE_ALLOCATION(d1.memcheck());
    CHECK_DEVICE_ALLOCATION(d2.memcheck());
    CHECK_DEVICE_ALLOCATION(x1.memcheck());
    CHECK_DEVICE_ALLOCATION(y1.memcheck());
    CHECK_DEVICE_ALLOCATION(param.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_fn(nullptr, d1, d2, x1, y1, param),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_fn(handle, nullptr, d2, x1, y1, param),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_fn(handle, d1, nullptr, x1, y1, param),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_fn(handle, d1, d2, nullptr, y1, param),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_fn(handle, d1, d2, x1, nullptr, param),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotgm_fn(handle, d1, d2, x1, y1, nullptr),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotmg(const Arguments& arg)
{
    auto rocblas_rotgm_fn = arg.fortran ? rocblas_rotmg<T, true> : rocblas_rotmg<T, false>;

    const int TEST_COUNT = 100;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double               error_host, error_device;
    const T              rel_error = std::numeric_limits<T>::epsilon() * 1000;
    host_vector<T>       params(9);

    for(int i = 0; i < TEST_COUNT; ++i)
    {
        // Initialize data on host memory
        rocblas_init_vector(params, arg, 9, 1, 0, 1, rocblas_client_alpha_sets_nan, true);

        // CPU BLAS
        host_vector<T> cparams = params;
        cpu_time_used          = get_time_us_no_sync();
        cblas_rotmg<T>(&cparams[0], &cparams[1], &cparams[2], &cparams[3], &cparams[4]);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> hparams = params;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(rocblas_rotgm_fn(
                handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]));

            //when (input vectors are initialized with NaN's) the resultant output vector for both the cblas and rocBLAS are NAn's.  The `near_check_general` function compares the output of both the results (i.e., Nan's) and
            //throws an error. That is the reason why it is enclosed in an `if(!rocblas_isnan(arg.alpha))` loop to skip the check.
            if(!rocblas_isnan(arg.alpha))
                if(arg.unit_check)
                    near_check_general<T>(1, 9, 1, cparams, hparams, rel_error);

            if(arg.norm_check)
                error_host = norm_check_general<T>('F', 1, 9, 1, cparams, hparams);
        }

        // Test rocblas_pointer_mode_device
        {
            device_vector<T> dparams(9);
            CHECK_DEVICE_ALLOCATION(dparams.memcheck());
            CHECK_HIP_ERROR(hipMemcpy(dparams, params, 9 * sizeof(T), hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(rocblas_rotgm_fn(
                handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));
            host_vector<T> hparams(9);
            CHECK_HIP_ERROR(hipMemcpy(hparams, dparams, 9 * sizeof(T), hipMemcpyDeviceToHost));

            if(!rocblas_isnan(arg.alpha))
                if(arg.unit_check)
                    near_check_general<T>(1, 9, 1, cparams, hparams, rel_error);

            if(arg.norm_check)
                error_device = norm_check_general<T>('F', 1, 9, 1, cparams, hparams);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        host_vector<T> hparams = params;
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocblas_rotgm_fn(
                handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hparams = params;
            rocblas_rotgm_fn(
                handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]);
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used) / number_hot_calls;

        rocblas_cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            rocblas_cout << ",norm_error_host_ptr,norm_error_dev_ptr";
        rocblas_cout << std::endl;

        rocblas_cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            rocblas_cout << ',' << error_host << ',' << error_device;
        rocblas_cout << std::endl;
    }
}
