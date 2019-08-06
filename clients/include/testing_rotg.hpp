/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T1, typename T2 = T1>
void testing_rotg_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;

    rocblas_local_handle handle;
    device_vector<T1>    a(safe_size);
    device_vector<T1>    b(safe_size);
    device_vector<T2>    c(safe_size);
    device_vector<T2>    s(safe_size);
    if (!a || !b || !c || !d)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T1, T2>(nullptr, a, b, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T1, T2>(handle, nullptr, b, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T1, T2>(handle, a, nullptr, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T1, T2>(handle, a, b, nullptr, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T1, T2>(handle, a, b, c, nullptr)),
                          rocblas_status_invalid_pointer);
}

template <typename T1, typename T2 = T1>
void testing_rotg(const Arguments& arg)
{
    rocblas_local_handle handle;
    double gpu_time_used, cpu_time_used;
    double error_host, error_device;

    host_vector<T1> params(4);

    // Initial data on CPU
    rocblas_seed_rand();
    rocblas_init<T1>(params, 1, 4, 1);

    // CPU BLAS
    host_vector<T1> cparams = params;
    cpu_time_used = get_time_us();
    cblas_rotg<T1>(&cparams[0], &cparams[1], &cparams[2], &cparams[3]);
    cpu_time_used = get_time_us() - cpu_time_used;

    // Test rocblas_pointer_mode_host
    {
        host_vector<T1> hparams = params;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_rotg<T1, T2>(handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3]));

        if (arg.unit_check)
            unit_check_general<T1>(1, 4, 1, cparams, hparams);

        if (arg.norm_check)
            error_host = norm_check_general<T1>('F', 1, 4, 1, cparams, hparams);
    }

    // Test rocblas_pointer_mode_device
    {
        device_vector<T1> dparams(4);
        CHECK_HIP_ERROR(hipMemcpy(dparams, params, 4 * sizeof(T1), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_rotg<T1, T2>(dparams, dparams + 1, dparams + 2, dparams + 3));
        host_vector<T1> hparams(4);
        CHECK_HIP_ERROR(hipMemcpy(hparams, dparams, 4 * sizeof(T1), hipMemcpyDeviceToHost));
        
        if (arg.unit_check)
            unit_check<T1>(1, 4, 1, cparams, hparams);

        if (arg.norm_check)
            error_device = norm_check_general<T1>('F', 1, 4, 1, cparams, hparams);
    }

    if (arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        host_vector<T1> hparams = params;
        for (int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocblas_rotg<T1, T2>(handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3]);
        }

        gpu_time_used = get_time_us();
        for (int iter = 0; iter < number_hot_calls; ++iter)
        {
            hparams = params;
            rocblas_rotg<T1, T2>(handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3]);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            std::cout << ",norm_error_host_ptr,norm_error_dev_ptr";
        std::cout << std::endl;

        std::cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            std::cout << ',' << error_host << ',' << error_device;
        std::cout << std::endl;
    }
}
