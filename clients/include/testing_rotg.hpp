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
void testing_rotg_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;

    rocblas_local_handle handle;
    device_vector<T>     a(safe_size);
    device_vector<T>     b(safe_size);
    device_vector<U>     c(safe_size);
    device_vector<T>     s(safe_size);
    if(!a || !b || !c || !s)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T, U>(nullptr, a, b, c, s)), rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T, U>(handle, nullptr, b, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T, U>(handle, a, nullptr, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T, U>(handle, a, b, nullptr, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg<T, U>(handle, a, b, c, nullptr)),
                          rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_rotg(const Arguments& arg)
{
    const int TEST_COUNT = 100;

    rocblas_local_handle handle;
    double               gpu_time_used, cpu_time_used;
    double               error_host, error_device;
    host_vector<T>       a(1);
    host_vector<T>       b(1);
    host_vector<U>       c(1);
    host_vector<T>       s(1);

    for(int i = 0; i < TEST_COUNT; ++i)
    {
        // Initial data on CPU
        rocblas_seedrand();
        rocblas_init<T>(a, 1, 1, 1);
        rocblas_init<T>(b, 1, 1, 1);
        rocblas_init<U>(c, 1, 1, 1);
        rocblas_init<T>(s, 1, 1, 1);

        // CPU BLAS
        host_vector<T> ca = a;
        host_vector<T> cb = b;
        host_vector<U> cc = c;
        host_vector<T> cs = s;
        cpu_time_used     = get_time_us();
        cblas_rotg<T, U>(ca, cb, cc, cs);
        cpu_time_used = get_time_us() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> ha = a;
            host_vector<T> hb = b;
            host_vector<U> hc = c;
            host_vector<T> hs = s;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR((rocblas_rotg<T, U>(handle, ha, hb, hc, hs)));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, 1, ca, ha);
                unit_check_general<T>(1, 1, 1, cb, hb);
                unit_check_general<U>(1, 1, 1, cc, hc);
                unit_check_general<T>(1, 1, 1, cs, hs);
            }

            if(arg.norm_check)
            {
                error_host = norm_check_general<T>('F', 1, 1, 1, ca, ha);
                error_host += norm_check_general<T>('F', 1, 1, 1, cb, hb);
                error_host += norm_check_general<U>('F', 1, 1, 1, cc, hc);
                error_host += norm_check_general<T>('F', 1, 1, 1, cs, hs);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_vector<T> da(1);
            device_vector<T> db(1);
            device_vector<U> dc(1);
            device_vector<T> ds(1);
            CHECK_HIP_ERROR(hipMemcpy(da, a, sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(db, b, sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, c, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, s, sizeof(T), hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg<T, U>(handle, da, db, dc, ds)));
            host_vector<T> ha(1);
            host_vector<T> hb(1);
            host_vector<U> hc(1);
            host_vector<T> hs(1);
            CHECK_HIP_ERROR(hipMemcpy(ha, da, sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hb, db, sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hc, dc, sizeof(U), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hs, ds, sizeof(T), hipMemcpyDeviceToHost));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, 1, ca, ha);
                unit_check_general<T>(1, 1, 1, cb, hb);
                unit_check_general<U>(1, 1, 1, cc, hc);
                unit_check_general<T>(1, 1, 1, cs, hs);
            }

            if(arg.norm_check)
            {
                error_device = norm_check_general<T>('F', 1, 1, 1, ca, ha);
                error_device += norm_check_general<T>('F', 1, 1, 1, cb, hb);
                error_device += norm_check_general<U>('F', 1, 1, 1, cc, hc);
                error_device += norm_check_general<T>('F', 1, 1, 1, cs, hs);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        host_vector<T> ha = a;
        host_vector<T> hb = b;
        host_vector<U> hc = c;
        host_vector<T> hs = s;
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocblas_rotg<T, U>(handle, ha, hb, hc, hs);
        }

        gpu_time_used = get_time_us();
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            ha = a;
            hb = b;
            hc = c;
            hs = s;
            rocblas_rotg<T, U>(handle, ha, hb, hc, hs);
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
