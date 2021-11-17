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
void testing_rotg_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_rotg_fn = FORTRAN ? rocblas_rotg<T, U, true> : rocblas_rotg<T, U, false>;

    static const size_t safe_size = 1;

    rocblas_local_handle handle{arg};
    device_vector<T>     a(safe_size);
    device_vector<T>     b(safe_size);
    device_vector<U>     c(safe_size);
    device_vector<T>     s(safe_size);
    CHECK_DEVICE_ALLOCATION(a.memcheck());
    CHECK_DEVICE_ALLOCATION(b.memcheck());
    CHECK_DEVICE_ALLOCATION(c.memcheck());
    CHECK_DEVICE_ALLOCATION(s.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(nullptr, a, b, c, s)), rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, nullptr, b, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, a, nullptr, c, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, a, b, nullptr, s)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, a, b, c, nullptr)),
                          rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_rotg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_rotg_fn = FORTRAN ? rocblas_rotg<T, U, true> : rocblas_rotg<T, U, false>;

    const int TEST_COUNT = 100;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double               error_host, error_device;
    const U              rel_error = std::numeric_limits<U>::epsilon() * 1000;
    host_vector<T>       a(1);
    host_vector<T>       b(1);
    host_vector<U>       c(1);
    host_vector<T>       s(1);

    for(int i = 0; i < TEST_COUNT; ++i)
    {
        // Initialize data on host memory
        rocblas_init_vector(a, arg, 1, 1, 0, 1, true);
        rocblas_init_vector(b, arg, 1, 1, 0, 1, false);
        rocblas_init_vector(c, arg, 1, 1, 0, 1, false);
        rocblas_init_vector(s, arg, 1, 1, 0, 1, false);

        // CPU BLAS
        host_vector<T> ca = a;
        host_vector<T> cb = b;
        host_vector<U> cc = c;
        host_vector<T> cs = s;
        cpu_time_used     = get_time_us_no_sync();
        cblas_rotg<T, U>(ca, cb, cc, cs);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> ha = a;
            host_vector<T> hb = b;
            host_vector<U> hc = c;
            host_vector<T> hs = s;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_fn(handle, ha, hb, hc, hs)));

            //when (input vectors are initialized with NaN's) the resultant output vector for both the cblas and rocBLAS are NAn's.  The `near_check_general` function compares the output of both the results (i.e., Nan's) and
            //throws an error. That is the reason why it is enclosed in an `if(!rocblas_isnan(arg.alpha))` loop to skip the check.
            if(!rocblas_isnan(arg.alpha))
            {
                if(arg.unit_check)
                {
                    near_check_general<T>(1, 1, 1, ca, ha, rel_error);
                    near_check_general<T>(1, 1, 1, cb, hb, rel_error);
                    near_check_general<U>(1, 1, 1, cc, hc, rel_error);
                    near_check_general<T>(1, 1, 1, cs, hs, rel_error);
                }
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
            CHECK_DEVICE_ALLOCATION(da.memcheck());
            CHECK_DEVICE_ALLOCATION(db.memcheck());
            CHECK_DEVICE_ALLOCATION(dc.memcheck());
            CHECK_DEVICE_ALLOCATION(ds.memcheck());
            CHECK_HIP_ERROR(hipMemcpy(da, a, sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(db, b, sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, c, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, s, sizeof(T), hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_fn(handle, da, db, dc, ds)));
            host_vector<T> ha(1);
            host_vector<T> hb(1);
            host_vector<U> hc(1);
            host_vector<T> hs(1);
            CHECK_HIP_ERROR(hipMemcpy(ha, da, sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hb, db, sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hc, dc, sizeof(U), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hs, ds, sizeof(T), hipMemcpyDeviceToHost));

            if(!rocblas_isnan(arg.alpha))
            {
                if(arg.unit_check)
                {
                    near_check_general<T>(1, 1, 1, ca, ha, rel_error);
                    near_check_general<T>(1, 1, 1, cb, hb, rel_error);
                    near_check_general<U>(1, 1, 1, cc, hc, rel_error);
                    near_check_general<T>(1, 1, 1, cs, hs, rel_error);
                }
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
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        host_vector<T> ha = a;
        host_vector<T> hb = b;
        host_vector<U> hc = c;
        host_vector<T> hs = s;
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocblas_rotg_fn(handle, ha, hb, hc, hs);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            ha = a;
            hb = b;
            hc = c;
            hs = s;
            rocblas_rotg_fn(handle, ha, hb, hc, hs);
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
