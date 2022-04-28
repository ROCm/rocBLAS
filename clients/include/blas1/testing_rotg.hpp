/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

    // Allocate device memory
    device_vector<T> da(1, 1);
    device_vector<T> db(1, 1);
    device_vector<U> dc(1, 1);
    device_vector<T> ds(1, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(nullptr, da, db, dc, ds)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, nullptr, db, dc, ds)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, da, nullptr, dc, ds)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, da, db, nullptr, ds)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_fn(handle, da, db, dc, nullptr)),
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

    host_vector<T> a(1, 1);
    host_vector<T> b(1, 1);
    host_vector<U> c(1, 1);
    host_vector<T> s(1, 1);

    bool enable_near_check_general = true;

#ifdef WIN32
    // During explicit NaN initialization (i.e., when arg.alpha=NaN), the host side computation results of OpenBLAS differs from the result of kernel computation in rocBLAS.
    // The output value of `hb_gold` is NaN in OpenBLAS and, the output value of `hb_gold` is 1.000 in rocBLAS. There was no difference observed when comparing the rocBLAS results with BLIS.
    // Therefore, using the bool enable_near_check_general to skip unit check for WIN32 during NaN initialization.

    enable_near_check_general = !rocblas_isnan(arg.alpha);
#endif

    for(int i = 0; i < TEST_COUNT; ++i)
    {
        // Initialize data on host memory
        rocblas_init_vector(a, arg, rocblas_client_alpha_sets_nan, true);
        rocblas_init_vector(b, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(c, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(s, arg, rocblas_client_alpha_sets_nan, false);

        // CPU BLAS
        host_vector<T> ha_gold = a;
        host_vector<T> hb_gold = b;
        host_vector<U> hc_gold = c;
        host_vector<T> hs_gold = s;
        cpu_time_used          = get_time_us_no_sync();
        cblas_rotg<T, U>(ha_gold, hb_gold, hc_gold, hs_gold);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            // Naming: `h` is in CPU (host) memory(eg ha), `d` is in GPU (device) memory (eg da).
            // Allocate host memory
            host_vector<T> ha = a;
            host_vector<T> hb = b;
            host_vector<U> hc = c;
            host_vector<T> hs = s;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_fn(handle, ha, hb, hc, hs)));

            if(arg.unit_check)
            {
                if(enable_near_check_general)
                {
                    near_check_general<T>(1, 1, 1, ha_gold, ha, rel_error);
                    near_check_general<T>(1, 1, 1, hb_gold, hb, rel_error);
                    near_check_general<U>(1, 1, 1, hc_gold, hc, rel_error);
                    near_check_general<T>(1, 1, 1, hs_gold, hs, rel_error);
                }
            }

            if(arg.norm_check)
            {
                error_host = norm_check_general<T>('F', 1, 1, 1, ha_gold, ha);
                error_host += norm_check_general<T>('F', 1, 1, 1, hb_gold, hb);
                error_host += norm_check_general<U>('F', 1, 1, 1, hc_gold, hc);
                error_host += norm_check_general<T>('F', 1, 1, 1, hs_gold, hs);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            // Allocate device memory
            device_vector<T> da(1, 1);
            device_vector<T> db(1, 1);
            device_vector<U> dc(1, 1);
            device_vector<T> ds(1, 1);

            // Check device memory allocation
            CHECK_DEVICE_ALLOCATION(da.memcheck());
            CHECK_DEVICE_ALLOCATION(db.memcheck());
            CHECK_DEVICE_ALLOCATION(dc.memcheck());
            CHECK_DEVICE_ALLOCATION(ds.memcheck());

            // Transfer from CPU to GPU
            CHECK_HIP_ERROR(da.transfer_from(a));
            CHECK_HIP_ERROR(db.transfer_from(b));
            CHECK_HIP_ERROR(dc.transfer_from(c));
            CHECK_HIP_ERROR(ds.transfer_from(s));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_fn(handle, da, db, dc, ds)));

            host_vector<T> ha(1);
            host_vector<T> hb(1);
            host_vector<U> hc(1);
            host_vector<T> hs(1);

            // Transfer from GPU to CPU
            CHECK_HIP_ERROR(ha.transfer_from(da));
            CHECK_HIP_ERROR(hb.transfer_from(db));
            CHECK_HIP_ERROR(hc.transfer_from(dc));
            CHECK_HIP_ERROR(hs.transfer_from(ds));

            if(arg.unit_check)
            {
                if(enable_near_check_general)
                {
                    near_check_general<T>(1, 1, 1, ha_gold, ha, rel_error);
                    near_check_general<T>(1, 1, 1, hb_gold, hb, rel_error);
                    near_check_general<U>(1, 1, 1, hc_gold, hc, rel_error);
                    near_check_general<T>(1, 1, 1, hs_gold, hs, rel_error);
                }
            }

            if(arg.norm_check)
            {
                error_device = norm_check_general<T>('F', 1, 1, 1, ha_gold, ha);
                error_device += norm_check_general<T>('F', 1, 1, 1, hb_gold, hb);
                error_device += norm_check_general<U>('F', 1, 1, 1, hc_gold, hc);
                error_device += norm_check_general<T>('F', 1, 1, 1, hs_gold, hs);
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
