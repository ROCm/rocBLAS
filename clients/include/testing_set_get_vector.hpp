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

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
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
void testing_set_get_vector(const Arguments& arg)
{
    rocblas_int          M    = arg.M;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_int          ldd  = arg.ldd;
    rocblas_local_handle handle{arg};

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || ldd <= 0)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_set_vector(M, sizeof(T), nullptr, incx, nullptr, ldd),
                              rocblas_status_invalid_size);
        EXPECT_ROCBLAS_STATUS(rocblas_get_vector(M, sizeof(T), nullptr, ldd, nullptr, incy),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M, incx);
    host_vector<T> hy(M, incy);
    host_vector<T> hy_gold(M, incy);

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    // allocate memory on device
    device_vector<T> db(M, ldd);
    CHECK_DEVICE_ALLOCATION(db.memcheck());

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);

    if(arg.unit_check || arg.norm_check)
    {
        // set GPU memory to zero
        CHECK_HIP_ERROR(hipMemset(db, 0, sizeof(T) * ldd * M));

        CHECK_ROCBLAS_ERROR(rocblas_set_vector(M, sizeof(T), hx, incx, db, ldd));
        CHECK_ROCBLAS_ERROR(rocblas_get_vector(M, sizeof(T), db, ldd, hy, incy));

        cpu_time_used = get_time_us_no_sync();

        // reference calculation
        for(size_t i = 0; i < M; i++)
        {
            hy_gold[i * incy] = hx[i * incx];
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, incy, hy, hy_gold);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, M, incy, hy, hy_gold);
        }
    }

    if(arg.timing)
    {
        int         number_timing_iterations = arg.iters;
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_set_vector(M, sizeof(T), hx, incx, db, ldd);
            rocblas_get_vector(M, sizeof(T), db, ldd, hy, incy);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_incx, e_incy, e_ldd>{}.log_args<T>(rocblas_cout,
                                                                arg,
                                                                gpu_time_used,
                                                                ArgumentLogging::NA_value,
                                                                set_get_vector_gbyte_count<T>(M),
                                                                cpu_time_used,
                                                                rocblas_error);
    }
}
