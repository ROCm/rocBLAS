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
void testing_set_get_vector_async(const Arguments& arg)
{
    rocblas_int          M    = arg.M;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_int          ldd  = arg.ldd;
    rocblas_local_handle handle{arg};

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || ldd <= 0)
    {
        static const size_t safe_size = 100;

        host_vector<T>   hx(safe_size);
        host_vector<T>   hy(safe_size);
        device_vector<T> db(safe_size);
        CHECK_DEVICE_ALLOCATION(db.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_set_vector_async(M, sizeof(T), hx, incx, db, ldd, stream),
                              rocblas_status_invalid_size);
        EXPECT_ROCBLAS_STATUS(rocblas_get_vector_async(M, sizeof(T), db, ldd, hy, incy, stream),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_pinned_vector<T> hp_x(M, incx);
    host_pinned_vector<T> hp_y(M, incy);

    host_vector<T> hy(M, incy);
    host_vector<T> hy_gold(M, size_t(incy));
    host_vector<T> hb(M, ldd);

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    // allocate memory on device
    device_vector<T> db(M * size_t(ldd));
    CHECK_DEVICE_ALLOCATION(db.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hp_x, 1, M, incx);
    rocblas_init<T>(hp_y, 1, M, incy);

    if(arg.unit_check || arg.norm_check)
    {
        // set device memory to be random
        rocblas_init<T>(hb, 1, M, ldd);
        CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * ldd * M, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_vector_async(M, sizeof(T), hp_x, incx, db, ldd, stream));
        CHECK_ROCBLAS_ERROR(rocblas_get_vector_async(M, sizeof(T), db, ldd, hp_y, incy, stream));

        cpu_time_used = get_time_us_no_sync();

        // reference calculation
        for(int i = 0; i < M; i++)
        {
            hy_gold[i * incy] = hp_x[i * incx];
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        hipStreamSynchronize(stream);
        hy.assign(&hp_y[0], &hp_y[0] + M * incy); // copy to host_vector for _check_ compatibility

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
            rocblas_set_vector_async(M, sizeof(T), hp_x, incx, db, ldd, stream);
            rocblas_get_vector_async(M, sizeof(T), db, ldd, hp_y, incy, stream);
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
