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
void testing_set_get_matrix_async(const Arguments& arg)
{
    rocblas_int          rows = arg.M;
    rocblas_int          cols = arg.N;
    rocblas_int          lda  = arg.lda;
    rocblas_int          ldb  = arg.ldb;
    rocblas_int          ldc  = arg.ldc;
    rocblas_local_handle handle{arg};

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalidGPUMatrix = rows < 0 || cols < 0 || ldc <= 0 || ldc < rows;
    bool invalidSet       = invalidGPUMatrix || lda <= 0 || lda < rows;
    bool invalidGet       = invalidGPUMatrix || ldb <= 0 || ldb < rows;

    if(invalidSet || invalidGet)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_set_matrix_async(rows, cols, sizeof(T), nullptr, lda, nullptr, ldc, stream),
            invalidSet ? rocblas_status_invalid_size : rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_get_matrix_async(rows, cols, sizeof(T), nullptr, ldc, nullptr, ldb, stream),
            invalidGet ? rocblas_status_invalid_size : rocblas_status_invalid_pointer);

        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory,
    host_pinned_vector<T> ha(cols * size_t(lda));
    host_pinned_vector<T> hb(cols * size_t(ldb));
    host_vector<T>        hc(cols * size_t(ldc));
    host_vector<T>        hb_gold(cols * size_t(ldb));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    // allocate memory on device
    device_vector<T> dc(cols * size_t(ldc));
    CHECK_DEVICE_ALLOCATION(dc.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(ha, rows, cols, lda);
    rocblas_init<T>(hb, rows, cols, ldb);
    rocblas_init<T>(hc, rows, cols, ldc);

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        rocblas_init<T>(hb, rows, cols, ldb);
        rocblas_init<T>(hc, rows, cols, ldc);
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(T) * ldc * cols, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(
            rocblas_set_matrix_async(rows, cols, sizeof(T), ha, lda, dc, ldc, stream));
        CHECK_ROCBLAS_ERROR(
            rocblas_get_matrix_async(rows, cols, sizeof(T), dc, ldc, hb, ldb, stream));

        // reference calculation
        cpu_time_used = get_time_us_no_sync();
        for(int i1 = 0; i1 < rows; i1++)
            for(int i2 = 0; i2 < cols; i2++)
                hb_gold[i1 + i2 * ldb] = ha[i1 + i2 * lda];

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        hipStreamSynchronize(stream);

        if(arg.unit_check)
        {
            unit_check_general<T>(rows, cols, ldb, hb, hb_gold);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', rows, cols, ldb, hb, hb_gold);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_set_matrix_async(rows, cols, sizeof(T), ha, lda, dc, ldc, stream);
            rocblas_get_matrix_async(rows, cols, sizeof(T), dc, ldc, hb, ldb, stream);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_set_matrix_async(rows, cols, sizeof(T), ha, lda, dc, ldc, stream);
            rocblas_get_matrix_async(rows, cols, sizeof(T), dc, ldc, hb, ldb, stream);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_lda, e_ldb, e_ldc>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            set_get_matrix_gbyte_count<T>(rows, cols),
            cpu_time_used,
            rocblas_error);
    }
}
