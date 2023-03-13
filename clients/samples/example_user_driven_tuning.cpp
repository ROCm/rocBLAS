/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

// need to enable unstable api
#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API
#include "rocblas.h"
#include "utility.hpp"
#include <hip/hip_runtime.h>

#include <chrono>
#include <map>
#include <random>
#include <vector>

#define DIM1 64
#define DIM2 64
#define DIM3 10024

int main()
{
    rocblas_int cold_calls = 2;
    rocblas_int hot_calls  = 10; // these values match rocblas-bench defaults

    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    float             alpha = 1.1f, beta = 0.9f;

    rocblas_int    m = DIM1, n = DIM2, k = DIM3;
    rocblas_int    lda, ldb, ldc;
    size_t         size_a, size_b, size_c;
    rocblas_stride a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    rocblas_cout << "user driven tuning example" << std::endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a     = size_t(k) * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
    }
    else
    {
        lda        = k;
        size_a     = size_t(m) * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b     = size_t(n) * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
    }
    else
    {
        ldb        = n;
        size_b     = size_t(k) * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
    }
    ldc    = m;
    size_c = size_t(n) * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);

    // initial data on host
    // Random number generator
    std::mt19937 rng;
    srand(1);
    for(size_t i = 0; i < size_a; ++i)
    {
        ha[i] = std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng);
    }
    for(size_t i = 0; i < size_b; ++i)
    {
        hb[i] = std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng);
    }
    for(size_t i = 0; i < size_c; ++i)
    {
        hc[i] = std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng);
    }

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    rocblas_datatype input_type   = rocblas_datatype_f32_r;
    rocblas_datatype output_type  = rocblas_datatype_f32_r;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    // macros
#define GEMM_EX_ARGS                                                                            \
    handle,                 \
    transa, transb,         \
    m, n, k,                \
    &alpha,                 \
    da, input_type, lda,    \
    db, input_type, ldb,    \
    &beta,                  \
    dc, output_type, ldc,   \
    dc, output_type, ldc,   \
    compute_type,           \
    rocblas_gemm_algo_solution_index
#define rocblas_gemm_exM(...) rocblas_gemm_ex(__VA_ARGS__)

    rocblas_int size;

    // Get number of solutions that match this GEMM problem's type
    // NOTE: for batched problems use 'rocblas_gemm_batched_ex_get_solutions_by_type'
    //       for strided/batched problems use 'rocblas_gemm_ex_get_solutions_by_type' 
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions_by_type(handle,
                                              input_type, output_type, compute_type,
                                              rocblas_gemm_flags_none,
                                              NULL,
                                              &size
                                              ));
    rocblas_cout << size << " solution(s) found that match this GEMM's type." << std::endl;

    // Get number of solutions that can solve this GEMM problem
    // NOTE: for batched problems use 'rocblas_gemm_batched_ex_get_solutions'
    //       for strided/batched problems use 'rocblas_gemm_strided_batched_ex_get_solutions' 
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));
    rocblas_cout << size << " solution(s) found that can solve this GEMM." << std::endl;

    // Fill array with list of solutions that match type
    // Note: some of these may be invalid
    std::vector<rocblas_int> ary(size);
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions_by_type(handle,
                                              input_type, output_type, compute_type,
                                              rocblas_gemm_flags_none,
                                              ary.data(),
                                              &size));

    // Example basic benchmark loop
    double         bestTime = std::numeric_limits<double>::max();
    rocblas_int    bestSol  = -1;
    rocblas_status status;
    for(auto sol : ary)
    {
        // Check solution is valid
        status = rocblas_gemm_exM(GEMM_EX_ARGS, sol, rocblas_gemm_flags_none);
        if(status == rocblas_status_invalid_value)
        {
            rocblas_cout << "Solution " << sol << " not valid for this problem." << std::endl;
            continue;
        }
        
        // warmup
        for(rocblas_int c = 0; c < cold_calls; ++c)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, sol, rocblas_gemm_flags_none));
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        double time = get_time_us_sync(stream); // in microseconds

        // timing loop
        for(rocblas_int c = 0; c < hot_calls; ++c)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, sol, rocblas_gemm_flags_none));
        }
        time = get_time_us_sync(stream) - time;

        double avg_time = hot_calls ? time / hot_calls : 0;
        rocblas_cout << "Sol " << sol << ": " << avg_time << " us" << std::endl;

        // track winner
        if(avg_time < bestTime)
        {
            bestSol  = sol;
            bestTime = avg_time;
        }
    }
    rocblas_cout << "Winner: " << bestSol << " in " << bestTime << " us" << std::endl;

    // Check if solution is valid for problem (success case)
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_exM(GEMM_EX_ARGS, bestSol, rocblas_gemm_flags_check_solution_index));

    // Solve using winner
    CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, bestSol, rocblas_gemm_flags_none));

    // Solve using default solution
    CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, 0, rocblas_gemm_flags_none));

    return EXIT_SUCCESS;
}
