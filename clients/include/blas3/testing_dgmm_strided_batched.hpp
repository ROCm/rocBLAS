/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */

template <typename T>
void testing_dgmm_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_strided_batched_fn = arg.fortran ? rocblas_dgmm_strided_batched<T, true>
                                                       : rocblas_dgmm_strided_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int ldc  = 100;

    const rocblas_int batch_count = 5;

    const rocblas_int abs_incx = incx > 0 ? incx : -incx;

    const rocblas_side side = rocblas_side_right;

    rocblas_local_handle handle{arg};

    const rocblas_stride stride_a = N * size_t(lda);
    const rocblas_stride stride_x = (rocblas_side_right == side ? N : M) * size_t(abs_incx);
    const rocblas_stride stride_c = N * size_t(ldc);

    size_t size_A = batch_count * stride_a;
    size_t size_x = batch_count * stride_x;
    size_t size_C = batch_count * stride_c;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dX(size_x);
    device_vector<T> dC(size_C);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dX.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          lda,
                                                          stride_a,
                                                          dX,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dX,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(nullptr,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dX,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_dgmm_strided_batched(const Arguments& arg)
{
    auto rocblas_dgmm_strided_batched_fn = arg.fortran ? rocblas_dgmm_strided_batched<T, true>
                                                       : rocblas_dgmm_strided_batched<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = rocblas_side_right == side ? size_t(N) : size_t(M);

    rocblas_int lda  = arg.lda;
    rocblas_int incx = arg.incx;
    rocblas_int ldc  = arg.ldc;

    rocblas_stride stride_a = arg.stride_a;
    rocblas_stride stride_x = arg.stride_x;
    if(!stride_x)
        stride_x = 1; // incx = 0 case
    rocblas_stride stride_c    = arg.stride_c;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_int abs_incx = incx > 0 ? incx : -incx;

    double gpu_time_used, cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    if((stride_a > 0) && (stride_a < size_t(lda) * N))
    {
        rocblas_cout << "WARNING: stride_a < lda * N, setting stride_a = lda * N " << std::endl;
        stride_a = N * size_t(lda);
    }
    if((stride_c > 0) && (stride_c < size_t(ldc) * N))
    {
        rocblas_cout << "WARNING: stride_c < ldc * N, setting stride_c = lda * N" << std::endl;
        stride_c = N * size_t(ldc);
    }
    if((stride_x > 0) && (stride_x < size_t(abs_incx) * K))
    {
        rocblas_cout << "WARNING: stride_x < incx * (rocblas_side_right == side ? N : M)),\n"
                        "setting stride_x = incx * (rocblas_side_right == side ? N : M))"
                     << std::endl;
        stride_x = K * size_t(abs_incx);
    }

    size_t size_A = batch_count * stride_a;
    size_t size_x = batch_count * stride_x;
    size_t size_C = batch_count * stride_c;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0;
    if(invalid_size || M == 0 || N == 0 || batch_count == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                              side,
                                                              M,
                                                              N,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A), hA_copy(size_A);
    host_vector<T> hX(size_x), hX_copy(size_x);
    host_vector<T> hC(size_C);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_gold(size_C);
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA_copy.memcheck());
    CHECK_HIP_ERROR(hX.memcheck());
    CHECK_HIP_ERROR(hX_copy.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        M,
                        N,
                        lda,
                        stride_a,
                        batch_count,
                        rocblas_client_never_set_nan,
                        rocblas_client_general_matrix,
                        true);
    rocblas_init_vector(hX, arg, size_x, 1, 1, 0, rocblas_client_never_set_nan, false, true);
    rocblas_init_matrix(hC,
                        arg,
                        M,
                        N,
                        ldc,
                        stride_c,
                        batch_count,
                        rocblas_client_never_set_nan,
                        rocblas_client_general_matrix);

    hA_copy = hA;
    hX_copy = hX;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dX(size_x);
    device_vector<T> dC(size_C);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dX.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dX.transfer_from(hX));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_dgmm_strided_batched_fn(handle,
                                                            side,
                                                            M,
                                                            N,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dX,
                                                            incx,
                                                            stride_x,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        // reference calculation for golden result
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (K - 1) : 0;
        cpu_time_used     = get_time_us_no_sync();

        for(size_t i3 = 0; i3 < batch_count; i3++)
        {
            for(size_t i1 = 0; i1 < M; i1++)
            {
                for(size_t i2 = 0; i2 < N; i2++)
                {
                    if(rocblas_side_right == side)
                    {
                        hC_gold[i1 + i2 * ldc + i3 * stride_c]
                            = hA_copy[i1 + i2 * lda + i3 * stride_a]
                              * hX_copy[shift_x + i2 * incx + i3 * stride_x];
                    }
                    else
                    {
                        hC_gold[i1 + i2 * ldc + i3 * stride_c]
                            = hA_copy[i1 + i2 * lda + i3 * stride_a]
                              * hX_copy[shift_x + i1 * incx + i3 * stride_x];
                    }
                }
            }
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU result
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_dgmm_strided_batched_fn(handle,
                                            side,
                                            M,
                                            N,
                                            dA,
                                            lda,
                                            stride_a,
                                            dX,
                                            incx,
                                            stride_x,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_dgmm_strided_batched_fn(handle,
                                            side,
                                            M,
                                            N,
                                            dA,
                                            lda,
                                            stride_a,
                                            dX,
                                            incx,
                                            stride_x,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side,
                      e_M,
                      e_N,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         dgmm_gflop_count<T>(M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
