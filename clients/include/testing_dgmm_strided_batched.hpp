/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_dgmm_strided_batched_fn
        = FORTRAN ? rocblas_dgmm_strided_batched<T, true> : rocblas_dgmm_strided_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int ldc  = 100;

    const rocblas_int batch_count = 5;

    const rocblas_int abs_incx = incx > 0 ? incx : -incx;

    const rocblas_side side = rocblas_side_right;

    rocblas_local_handle handle;

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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_dgmm_strided_batched_fn
        = FORTRAN ? rocblas_dgmm_strided_batched<T, true> : rocblas_dgmm_strided_batched<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int lda  = arg.lda;
    rocblas_int incx = arg.incx;
    rocblas_int ldc  = arg.ldc;

    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_x    = arg.stride_x;
    rocblas_int stride_c    = arg.stride_c;
    rocblas_int batch_count = arg.batch_count;

    rocblas_int abs_incx = incx > 0 ? incx : -incx;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = std::numeric_limits<T>::max();

    if((stride_a > 0) && (stride_a < lda * N))
        rocblas_cout << "WARNING: stride_a < lda * N" << std::endl;
    if((stride_c > 0) && (stride_c < ldc * N))
        rocblas_cout << "WARNING: stride_c < ldc * N" << std::endl;
    if((stride_x > 0) && (stride_x < incx * (rocblas_side_right == side ? N : M)))
        rocblas_cout << "WARNING: stride_x < incx * (rocblas_side_right == side ? N : M))"
                     << std::endl;

    size_t size_A = batch_count * stride_a;
    size_t size_x = batch_count * stride_x;
    size_t size_C = batch_count * stride_c;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0 || incx == 0;
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

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA);
    rocblas_init<T>(hX);
    rocblas_init<T>(hC);

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

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // reference calculation for golden result
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        cpu_time_used     = get_time_us();

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
                              + hX_copy[shift_x + i2 * incx + i3 * stride_x];
                    }
                    else
                    {
                        hC_gold[i1 + i2 * ldc + i3 * stride_c]
                            = hA_copy[i1 + i2 * lda + i3 * stride_a]
                              + hX_copy[shift_x + i1 * incx + i3 * stride_x];
                    }
                }
            }
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = dgmm_gflop_count<T>(M, N) / cpu_time_used * 1e6;

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

        gpu_time_used = get_time_us(); // in microseconds
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
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = dgmm_gflop_count<T>(M, N) * batch_count * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "side,M,N,lda,incx,ldc,batch_count,rocblas-Gflops,us";
        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << ",CPU-Gflops,us,norm_error";
        }
        rocblas_cout << std::endl;

        rocblas_cout << arg.side << "," << M << "," << N << "," << lda << "," << incx << "," << ldc
                     << "," << batch_count << "," << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls << ",";

        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << cblas_gflops << "," << cpu_time_used << ",";
            rocblas_cout << rocblas_error;
        }
        rocblas_cout << std::endl;
    }
}
