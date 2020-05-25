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
void testing_dgmm_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_dgmm_batched_fn
        = FORTRAN ? rocblas_dgmm_batched<T, true> : rocblas_dgmm_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda      = 100;
    const rocblas_int incx     = 1;
    const rocblas_int ldc      = 100;
    const rocblas_int abs_incx = incx > 0 ? incx : -incx;

    const rocblas_int batch_count = 5;

    const rocblas_side side = rocblas_side_right;

    rocblas_local_handle handle;

    size_t size_A = N * size_t(lda);
    size_t size_x = (rocblas_side_right == side ? N : M) * size_t(abs_incx);
    size_t size_C = N * size_t(ldc);

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dX(size_x, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dX.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(handle, side, M, N, nullptr, lda, dX, incx, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(handle, side, M, N, dA, lda, nullptr, incx, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(handle, side, M, N, dA, lda, dX, incx, nullptr, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(nullptr, side, M, N, dA, lda, dX, incx, dC, ldc, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_dgmm_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_dgmm_batched_fn
        = FORTRAN ? rocblas_dgmm_batched<T, true> : rocblas_dgmm_batched<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int lda      = arg.lda;
    rocblas_int incx     = arg.incx;
    rocblas_int ldc      = arg.ldc;
    rocblas_int abs_incx = incx > 0 ? incx : -incx;

    rocblas_int batch_count = arg.batch_count;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = std::numeric_limits<T>::max();

    rocblas_local_handle handle;

    size_t size_A = size_t(lda) * size_t(N);
    size_t size_C = size_t(ldc) * size_t(N);
    size_t size_x = size_t(abs_incx) * (rocblas_side_right == side ? size_t(N) : size_t(M));

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0 || incx == 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_dgmm_batched_fn(
                handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc, batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count), hA_copy(size_A, 1, batch_count);
    host_batch_vector<T> hX(size_x, 1, batch_count), hX_copy(size_x, 1, batch_count);
    host_batch_vector<T> hC(size_C, 1, batch_count);
    host_batch_vector<T> hC_1(size_C, 1, batch_count);
    host_batch_vector<T> hC_gold(size_C, 1, batch_count);
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

    hA_copy.copy_from(hA);
    hX_copy.copy_from(hX);

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dX(size_x, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
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
        CHECK_ROCBLAS_ERROR(rocblas_dgmm_batched_fn(handle,
                                                    side,
                                                    M,
                                                    N,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dX.ptr_on_device(),
                                                    incx,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // reference calculation for golden result
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        cpu_time_used     = get_time_us();

        for(int i3 = 0; i3 < batch_count; i3++)
        {

            auto hA_copy_p = hA_copy[i3];
            auto hX_copy_p = hX_copy[i3];
            auto hC_gold_p = hC_gold[i3];

            for(size_t i1 = 0; i1 < M; i1++)
            {
                for(size_t i2 = 0; i2 < N; i2++)
                {
                    if(rocblas_side_right == side)
                    {
                        hC_gold_p[i1 + i2 * ldc]
                            = hA_copy_p[i1 + i2 * lda] + hX_copy_p[shift_x + i2 * incx];
                    }
                    else
                    {
                        hC_gold_p[i1 + i2 * ldc]
                            = hA_copy_p[i1 + i2 * lda] + hX_copy_p[shift_x + i1 * incx];
                    }
                }
            }
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = dgmm_gflop_count<T>(M, N) * batch_count / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_dgmm_batched_fn(handle,
                                    side,
                                    M,
                                    N,
                                    dA.ptr_on_device(),
                                    lda,
                                    dX.ptr_on_device(),
                                    incx,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_dgmm_batched_fn(handle,
                                    side,
                                    M,
                                    N,
                                    dA.ptr_on_device(),
                                    lda,
                                    dX.ptr_on_device(),
                                    incx,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = dgmm_gflop_count<T>(M, N) * number_hot_calls * batch_count / gpu_time_used * 1e6;

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
