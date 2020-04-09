/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
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
void testing_her_batched_bad_arg()
{
    using U = rocblas_real_t<T>;

    rocblas_fill         uplo        = rocblas_fill_upper;
    rocblas_int          N           = 100;
    rocblas_int          lda         = 100;
    rocblas_int          incx        = 1;
    U                    alpha       = 0.6;
    rocblas_int          batch_count = 2;
    rocblas_local_handle handle;

    size_t size_A = size_t(N) * lda;

    // allocate memory on device
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dA_1(size_A, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_her_batched<T, U>)(handle,
                                                      rocblas_fill_full,
                                                      N,
                                                      &alpha,
                                                      dx,
                                                      incx,
                                                      dA_1,
                                                      lda,
                                                      batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her_batched<T, U>)(handle, uplo, N, &alpha, nullptr, incx, dA_1, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her_batched<T, U>)(handle, uplo, N, &alpha, dx, incx, nullptr, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her_batched<T, U>)(nullptr, uplo, N, &alpha, dx, incx, dA_1, lda, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_her_batched(const Arguments& arg)
{
    using U = rocblas_real_t<T>;

    rocblas_int  N           = arg.N;
    rocblas_int  lda         = arg.lda;
    rocblas_int  incx        = arg.incx;
    U            h_alpha     = arg.get_alpha<U>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    rocblas_int  batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(N <= 0 || !incx || lda < 1 || lda < N || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS((rocblas_her_batched<T, U>)(handle,
                                                          uplo,
                                                          N,
                                                          nullptr,
                                                          nullptr,
                                                          incx,
                                                          nullptr,
                                                          lda,
                                                          batch_count),
                              N < 0 || !incx || lda < 1 || lda < N || batch_count < 0
                                  ? rocblas_status_invalid_size
                                  : rocblas_status_success);
        return;
    }

    size_t size_A = size_t(N) * lda;
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA_1(size_A, 1, batch_count);
    host_batch_vector<T> hA_2(size_A, 1, batch_count);
    host_batch_vector<T> hA_gold(size_A, 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<U>       halpha(1);
    CHECK_HIP_ERROR(hA_1.memcheck());
    CHECK_HIP_ERROR(hA_2.memcheck());
    CHECK_HIP_ERROR(hA_gold.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(halpha.memcheck());

    halpha[0] = h_alpha;

    // allocate memory on device
    device_batch_vector<T> dA_1(size_A, 1, batch_count);
    device_batch_vector<T> dA_2(size_A, 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<U>       d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initial Data on CPU
    rocblas_init(hA_1, true);
    rocblas_init(hx, false);

    hA_2.copy_from(hA_1);
    hA_gold.copy_from(hA_1);
    CHECK_HIP_ERROR(dA_1.transfer_from(hA_1));
    CHECK_HIP_ERROR(dA_2.transfer_from(hA_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_her_batched<T, U>)(handle,
                                                        uplo,
                                                        N,
                                                        &h_alpha,
                                                        dx.ptr_on_device(),
                                                        incx,
                                                        dA_1.ptr_on_device(),
                                                        lda,
                                                        batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_her_batched<T, U>)(handle,
                                                        uplo,
                                                        N,
                                                        d_alpha,
                                                        dx.ptr_on_device(),
                                                        incx,
                                                        dA_2.ptr_on_device(),
                                                        lda,
                                                        batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_1));
        CHECK_HIP_ERROR(hA_2.transfer_from(dA_2));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_her<T, U>(uplo, N, h_alpha, hx[i], incx, hA_gold[i], lda);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * her_gflop_count<T>(N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            const double tol = N * sum_error_tolerance<T>;
            near_check_general<T, T>(N, N, batch_count, lda, hA_gold, hA_1, tol);
            near_check_general<T, T>(N, N, batch_count, lda, hA_gold, hA_2, tol);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', N, N, lda, batch_count, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', N, N, lda, batch_count, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_her_batched<T, U>(handle,
                                      uplo,
                                      N,
                                      &h_alpha,
                                      dx.ptr_on_device(),
                                      incx,
                                      dA_1.ptr_on_device(),
                                      lda,
                                      batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_her_batched<T, U>(handle,
                                      uplo,
                                      N,
                                      &h_alpha,
                                      dx.ptr_on_device(),
                                      incx,
                                      dA_1.ptr_on_device(),
                                      lda,
                                      batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * her_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * her_gbyte_count<T>(N) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,alpha,lda,incx,batch_count,rocblas-Gflops,rocblas-GB/s";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << N << "," << h_alpha << "," << lda << "," << incx << "," << batch_count << ","
                  << rocblas_gflops << "," << rocblas_bandwidth;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
