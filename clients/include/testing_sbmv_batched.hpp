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
void testing_sbmv_batched_bad_arg()
{
    rocblas_fill uplo        = rocblas_fill_upper;
    rocblas_int  N           = 100;
    rocblas_int  K           = 2;
    rocblas_int  incx        = 1;
    rocblas_int  incy        = 1;
    rocblas_int  lda         = 100;
    T            alpha       = 0.6;
    T            beta        = 0.6;
    rocblas_int  batch_count = 2;

    rocblas_local_handle handle;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = lda * N;
    size_t size_x   = N * abs_incx * batch_count;
    size_t size_y   = N * abs_incy * batch_count;

    // allocate memory on device
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);
    if(!dA || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        rocblas_sbmv_batched<T>(
            nullptr, uplo, N, K, &alpha, dA, lda, dx, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  rocblas_fill_full,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  dx,
                                                  incx,
                                                  &beta,
                                                  dy,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        rocblas_sbmv_batched<T>(
            handle, uplo, N, K, nullptr, dA, lda, dx, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_sbmv_batched<T>(
            handle, uplo, N, K, &alpha, nullptr, lda, dx, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_sbmv_batched<T>(
            handle, uplo, N, K, &alpha, dA, lda, nullptr, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_sbmv_batched<T>(
            handle, uplo, N, K, &alpha, dA, lda, dx, incx, nullptr, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_sbmv_batched<T>(
            handle, uplo, N, K, &alpha, dA, lda, dx, incx, &beta, nullptr, incy, batch_count),
        rocblas_status_invalid_pointer);
}

template <typename T>
void testing_sbmv_batched(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int lda  = arg.lda;
    rocblas_int K    = arg.K;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    host_vector<T> alpha(1);
    host_vector<T> beta(1);
    alpha[0] = arg.alpha;
    beta[0]  = arg.beta;

    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    rocblas_int  batch_count = arg.batch_count;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_A = size_t(lda) * N;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || lda < 0 || K < 0 || !incx || !incy || batch_count <= 0)
    {
        static const size_t    safe_size = 100;
        device_batch_vector<T> dA(safe_size, 1, 1);
        device_batch_vector<T> dx(safe_size, 1, 1);
        device_batch_vector<T> dy(safe_size, 1, 1);
        if(!dA || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }
        EXPECT_ROCBLAS_STATUS(
            rocblas_sbmv_batched<T>(
                handle, uplo, N, K, alpha, dA, lda, dx, incx, beta, dy, incy, batch_count),
            N < 0 || lda < 0 || K < 0 || !incx || !incy || batch_count < 0
                ? rocblas_status_invalid_size
                : rocblas_status_success);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hy2(N, incy, batch_count);
    host_batch_vector<T> hg(N, incy, batch_count);

    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());
    CHECK_HIP_ERROR(hy2.memcheck());
    CHECK_HIP_ERROR(hg.memcheck());

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double h_error, d_error;

    char char_fill = arg.uplo;

    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init(hA);

    rocblas_init(hx);
    rocblas_init(hy);

    // save a copy in hg which will later get output of CPU BLAS
    hg.copy_from(hy);
    hy2.copy_from(hy);

    if(arg.unit_check || arg.norm_check)
    {
        cpu_time_used = get_time_us();

        // cpu reference
        for(int i = 0; i < batch_count; i++)
        {
            cblas_sbmv<T>(uplo, N, K, alpha[0], hA[i], lda, hx[i], incx, beta[0], hg[i], incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * sbmv_gflop_count<T>(N, K) / cpu_time_used * 1e6;
    }

    // copy data from CPU to device
    dx.transfer_from(hx);
    dy.transfer_from(hy);
    dA.transfer_from(hA);

    if(arg.unit_check || arg.norm_check)
    {

        //
        // rocblas_pointer_mode_host test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                    uplo,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    beta,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        //
        // rocblas_pointer_mode_device test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(d_alpha.transfer_from(alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(beta));

        dy.transfer_from(hy2);

        CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                    uplo,
                                                    N,
                                                    K,
                                                    d_alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    d_beta,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy2.transfer_from(dy));

        if(arg.unit_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                unit_check_general<T>(1, N, abs_incy, hg[i], hy[i]);
                unit_check_general<T>(1, N, abs_incy, hg[i], hy2[i]);
            }
        }

        if(arg.norm_check)
        {
            h_error = norm_check_general<T>('F', 1, N, abs_incy, batch_count, hg, hy);
            d_error = norm_check_general<T>('F', 1, N, abs_incy, batch_count, hg, hy2);
        }
    }

    if(arg.timing)
    {

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                        uplo,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dx.ptr_on_device(),
                                                        incx,
                                                        beta,
                                                        dy.ptr_on_device(),
                                                        incy,
                                                        batch_count));
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                        uplo,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dx.ptr_on_device(),
                                                        incx,
                                                        beta,
                                                        dy.ptr_on_device(),
                                                        incy,
                                                        batch_count));
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * sbmv_gflop_count<T>(N, K) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * sbmv_gbyte_count<T>(N, K) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        std::cout
            << "uplo, N, lda, K, incx, incy, batch_count, rocblas-Gflops, rocblas-GB/s, (us) ";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,(us),norm_error_host_ptr,norm_error_dev_ptr";
        }
        std::cout << std::endl;

        std::cout << arg.uplo << ',' << N << ',' << lda << ',' << K << ',' << incx << "," << incy
                  << "," << batch_count << "," << rocblas_gflops << "," << rocblas_bandwidth << ",("
                  << gpu_time_used << "),";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ",(" << cpu_time_used << ")," << h_error << "," << d_error;
        }
        std::cout << std::endl;
    }
}
