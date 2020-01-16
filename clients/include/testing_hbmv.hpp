/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
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

template <typename T>
void testing_hbmv_bad_arg(const Arguments& arg)
{
    const rocblas_int N    = 100;
    const rocblas_int K    = 10;
    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int incy = 1;
    T                 alpha;
    T                 beta;
    alpha = beta = 1.0;

    const rocblas_fill   uplo = rocblas_fill_upper;
    rocblas_local_handle handle;

    size_t size_A = lda * size_t(N);
    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_hbmv<T>(handle, uplo, N, K, &alpha, nullptr, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hbmv<T>(handle, uplo, N, K, &alpha, dA, lda, nullptr, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hbmv<T>(handle, uplo, N, K, &alpha, dA, lda, dx, incx, &beta, nullptr, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hbmv<T>(handle, uplo, N, K, nullptr, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hbmv<T>(handle, uplo, N, K, &alpha, dA, lda, dx, incx, nullptr, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hbmv<T>(nullptr, uplo, N, K, &alpha, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_hbmv(const Arguments& arg)
{
    rocblas_int  N       = arg.N;
    rocblas_int  K       = arg.K;
    rocblas_int  lda     = arg.lda;
    rocblas_int  incx    = arg.incx;
    rocblas_int  incy    = arg.incy;
    T            h_alpha = arg.get_alpha<T>();
    T            h_beta  = arg.get_beta<T>();
    rocblas_fill uplo    = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N < 0 || K < 0 || lda <= K || !incx || !incy)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA1(safe_size);
        device_vector<T>    dx1(safe_size);
        device_vector<T>    dy1(safe_size);
        CHECK_HIP_ERROR(dA1.memcheck());
        CHECK_HIP_ERROR(dx1.memcheck());
        CHECK_HIP_ERROR(dy1.memcheck());

        EXPECT_ROCBLAS_STATUS(
            rocblas_hbmv<T>(handle, uplo, N, K, &h_alpha, dA1, lda, dx1, incx, &h_beta, dy1, incy),
            rocblas_status_invalid_size);

        return;
    }

    size_t size_A   = lda * size_t(N);
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy_1(size_y);
    host_vector<T> hy_2(size_y);
    host_vector<T> hy_gold(size_y);
    host_vector<T> halpha(1);
    host_vector<T> hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy_1(size_y);
    device_vector<T> dy_2(size_y);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy_1.memcheck());
    CHECK_HIP_ERROR(dy_2.memcheck());
    CHECK_HIP_ERROR(d_alpha.memcheck());
    CHECK_HIP_ERROR(d_beta.memcheck());

    // Initial Data on CPU
    rocblas_init(hA, true);
    rocblas_init<T>(hx, 1, N, abs_incx);

    if(rocblas_isnan(arg.beta))
        rocblas_init_nan<T>(hy_1, 1, N, abs_incy);
    else
        rocblas_init<T>(hy_1, 1, N, abs_incy);

    // copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold = hy_1;
    hy_2    = hy_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy_1.transfer_from(hy_1));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(dy_1.transfer_from(hy_1));
        CHECK_HIP_ERROR(dy_2.transfer_from(hy_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            rocblas_hbmv<T>(handle, uplo, N, K, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_hbmv<T>(handle, uplo, N, K, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_hbmv<T>(uplo, N, K, h_alpha, hA, lda, hx, incx, h_beta, hy_gold, incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = hbmv_gflop_count<T>(N, K) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hbmv<T>(handle, uplo, N, K, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_hbmv<T>(handle, uplo, N, K, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used  = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops = hbmv_gflop_count<T>(N, K) / gpu_time_used * 1e6;
        rocblas_int k1 = K < N ? K : N;
        rocblas_bandwidth
            = (N * k1 - ((k1 * (k1 + 1)) / 2.0) + 3 * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,K,alpha,lda,incx,beta,incy,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << N << "," << K << "," << h_alpha << "," << lda << "," << incx << "," << h_beta
                  << "," << incy << "," << rocblas_gflops << "," << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        std::cout << std::endl;
    }
}
