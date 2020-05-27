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

template <typename T, bool CONJ = false>
void testing_ger_bad_arg(const Arguments& arg)
{
    const bool FORTRAN        = arg.fortran;
    auto       rocblas_ger_fn = FORTRAN
                              ? (CONJ ? rocblas_ger<T, true, true> : rocblas_ger<T, false, true>)
                              : (CONJ ? rocblas_ger<T, true, false> : rocblas_ger<T, false, false>);

    rocblas_int M     = 100;
    rocblas_int N     = 100;
    rocblas_int incx  = 1;
    rocblas_int incy  = 1;
    rocblas_int lda   = 100;
    T           alpha = 0.6;

    rocblas_local_handle handle;

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_A   = lda * size_t(N);
    size_t      size_x   = M * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_ger_fn(handle, M, N, &alpha, nullptr, incx, dy, incy, dA_1, lda)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_ger_fn(handle, M, N, &alpha, dx, incx, nullptr, incy, dA_1, lda)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_fn(handle, M, N, &alpha, dx, incx, dy, incy, nullptr, lda)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_fn(nullptr, M, N, &alpha, dx, incx, dy, incy, dA_1, lda)),
                          rocblas_status_invalid_handle);
}

template <typename T, bool CONJ = false>
void testing_ger(const Arguments& arg)
{
    const bool FORTRAN        = arg.fortran;
    auto       rocblas_ger_fn = FORTRAN
                              ? (CONJ ? rocblas_ger<T, true, true> : rocblas_ger<T, false, true>)
                              : (CONJ ? rocblas_ger<T, true, false> : rocblas_ger<T, false, false>);

    rocblas_int M       = arg.M;
    rocblas_int N       = arg.N;
    rocblas_int incx    = arg.incx;
    rocblas_int incy    = arg.incy;
    rocblas_int lda     = arg.lda;
    T           h_alpha = arg.get_alpha<T>();

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy)
    {
        EXPECT_ROCBLAS_STATUS(
            (rocblas_ger_fn(handle, M, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda)),
            rocblas_status_invalid_size);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = lda * N;
    size_t size_x   = M * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_1(size_A);
    host_vector<T> hA_2(size_A);
    host_vector<T> hA_gold(size_A);
    host_vector<T> hx(M * abs_incx);
    host_vector<T> hy(N * abs_incy);

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dA_2(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initial Data on CPU
    rocblas_seedrand();
    if(lda >= M)
    {
        rocblas_init<T>(hA_1, M, N, lda);
    }
    rocblas_init<T>(hx, 1, M, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA_1, hA_1, sizeof(T) * lda * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * M * abs_incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * N * abs_incy, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA_2, hA_2, sizeof(T) * lda * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            (rocblas_ger_fn(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda)));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_ger_fn(handle, M, N, d_alpha, dx, incx, dy, incy, dA_2, lda)));

        // copy output from device to CPU
        hipMemcpy(hA_1, dA_1, sizeof(T) * N * lda, hipMemcpyDeviceToHost);
        hipMemcpy(hA_2, dA_2, sizeof(T) * N * lda, hipMemcpyDeviceToHost);

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_ger<T, CONJ>(M, N, h_alpha, hx, incx, hy, incy, hA_gold, lda);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = ger_gflop_count<T, CONJ>(M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(M, N, lda, hA_gold, hA_1);
                unit_check_general<T>(M, N, lda, hA_gold, hA_2);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(M, N, lda, hA_gold, hA_1, tol);
                near_check_general<T>(M, N, lda, hA_gold, hA_2, tol);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, lda, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', M, N, lda, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_ger_fn(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_ger_fn(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_gflops    = ger_gflop_count<T, CONJ>(M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = ger_gbyte_count<T>(M, N) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "M,N,alpha,incx,incy,lda,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << M << "," << N << "," << h_alpha << "," << incx << "," << incy << "," << lda
                     << "," << rocblas_gflops << "," << rocblas_bandwidth << "," << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        rocblas_cout << std::endl;
    }
}
