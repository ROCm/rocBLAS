/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
void testing_spmv_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_spmv_fn = FORTRAN ? rocblas_spmv<T, true> : rocblas_spmv<T, false>;

    rocblas_fill         uplo  = rocblas_fill_upper;
    rocblas_int          N     = 100;
    rocblas_int          incx  = 1;
    rocblas_int          incy  = 1;
    T                    alpha = 0.6;
    T                    beta  = 0.6;
    rocblas_local_handle handle;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = N * N;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_spmv_fn(nullptr, uplo, N, &alpha, dA, dx, incx, &beta, dy, incy),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, rocblas_fill_full, N, &alpha, dA, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_spmv_fn(handle, uplo, N, nullptr, dA, dx, incx, &beta, dy, incy),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, nullptr, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, dA, nullptr, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_spmv_fn(handle, uplo, N, &alpha, dA, dx, incx, nullptr, dy, incy),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, dA, dx, incx, &beta, nullptr, incy),
        rocblas_status_invalid_pointer);
}

template <typename T>
void testing_spmv(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_spmv_fn = FORTRAN ? rocblas_spmv<T, true> : rocblas_spmv<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    host_vector<T> alpha(1);
    host_vector<T> beta(1);
    alpha[0] = arg.alpha;
    beta[0]  = arg.beta;

    rocblas_fill uplo = char2rocblas_fill(arg.uplo);

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_A = tri_count(N);
    size_t size_X = size_t(N) * abs_incx;
    size_t size_Y = size_t(N) * abs_incy;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || !incy;
    if(invalid_size || !N)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_spmv_fn(
                handle, uplo, N, nullptr, nullptr, nullptr, incx, nullptr, nullptr, incy),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    host_vector<T> hA(size_A);
    host_vector<T> hx(size_X);
    host_vector<T> hy(size_Y);
    host_vector<T> hy2(size_Y);
    host_vector<T> hg(size_Y); // gold standard

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double h_error, d_error;

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_X);
    device_vector<T> dy(size_Y);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA);
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    // make copy in hg which will later be used with CPU BLAS
    hg  = hy;
    hy2 = hy; // device memory re-test

    if(arg.unit_check || arg.norm_check)
    {
        // cpu ref
        cpu_time_used = get_time_us();

        // cpu reference
        cblas_spmv<T>(uplo, N, alpha[0], hA, hx, incx, beta[0], hg, incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = spmv_gflop_count<T>(N) / cpu_time_used * 1e6;
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

        CHECK_ROCBLAS_ERROR(rocblas_spmv_fn(handle, uplo, N, alpha, dA, dx, incx, beta, dy, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        //
        // rocblas_pointer_mode_device test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(d_alpha.transfer_from(alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(beta));

        dy.transfer_from(hy2);

        CHECK_ROCBLAS_ERROR(
            rocblas_spmv_fn(handle, uplo, N, d_alpha, dA, dx, incx, d_beta, dy, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy2.transfer_from(dy));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hg, hy);
            unit_check_general<T>(1, N, abs_incy, hg, hy2);
        }

        if(arg.norm_check)
        {
            h_error = norm_check_general<T>('F', 1, N, abs_incy, hg, hy);
            d_error = norm_check_general<T>('F', 1, N, abs_incy, hg, hy2);
        }
    }

    if(arg.timing)
    {

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(
                rocblas_spmv_fn(handle, uplo, N, alpha, dA, dx, incx, beta, dy, incy));
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(
                rocblas_spmv_fn(handle, uplo, N, alpha, dA, dx, incx, beta, dy, incy));
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = spmv_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = spmv_gbyte_count<T>(N) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "uplo, N, incx, incy, rocblas-Gflops, rocblas-GB/s, (us) ";
        if(arg.norm_check)
        {
            rocblas_cout << "CPU-Gflops (us),norm_error_host_ptr,norm_error_dev_ptr";
        }
        rocblas_cout << std::endl;

        rocblas_cout << arg.uplo << ',' << N << ',' << incx << "," << incy << "," << rocblas_gflops
                     << "," << rocblas_bandwidth << ",(" << gpu_time_used << "),";

        if(arg.norm_check)
        {
            rocblas_cout << cblas_gflops << "(" << cpu_time_used << "),";
            rocblas_cout << h_error << "," << d_error;
        }

        rocblas_cout << std::endl;
    }
}
