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
void testing_gemv_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gemv_fn = FORTRAN ? rocblas_gemv<T, true> : rocblas_gemv<T, false>;

    const rocblas_int M    = 100;
    const rocblas_int N    = 100;
    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int incy = 1;
    T                 alpha;
    T                 beta;
    alpha = beta = 1.0;

    const rocblas_operation transA = rocblas_operation_none;

    rocblas_local_handle handle;

    size_t size_A = lda * size_t(N);
    size_t size_x = N * size_t(incx);
    size_t size_y = M * size_t(incy);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, M, incy);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_fn(handle, transA, M, N, &alpha, nullptr, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_fn(handle, transA, M, N, &alpha, dA, lda, nullptr, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_fn(handle, transA, M, N, &alpha, dA, lda, dx, incx, &beta, nullptr, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_fn(handle, transA, M, N, nullptr, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_fn(handle, transA, M, N, &alpha, dA, lda, dx, incx, nullptr, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_fn(nullptr, transA, M, N, &alpha, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_gemv(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gemv_fn = FORTRAN ? rocblas_gemv<T, true> : rocblas_gemv<T, false>;

    rocblas_int       M       = arg.M;
    rocblas_int       N       = arg.N;
    rocblas_int       lda     = arg.lda;
    rocblas_int       incx    = arg.incx;
    rocblas_int       incy    = arg.incy;
    T                 h_alpha = arg.get_alpha<T>();
    T                 h_beta  = arg.get_beta<T>();
    rocblas_operation transA  = char2rocblas_operation(arg.transA);

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy;
    if(invalid_size || !M || !N)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(
                handle, transA, M, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t size_A = lda * size_t(N);
    size_t size_x, dim_x, abs_incx;
    size_t size_y, dim_y, abs_incy;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    abs_incx = incx >= 0 ? incx : -incx;
    abs_incy = incy >= 0 ? incy : -incy;

    size_x = dim_x * abs_incx;
    size_y = dim_y * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy_1(size_y);
    host_vector<T> hy_2(size_y);
    host_vector<T> hy_gold(size_y);

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy_1(size_y);
    device_vector<T> dy_2(size_y);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, dim_x, abs_incx);

    if(rocblas_isnan(arg.beta))
        rocblas_init_nan<T>(hy_1, 1, dim_y, abs_incy);
    else
        rocblas_init<T>(hy_1, 1, dim_y, abs_incy);

    // copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold = hy_1;
    hy_2    = hy_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemv_fn(
            handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_gemv_fn(handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_gemv<T>(transA, M, N, h_alpha, hA, lda, hx, incx, h_beta, hy_gold, incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemv_gflop_count<T>(transA, M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gemv_fn(handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_gemv_fn(handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = gemv_gflop_count<T>(transA, M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = (1.0 * M * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "M,N,alpha,lda,incx,beta,incy,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            rocblas_cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        rocblas_cout << std::endl;

        rocblas_cout << M << "," << N << "," << h_alpha << "," << lda << "," << incx << ","
                     << h_beta << "," << incy << "," << rocblas_gflops << "," << rocblas_bandwidth
                     << ",";

        if(arg.norm_check)
        {
            rocblas_cout << cblas_gflops << ',';
            rocblas_cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        rocblas_cout << std::endl;
    }
}
