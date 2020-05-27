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
void testing_gbmv_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gbmv_fn = FORTRAN ? rocblas_gbmv<T, true> : rocblas_gbmv<T, false>;

    const rocblas_int M    = 100;
    const rocblas_int N    = 100;
    const rocblas_int KL   = 5;
    const rocblas_int KU   = 5;
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
    dA.transfer_from(hA);
    dx.transfer_from(hx);
    dy.transfer_from(hy);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gbmv_fn(
            handle, transA, M, N, KL, KU, &alpha, nullptr, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gbmv_fn(
            handle, transA, M, N, KL, KU, &alpha, dA, lda, nullptr, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gbmv_fn(
            handle, transA, M, N, KL, KU, &alpha, dA, lda, dx, incx, &beta, nullptr, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gbmv_fn(handle, transA, M, N, KL, KU, nullptr, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gbmv_fn(handle, transA, M, N, KL, KU, &alpha, dA, lda, dx, incx, nullptr, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gbmv_fn(nullptr, transA, M, N, KL, KU, &alpha, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_handle);

    // TODO: See rocblas_gbmv.cpp comment on alpha==0 && beta==1 case.
    // CHECK_ROCBLAS_ERROR(rocblas_gbmv_fn(handle, transA, 1, 1, 1, 1, &alpha, nullptr, 10, nullptr, 1, &beta, nullptr, 1));
}

template <typename T>
void testing_gbmv(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gbmv_fn = FORTRAN ? rocblas_gbmv<T, true> : rocblas_gbmv<T, false>;

    rocblas_int       M       = arg.M;
    rocblas_int       N       = arg.N;
    rocblas_int       KL      = arg.KL;
    rocblas_int       KU      = arg.KU;
    rocblas_int       lda     = arg.lda;
    rocblas_int       incx    = arg.incx;
    rocblas_int       incy    = arg.incy;
    T                 h_alpha = arg.get_alpha<T>();
    T                 h_beta  = arg.get_beta<T>();
    rocblas_operation transA  = char2rocblas_operation(arg.transA);

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < KL + KU + 1 || !incx || !incy || KL < 0 || KU < 0;
    if(invalid_size || !M || !N)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gbmv_fn(handle,
                                              transA,
                                              M,
                                              N,
                                              KL,
                                              KU,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              incx,
                                              nullptr,
                                              nullptr,
                                              incy),
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
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    // Init a lda * N matrix, not M * N
    rocblas_init<T>(hA, lda, N, lda);
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
    dA.transfer_from(hA);
    dx.transfer_from(hx);
    dy_1.transfer_from(hy_1);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        dy_1.transfer_from(hy_1);
        dy_2.transfer_from(hy_2);
        d_alpha.transfer_from(halpha);
        d_beta.transfer_from(hbeta);

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_fn(
            handle, transA, M, N, KL, KU, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_fn(
            handle, transA, M, N, KL, KU, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        // copy output from device to CPU
        hy_1.transfer_from(dy_1);
        hy_2.transfer_from(dy_2);

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_gbmv<T>(transA, M, N, KL, KU, h_alpha, hA, lda, hx, incx, h_beta, hy_gold, incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gbmv_gflop_count<T>(transA, M, N, KL, KU) / cpu_time_used * 1e6;

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
            rocblas_gbmv_fn(
                handle, transA, M, N, KL, KU, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_gbmv_fn(
                handle, transA, M, N, KL, KU, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used  = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops = gbmv_gflop_count<T>(transA, M, N, KL, KU) / gpu_time_used * 1e6;

        rocblas_int k1      = dim_x < KL ? dim_x : KL;
        rocblas_int k2      = dim_x < KU ? dim_x : KU;
        rocblas_int d1      = ((k1 * dim_x) - (k1 * (k1 + 1) / 2));
        rocblas_int d2      = ((k2 * dim_x) - (k2 * (k2 + 1) / 2));
        double      num_els = double(d1 + d2 + dim_x);
        rocblas_bandwidth   = (num_els) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "M,N,KL,KU,alpha,lda,incx,beta,incy,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            rocblas_cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        rocblas_cout << std::endl;

        rocblas_cout << M << "," << N << "," << KL << "," << KU << "," << h_alpha << "," << lda
                     << "," << incx << "," << h_beta << "," << incy << "," << rocblas_gflops << ","
                     << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            rocblas_cout << cblas_gflops << ',';
            rocblas_cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        rocblas_cout << std::endl;
    }
}
