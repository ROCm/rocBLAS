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
void testing_hpmv_strided_batched_bad_arg(const Arguments& arg)
{
    const rocblas_int    N           = 100;
    const rocblas_int    incx        = 1;
    const rocblas_int    incy        = 1;
    const rocblas_int    batch_count = 5;
    const rocblas_stride stride_A    = 10000;
    const rocblas_stride stride_x    = 100;
    const rocblas_stride stride_y    = 100;
    T                    alpha       = 1.0;
    T                    beta        = 1.0;

    const rocblas_fill   uplo = rocblas_fill_upper;
    rocblas_local_handle handle;

    size_t size_A = size_t(N);
    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          nullptr,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          nullptr,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(nullptr,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          nullptr,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          0),
                          rocblas_status_success);
}

template <typename T>
void testing_hpmv_strided_batched(const Arguments& arg)
{
    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    T              h_alpha     = arg.get_alpha<T>();
    T              h_beta      = arg.get_beta<T>();
    rocblas_fill   uplo        = char2rocblas_fill(arg.uplo);
    rocblas_stride stride_A    = arg.stride_a;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || !incx || !incy || batch_count <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA1(safe_size);
        device_vector<T>    dx1(safe_size);
        device_vector<T>    dy1(safe_size);
        CHECK_HIP_ERROR(dA1.memcheck());
        CHECK_HIP_ERROR(dx1.memcheck());
        CHECK_HIP_ERROR(dy1.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched<T>(handle,
                                                              uplo,
                                                              N,
                                                              &h_alpha,
                                                              dA1,
                                                              stride_A,
                                                              dx1,
                                                              incx,
                                                              stride_x,
                                                              &h_beta,
                                                              dy1,
                                                              incy,
                                                              stride_y,
                                                              batch_count),
                              (N < 0 || !incx || !incy || batch_count < 0)
                                  ? rocblas_status_invalid_size
                                  : rocblas_status_success);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = (size_t(N * (N + 1)) / 2);
    size_t size_x   = N;
    size_t size_y   = N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_vector<T> hA(size_A, 1, stride_A, batch_count);
    host_strided_batch_vector<T> hx(size_x, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy_1(size_y, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_2(size_y, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_gold(size_y, incy, stride_y, batch_count);
    host_vector<T>               halpha(1);
    host_vector<T>               hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());
    CHECK_HIP_ERROR(halpha.memcheck());
    CHECK_HIP_ERROR(hbeta.memcheck());

    device_strided_batch_vector<T> dA(size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(size_x, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy_1(size_y, incy, stride_y, batch_count);
    device_strided_batch_vector<T> dy_2(size_y, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy_1.memcheck());
    CHECK_HIP_ERROR(dy_2.memcheck());
    CHECK_HIP_ERROR(d_alpha.memcheck());
    CHECK_HIP_ERROR(d_beta.memcheck());

    // Initial Data on CPU
    rocblas_init(hA, true);
    rocblas_init(hx, false);

    if(rocblas_isnan(arg.beta))
        rocblas_init_nan(hy_1, false);
    else
        rocblas_init(hy_1, false);

    hy_gold.copy_from(hy_1);
    hy_2.copy_from(hy_1);

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
        CHECK_ROCBLAS_ERROR(rocblas_hpmv_strided_batched<T>(handle,
                                                            uplo,
                                                            N,
                                                            &h_alpha,
                                                            dA,
                                                            stride_A,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            &h_beta,
                                                            dy_1,
                                                            incy,
                                                            stride_y,
                                                            batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_hpmv_strided_batched<T>(handle,
                                                            uplo,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            stride_A,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            d_beta,
                                                            dy_2,
                                                            incy,
                                                            stride_y,
                                                            batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        // CPU BLAS
        cpu_time_used = get_time_us();

        for(int b = 0; b < batch_count; b++)
            cblas_hpmv<T>(uplo,
                          N,
                          h_alpha,
                          hA + b * stride_A,
                          hx + b * stride_x,
                          incx,
                          h_beta,
                          hy_gold + b * stride_y,
                          incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * hpmv_gflop_count<T>(N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_gold, hy_1);
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, batch_count, hy_gold, hy_1);
            rocblas_error_2
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, batch_count, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hpmv_strided_batched<T>(handle,
                                            uplo,
                                            N,
                                            &h_alpha,
                                            dA,
                                            stride_A,
                                            dx,
                                            incx,
                                            stride_x,
                                            &h_beta,
                                            dy_1,
                                            incy,
                                            stride_y,
                                            batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_hpmv_strided_batched<T>(handle,
                                            uplo,
                                            N,
                                            &h_alpha,
                                            dA,
                                            stride_A,
                                            dx,
                                            incx,
                                            stride_x,
                                            &h_beta,
                                            dy_1,
                                            incy,
                                            stride_y,
                                            batch_count);
        }

        gpu_time_used  = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops = batch_count * hpmv_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth
            = batch_count * (((N * (N + 1.0)) / 2.0) + 3.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,alpha,stride_A,incx,stride_x,beta,incy,stride_y,batch_count,rocblas-"
                     "Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << N << "," << h_alpha << "," << stride_A << "," << incx << "," << stride_x << ","
                  << h_beta << "," << incy << "," << stride_y << "," << batch_count << ","
                  << rocblas_gflops << "," << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        std::cout << std::endl;
    }
}
