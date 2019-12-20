/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "cblas_interface.hpp"
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

template <typename T>
void testing_gbmv_strided_batched_bad_arg(const Arguments& arg)
{
    const rocblas_int M           = 100;
    const rocblas_int N           = 100;
    const rocblas_int KL          = 5;
    const rocblas_int KU          = 5;
    const rocblas_int lda         = 100;
    const rocblas_int incx        = 1;
    const rocblas_int incy        = 1;
    const T           alpha       = 1.0;
    const T           beta        = 1.0;
    const rocblas_int stride_A    = 10000;
    const rocblas_int stride_x    = 100;
    const rocblas_int stride_y    = 100;
    const rocblas_int batch_count = 5;

    const rocblas_operation transA = rocblas_operation_none;

    rocblas_local_handle handle;

    size_t size_A = lda * static_cast<size_t>(N);
    size_t size_x = N * static_cast<size_t>(incx);
    size_t size_y = M * static_cast<size_t>(incy);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    if(!dA || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          nullptr,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dA,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dA,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          nullptr,
                                                          dA,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dA,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(nullptr,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dA,
                                                          lda,
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
}

template <typename T>
void testing_gbmv_strided_batched(const Arguments& arg)
{
    rocblas_int       M           = arg.M;
    rocblas_int       N           = arg.N;
    rocblas_int       KL          = arg.KL;
    rocblas_int       KU          = arg.KU;
    rocblas_int       lda         = arg.lda;
    rocblas_int       incx        = arg.incx;
    rocblas_int       incy        = arg.incy;
    T                 h_alpha     = arg.get_alpha<T>();
    T                 h_beta      = arg.get_beta<T>();
    rocblas_operation transA      = char2rocblas_operation(arg.transA);
    rocblas_int       stride_A    = arg.stride_a;
    rocblas_int       stride_x    = arg.stride_x;
    rocblas_int       stride_y    = arg.stride_y;
    rocblas_int       batch_count = arg.batch_count;

    rocblas_local_handle handle;
    size_t               size_A = lda * static_cast<size_t>(N);
    size_t               size_x, dim_x, abs_incx;
    size_t               size_y, dim_y, abs_incy;

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

    // argument sanity check before allocating invalid memory
    if(M < 0 || N < 0 || lda < KL + KU + 1 || !incx || !incy || KL < 0 || KU < 0 || batch_count < 0)
    {
        static constexpr size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>        dA1(safe_size);
        device_vector<T>        dx1(safe_size);
        device_vector<T>        dy1(safe_size);
        if(!dA1 || !dx1 || !dy1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                              transA,
                                                              M,
                                                              N,
                                                              KL,
                                                              KU,
                                                              &h_alpha,
                                                              dA1,
                                                              lda,
                                                              stride_A,
                                                              dx1,
                                                              incx,
                                                              stride_x,
                                                              &h_beta,
                                                              dy1,
                                                              incy,
                                                              stride_y,
                                                              batch_count),
                              rocblas_status_invalid_size);

        return;
    }

    //quick return
    if(!M || !N || !batch_count)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA1(safe_size);
        device_vector<T>    dx1(safe_size);
        device_vector<T>    dy1(safe_size);
        if(!dA1 || !dx1 || !dy1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched<T>(handle,
                                                              transA,
                                                              M,
                                                              N,
                                                              KL,
                                                              KU,
                                                              &h_alpha,
                                                              dA1,
                                                              lda,
                                                              stride_A,
                                                              dx1,
                                                              incx,
                                                              stride_x,
                                                              &h_beta,
                                                              dy1,
                                                              incy,
                                                              stride_y,
                                                              batch_count),
                              rocblas_status_success);

        return;
    }

    size_A = size_A + static_cast<size_t>(stride_A) * static_cast<size_t>(batch_count - 1);
    size_x = size_x + static_cast<size_t>(stride_x) * static_cast<size_t>(batch_count - 1);
    size_y = size_y + static_cast<size_t>(stride_y) * static_cast<size_t>(batch_count - 1);

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
    if((!dA && size_A) || (!dx && size_x) || ((!dy_1 || !dy_2) && size_y) || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    // Init a lda * N matrix, not M * N
    rocblas_init<T>(hA, lda, N, lda, stride_A, batch_count);
    rocblas_init<T>(hx, 1, dim_x, abs_incx, stride_x, batch_count);

    if(rocblas_isnan(arg.beta))
        rocblas_init_nan<T>(hy_1, 1, dim_y, abs_incy, stride_y, batch_count);
    else
        rocblas_init<T>(hy_1, 1, dim_y, abs_incy, stride_y, batch_count);

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
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_strided_batched<T>(handle,
                                                            transA,
                                                            M,
                                                            N,
                                                            KL,
                                                            KU,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
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
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_strided_batched<T>(handle,
                                                            transA,
                                                            M,
                                                            N,
                                                            KL,
                                                            KU,
                                                            d_alpha,
                                                            dA,
                                                            lda,
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
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_gbmv<T>(transA,
                          M,
                          N,
                          KL,
                          KU,
                          h_alpha,
                          hA + b * stride_A,
                          lda,
                          hx + b * stride_x,
                          incx,
                          h_beta,
                          hy_gold + b * stride_y,
                          incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops
            = batch_count * gbmv_gflop_count<T>(transA, M, N, KL, KU) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, 1, abs_incy, stride_y, hy_gold, hy_1);
            unit_check_general<T>(1, dim_y, 1, abs_incy, stride_y, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>(
                'F', 1, dim_y, abs_incy, stride_y, batch_count, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>(
                'F', 1, dim_y, abs_incy, stride_y, batch_count, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gbmv_strided_batched<T>(handle,
                                            transA,
                                            M,
                                            N,
                                            KL,
                                            KU,
                                            &h_alpha,
                                            dA,
                                            lda,
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
            rocblas_gbmv_strided_batched<T>(handle,
                                            transA,
                                            M,
                                            N,
                                            KL,
                                            KU,
                                            &h_alpha,
                                            dA,
                                            lda,
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

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops
            = batch_count * gbmv_gflop_count<T>(transA, M, N, KL, KU) / gpu_time_used * 1e6;

        rocblas_int k1      = dim_x < KL ? dim_x : KL;
        rocblas_int k2      = dim_x < KU ? dim_x : KU;
        rocblas_int d1      = ((k1 * dim_x) - (k1 * (k1 + 1) / 2));
        rocblas_int d2      = ((k2 * dim_x) - (k2 * (k2 + 1) / 2));
        double      num_els = double(d1 + d2 + dim_x);
        rocblas_bandwidth   = batch_count * (num_els) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout
            << "M,N,KL,KU,alpha,lda,stride_A,incx,stride_x,beta,incy,stride_y,batch_count,rocblas-"
               "Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << M << "," << N << "," << KL << "," << KU << "," << h_alpha << "," << lda << ","
                  << stride_A << "," << incx << "," << stride_x << "," << h_beta << "," << incy
                  << "," << stride_y << "," << batch_count << "," << rocblas_gflops << ","
                  << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        std::cout << std::endl;
    }
}
