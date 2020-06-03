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

template <typename T, bool CONJ>
void testing_ger_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_ger_strided_batched_fn
        = FORTRAN ? (CONJ ? rocblas_ger_strided_batched<T, true, true>
                          : rocblas_ger_strided_batched<T, false, true>)
                  : (CONJ ? rocblas_ger_strided_batched<T, true, false>
                          : rocblas_ger_strided_batched<T, false, false>);

    rocblas_int M           = 100;
    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int lda         = 100;
    T           alpha       = 0.6;
    rocblas_int abs_incx    = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy    = incy >= 0 ? incy : -incy;
    rocblas_int stride_a    = lda * N;
    rocblas_int stride_x    = abs_incx * M;
    rocblas_int stride_y    = abs_incy * N;
    rocblas_int batch_count = 5;

    rocblas_local_handle handle;

    size_t size_A = stride_a * batch_count;
    size_t size_x = stride_x * batch_count;
    size_t size_y = stride_y * batch_count;

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_ger_strided_batched_fn(handle,
                                                          M,
                                                          N,
                                                          &alpha,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          dA_1,
                                                          lda,
                                                          stride_a,
                                                          batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_strided_batched_fn(handle,
                                                          M,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          dA_1,
                                                          lda,
                                                          stride_a,
                                                          batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_strided_batched_fn(handle,
                                                          M,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          nullptr,
                                                          lda,
                                                          stride_a,
                                                          batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_strided_batched_fn(nullptr,
                                                          M,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          dA_1,
                                                          lda,
                                                          stride_a,
                                                          batch_count)),
                          rocblas_status_invalid_handle);
}

template <typename T, bool CONJ>
void testing_ger_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_ger_strided_batched_fn
        = FORTRAN ? (CONJ ? rocblas_ger_strided_batched<T, true, true>
                          : rocblas_ger_strided_batched<T, false, true>)
                  : (CONJ ? rocblas_ger_strided_batched<T, true, false>
                          : rocblas_ger_strided_batched<T, false, false>);

    rocblas_int M           = arg.M;
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int lda         = arg.lda;
    T           h_alpha     = arg.get_alpha<T>();
    rocblas_int stride_x    = arg.stride_x;
    rocblas_int stride_y    = arg.stride_y;
    rocblas_int stride_a    = arg.stride_a;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = size_t(lda) * N;
    size_t size_x   = M * abs_incx;
    size_t size_y   = N * abs_incy;

    // argument check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS((rocblas_ger_strided_batched_fn(handle,
                                                              M,
                                                              N,
                                                              nullptr,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              nullptr,
                                                              incy,
                                                              stride_y,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              batch_count)),
                              invalid_size || batch_count < 0 ? rocblas_status_invalid_size
                                                              : rocblas_status_success);

        return;
    }

    size_A += size_t(stride_a) * size_t(batch_count - 1);
    size_x += size_t(stride_x) * size_t(batch_count - 1);
    size_y += size_t(stride_y) * size_t(batch_count - 1);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_1(size_A);
    host_vector<T> hA_2(size_A);
    host_vector<T> hA_gold(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);

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
        rocblas_init<T>(hA_1, M, N, lda, stride_a, batch_count);
    }
    rocblas_init<T>(hx, 1, M, abs_incx, stride_x, batch_count);
    rocblas_init<T>(hy, 1, N, abs_incy, stride_y, batch_count);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA_1, hA_1, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA_2, hA_2, sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_ger_strided_batched_fn(handle,
                                                            M,
                                                            N,
                                                            &h_alpha,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dy,
                                                            incy,
                                                            stride_y,
                                                            dA_1,
                                                            lda,
                                                            stride_a,
                                                            batch_count)));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_ger_strided_batched_fn(handle,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dy,
                                                            incy,
                                                            stride_y,
                                                            dA_2,
                                                            lda,
                                                            stride_a,
                                                            batch_count)));

        // copy output from device to CPU
        hipMemcpy(hA_1, dA_1, sizeof(T) * size_A, hipMemcpyDeviceToHost);
        hipMemcpy(hA_2, dA_2, sizeof(T) * size_A, hipMemcpyDeviceToHost);

        // CPU BLAS
        cpu_time_used = get_time_us();

        for(int b = 0; b < batch_count; ++b)
        {
            cblas_ger<T, CONJ>(M,
                               N,
                               h_alpha,
                               hx + b * stride_x,
                               incx,
                               hy + b * stride_y,
                               incy,
                               hA_gold + b * stride_a,
                               lda);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * ger_gflop_count<T, CONJ>(M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(M, N, lda, stride_a, hA_gold, hA_1, batch_count);
                unit_check_general<T>(M, N, lda, stride_a, hA_gold, hA_2, batch_count);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(M, N, lda, stride_a, hA_gold, hA_1, batch_count, tol);
                near_check_general<T>(M, N, lda, stride_a, hA_gold, hA_2, batch_count, tol);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', M, N, lda, stride_a, hA_gold, hA_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', M, N, lda, stride_a, hA_gold, hA_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_ger_strided_batched_fn(handle,
                                           M,
                                           N,
                                           &h_alpha,
                                           dx,
                                           incx,
                                           stride_x,
                                           dy,
                                           incy,
                                           stride_y,
                                           dA_1,
                                           lda,
                                           stride_a,
                                           batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_ger_strided_batched_fn(handle,
                                           M,
                                           N,
                                           &h_alpha,
                                           dx,
                                           incx,
                                           stride_x,
                                           dy,
                                           incy,
                                           stride_y,
                                           dA_1,
                                           lda,
                                           stride_a,
                                           batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * ger_gflop_count<T, CONJ>(M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * ger_gbyte_count<T>(M, N) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "M,N,alpha,incx,stride_x,incy,stride_y,lda,stride_a,batch_count,rocblas-"
                        "Gflops,rocblas-GB/s,rocblas-us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << M << "," << N << "," << h_alpha << "," << incx << "," << stride_x << ","
                     << incy << "," << stride_y << "," << lda << "," << stride_a << ","
                     << batch_count << "," << rocblas_gflops << "," << rocblas_bandwidth << ","
                     << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        rocblas_cout << std::endl;
    }
}
