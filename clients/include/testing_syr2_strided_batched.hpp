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
void testing_syr2_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_syr2_strided_batched_fn
        = FORTRAN ? rocblas_syr2_strided_batched<T, true> : rocblas_syr2_strided_batched<T, false>;

    rocblas_fill uplo        = rocblas_fill_upper;
    rocblas_int  N           = 100;
    rocblas_int  incx        = 1;
    rocblas_int  incy        = 1;
    rocblas_int  lda         = 100;
    T            alpha       = 0.6;
    rocblas_int  batch_count = 5;

    rocblas_local_handle handle;

    size_t size_A = size_t(lda) * N;

    rocblas_stride stride_x = incx * N;
    rocblas_stride stride_y = incy * N;
    rocblas_stride stride_A = size_A;

    // allocate memory on device
    device_strided_batch_vector<T> dA_1(size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(handle,
                                                          rocblas_fill_full,
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
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          nullptr,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          dA_1,
                                                          lda,
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(handle,
                                                          uplo,
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
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(handle,
                                                          uplo,
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
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(handle,
                                                          uplo,
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
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(nullptr,
                                                          uplo,
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
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_syr2_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_syr2_strided_batched_fn
        = FORTRAN ? rocblas_syr2_strided_batched<T, true> : rocblas_syr2_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    rocblas_int    lda         = arg.lda;
    T              h_alpha     = arg.get_alpha<T>();
    rocblas_fill   uplo        = char2rocblas_fill(arg.uplo);
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    rocblas_stride stride_A    = arg.stride_a;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_syr2_strided_batched_fn(handle,
                                                              uplo,
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
                                                              stride_A,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t size_A = size_t(lda) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_vector<T> hA_1(size_A, 1, stride_A, batch_count);
    host_strided_batch_vector<T> hA_2(size_A, 1, stride_A, batch_count);
    host_strided_batch_vector<T> hA_gold(size_A, 1, stride_A, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stride_y, batch_count);
    host_vector<T>               halpha(1);
    halpha[0] = h_alpha;

    // allocate memory on device
    device_strided_batch_vector<T> dA_1(size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dA_2(size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);
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
    rocblas_init(hA_1, true);
    rocblas_init(hx, false);
    rocblas_init(hy, false);

    hA_2.copy_from(hA_1);
    hA_gold.copy_from(hA_1);

    CHECK_HIP_ERROR(dA_1.transfer_from(hA_1));
    CHECK_HIP_ERROR(dA_2.transfer_from(hA_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_syr2_strided_batched_fn(handle,
                                                            uplo,
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
                                                            stride_A,
                                                            batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_syr2_strided_batched_fn(handle,
                                                            uplo,
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
                                                            stride_A,
                                                            batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_1));
        CHECK_HIP_ERROR(hA_2.transfer_from(dA_2));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_syr2<T>(uplo,
                          N,
                          h_alpha,
                          hx + i * stride_x,
                          incx,
                          hy + i * stride_y,
                          incy,
                          hA_gold + i * stride_A,
                          lda);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * syr2_gflop_count<T>(N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, lda, stride_A, hA_gold, hA_1, batch_count);
            unit_check_general<T>(N, N, lda, stride_A, hA_gold, hA_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', N, N, lda, stride_A, hA_gold, hA_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', N, N, lda, stride_A, hA_gold, hA_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_syr2_strided_batched_fn(handle,
                                            uplo,
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
                                            stride_A,
                                            batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_syr2_strided_batched_fn(handle,
                                            uplo,
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
                                            stride_A,
                                            batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * syr2_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * syr2_gflop_count<T>(N) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "N,alpha,incx,stridex,incy,stridey,lda,strideA,batch_count,rocblas-Gflops,"
                        "rocblas-GB/s";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << N << "," << h_alpha << "," << incx << "," << stride_x << "," << incy << ","
                     << stride_y << "," << lda << "," << stride_A << "," << batch_count << ","
                     << rocblas_gflops << "," << rocblas_bandwidth;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        rocblas_cout << std::endl;
    }
}
