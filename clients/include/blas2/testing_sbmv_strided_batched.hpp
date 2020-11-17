/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
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
void testing_sbmv_strided_batched_bad_arg(const Arguments& arg)
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

    rocblas_local_handle handle{arg};

    size_t         abs_incx = incx >= 0 ? incx : -incx;
    size_t         abs_incy = incy >= 0 ? incy : -incy;
    size_t         size_A   = lda * N;
    rocblas_stride strideA  = size_A;
    rocblas_stride stridex  = N * abs_incx;
    rocblas_stride stridey  = N * abs_incy;

    // allocate memory on device
    static const size_t safe_size = 100;
    device_vector<T>    dA(safe_size);
    device_vector<T>    dx(safe_size);
    device_vector<T>    dy(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(nullptr,
                                                          uplo,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                          rocblas_fill_full,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          incx,
                                                          stridex,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          nullptr,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                          uplo,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          &beta,
                                                          nullptr,
                                                          incy,
                                                          stridey,
                                                          batch_count),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_sbmv_strided_batched(const Arguments& arg)
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

    rocblas_stride strideA = arg.stride_a;
    size_t         size_A  = size_t(lda) * N;
    rocblas_stride stridex = arg.stride_x;
    rocblas_stride stridey = arg.stride_y;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || lda < K + 1 || K < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_sbmv_strided_batched<T>(handle,
                                                              uplo,
                                                              N,
                                                              K,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              strideA,
                                                              nullptr,
                                                              incx,
                                                              stridex,
                                                              nullptr,
                                                              nullptr,
                                                              incy,
                                                              stridey,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    host_strided_batch_vector<T> hA(size_A, 1, strideA, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stridex, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stridey, batch_count);
    host_strided_batch_vector<T> hy2(N, incy, stridey, batch_count);
    host_strided_batch_vector<T> hg(N, incy, stridey, batch_count); // gold standard

    double gpu_time_used, cpu_time_used;
    double h_error, d_error;

    device_strided_batch_vector<T> dA(size_A, 1, strideA, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stridex, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stridey, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA);

    rocblas_init<T>(hx);
    rocblas_init<T>(hy);

    // make copy in hg which will later be used with CPU BLAS
    hg.copy_from(hy);
    hy2.copy_from(hy); // device memory re-test

    if(arg.unit_check || arg.norm_check)
    {
        // cpu reference
        cpu_time_used = get_time_us_no_sync();

        for(int i = 0; i < batch_count; i++)
        {
            cblas_sbmv<T>(uplo, N, K, alpha[0], hA[i], lda, hx[i], incx, beta[0], hg[i], incy);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;
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

        CHECK_ROCBLAS_ERROR(rocblas_sbmv_strided_batched<T>(handle,
                                                            uplo,
                                                            N,
                                                            K,
                                                            alpha,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            dx,
                                                            incx,
                                                            stridex,
                                                            beta,
                                                            dy,
                                                            incy,
                                                            stridey,
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

        CHECK_ROCBLAS_ERROR(rocblas_sbmv_strided_batched<T>(handle,
                                                            uplo,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            dx,
                                                            incx,
                                                            stridex,
                                                            d_beta,
                                                            dy,
                                                            incy,
                                                            stridey,
                                                            batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy2.transfer_from(dy));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, stridey, hg, hy, batch_count);
            unit_check_general<T>(1, N, abs_incy, stridey, hg, hy2, batch_count);
        }

        if(arg.norm_check)
        {
            h_error = norm_check_general<T>('F', 1, N, abs_incy, stridey, hg, hy, batch_count);
            d_error = norm_check_general<T>('F', 1, N, abs_incy, stridey, hg, hy2, batch_count);
        }
    }

    if(arg.timing)
    {

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sbmv_strided_batched<T>(handle,
                                                                uplo,
                                                                N,
                                                                K,
                                                                alpha,
                                                                dA,
                                                                lda,
                                                                strideA,
                                                                dx,
                                                                incx,
                                                                stridex,
                                                                beta,
                                                                dy,
                                                                incy,
                                                                stridey,
                                                                batch_count));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sbmv_strided_batched<T>(handle,
                                                                uplo,
                                                                N,
                                                                K,
                                                                alpha,
                                                                dA,
                                                                lda,
                                                                strideA,
                                                                dx,
                                                                incx,
                                                                stridex,
                                                                beta,
                                                                dy,
                                                                incy,
                                                                stridey,
                                                                batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_beta,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         sbmv_gflop_count<T>(N, K),
                         sbmv_gbyte_count<T>(N, K),
                         cpu_time_used,
                         h_error,
                         d_error);
    }
}
