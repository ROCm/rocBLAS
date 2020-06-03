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
void testing_symv_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_symv_strided_batched_fn
        = FORTRAN ? rocblas_symv_strided_batched<T, true> : rocblas_symv_strided_batched<T, false>;

    rocblas_fill uplo        = rocblas_fill_upper;
    rocblas_int  N           = 100;
    rocblas_int  incx        = 1;
    rocblas_int  incy        = 1;
    rocblas_int  lda         = 100;
    T            alpha       = 0.6;
    T            beta        = 0.6;
    rocblas_int  batch_count = 2;

    rocblas_local_handle handle;

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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(nullptr,
                                                          uplo,
                                                          N,
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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                          rocblas_fill_full,
                                                          N,
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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
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

    EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
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
void testing_symv_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_symv_strided_batched_fn
        = FORTRAN ? rocblas_symv_strided_batched<T, true> : rocblas_symv_strided_batched<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int lda  = arg.lda;
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

    rocblas_stride strideA = size_t(lda) * N;
    size_t         size_A  = strideA * batch_count;
    rocblas_stride stridex = size_t(N) * abs_incx;
    rocblas_stride stridey = size_t(N) * abs_incy;
    size_t         size_X  = stridex * batch_count;
    size_t         size_Y  = stridey * batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || lda < 1 || lda < N || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_symv_strided_batched_fn(handle,
                                                              uplo,
                                                              N,
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

    host_vector<T> hA(size_A);
    host_vector<T> hx(size_X);
    host_vector<T> hy(size_Y);
    host_vector<T> hy2(size_Y);
    host_vector<T> hg(size_Y); // gold standard

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double h_error, d_error;

    char char_fill = arg.uplo;

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
        // cpu reference
        cpu_time_used = get_time_us();

        for(int i = 0; i < batch_count; i++)
        {
            cblas_symv<T>(uplo,
                          N,
                          alpha[0],
                          hA + i * strideA,
                          lda,
                          hx + i * stridex,
                          incx,
                          beta[0],
                          hg + i * stridey,
                          incy);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * symv_gflop_count<T>(N) / cpu_time_used * 1e6;
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

        CHECK_ROCBLAS_ERROR(rocblas_symv_strided_batched_fn(handle,
                                                            uplo,
                                                            N,
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

        CHECK_ROCBLAS_ERROR(rocblas_symv_strided_batched_fn(handle,
                                                            uplo,
                                                            N,
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
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(1, N, abs_incy, stridey, hg, hy, batch_count);
                unit_check_general<T>(1, N, abs_incy, stridey, hg, hy2, batch_count);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(1, N, abs_incy, stridey, hg, hy, batch_count, tol);
                near_check_general<T>(1, N, abs_incy, stridey, hg, hy2, batch_count, tol);
            }
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
            CHECK_ROCBLAS_ERROR(rocblas_symv_strided_batched_fn(handle,
                                                                uplo,
                                                                N,
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

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_symv_strided_batched_fn(handle,
                                                                uplo,
                                                                N,
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

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * symv_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * symv_gbyte_count<T>(N) / gpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "uplo, N, lda, strideA, incx, strideX, incy, stridey, batch_count, "
                        "rocblas-Gflops, rocblas-GB/s, (us) ";
        if(arg.norm_check)
        {
            rocblas_cout << "CPU-Gflops,(us),norm_error_host_ptr,norm_error_dev_ptr";
        }
        rocblas_cout << std::endl;

        rocblas_cout << arg.uplo << ',' << N << ',' << lda << ',' << strideA << incx << ","
                     << stridex << "," << incy << "," << stridey << "," << batch_count << ","
                     << rocblas_gflops << "," << rocblas_bandwidth << ",(" << gpu_time_used << "),";

        if(arg.norm_check)
        {
            rocblas_cout << cblas_gflops << ",(" << cpu_time_used << ")," << h_error << ","
                         << d_error;
        }
        rocblas_cout << std::endl;
    }
}
