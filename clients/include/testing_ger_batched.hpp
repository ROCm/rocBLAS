/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
void testing_ger_batched_bad_arg(const Arguments& arg)
{
    rocblas_int       M           = 100;
    rocblas_int       N           = 100;
    rocblas_int       incx        = 1;
    rocblas_int       incy        = 1;
    rocblas_int       lda         = 100;
    T                 alpha       = 0.6;
    const rocblas_int batch_count = 5;

    size_t size_A = lda * size_t(N);

    rocblas_local_handle handle;

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched<T, CONJ>(
                              handle, M, N, &alpha, nullptr, incx, dy, incy, dA, lda, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched<T, CONJ>(
                              handle, M, N, &alpha, dx, incx, nullptr, incy, dA, lda, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched<T, CONJ>(
                              handle, M, N, &alpha, dx, incx, dy, incy, nullptr, lda, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched<T, CONJ>(
                              nullptr, M, N, &alpha, dx, incx, dy, incy, dA, lda, batch_count)),
                          rocblas_status_invalid_handle);
}

template <typename T, bool CONJ>
void testing_ger_batched(const Arguments& arg)
{
    rocblas_int M           = arg.M;
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int lda         = arg.lda;
    T           h_alpha     = arg.get_alpha<T>();
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(M <= 0 || N <= 0 || lda < M || lda < 1 || !incx || !incy || batch_count <= 0)
    {
        static constexpr size_t safe_size = 100;

        EXPECT_ROCBLAS_STATUS(
            (rocblas_ger_batched<T, CONJ>(
                handle, M, N, &h_alpha, nullptr, incx, nullptr, incy, nullptr, lda, batch_count)),

            M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy || batch_count < 0
                ? rocblas_status_invalid_size
                : rocblas_status_success);
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = size_t(lda) * N;
    size_t size_x   = M * abs_incx;
    size_t size_y   = N * abs_incy;

    // check memory info before allocation
    size_t needed_mem = (size_x + size_y + size_A + size_A) * batch_count * sizeof(T) + sizeof(T);
    if(is_limited_memory(needed_mem))
    {
#ifdef GOOGLE_TEST
        SUCCEED() << LIMITED_MEMORY_STRING;
#endif
        return;
    }

    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);
    device_batch_vector<T> dA_1(size_A, 1, batch_count);
    device_batch_vector<T> dA_2(size_A, 1, batch_count);
    device_vector<T>       d_alpha(1);
    CHECK_HIP_ERROR(dy.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dA_1.memcheck());
    CHECK_HIP_ERROR(dA_2.memcheck());

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    // Host-arrays of pointers to host memory
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hA_1(size_A, 1, batch_count);
    host_batch_vector<T> hA_2(size_A, 1, batch_count);
    host_batch_vector<T> hA_gold(size_A, 1, batch_count);

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
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA_2.transfer_from(hA_2));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_ger_batched<T, CONJ>(handle,
                                                          M,
                                                          N,
                                                          &h_alpha,
                                                          dx.ptr_on_device(),
                                                          incx,
                                                          dy.ptr_on_device(),
                                                          incy,
                                                          dA_1.ptr_on_device(),
                                                          lda,
                                                          batch_count)));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_ger_batched<T, CONJ>(handle,
                                                          M,
                                                          N,
                                                          d_alpha,
                                                          dx.ptr_on_device(),
                                                          incx,
                                                          dy.ptr_on_device(),
                                                          incy,
                                                          dA_2.ptr_on_device(),
                                                          lda,
                                                          batch_count)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_1));
        CHECK_HIP_ERROR(hA_2.transfer_from(dA_2));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_ger<T, CONJ>(M, N, h_alpha, hx[b], incx, hy[b], incy, hA_gold[b], lda);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * ger_gflop_count<T, CONJ>(M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(M, N, batch_count, lda, hA_gold, hA_1);
                unit_check_general<T>(M, N, batch_count, lda, hA_gold, hA_2);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                for(int i = 0; i < batch_count; i++)
                {
                    near_check_general<T>(M, N, lda, hA_gold[i], hA_1[i], tol);
                    near_check_general<T>(M, N, lda, hA_gold[i], hA_2[i], tol);
                }
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, lda, batch_count, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', M, N, lda, batch_count, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_ger_batched<T, CONJ>(handle,
                                         M,
                                         N,
                                         &h_alpha,
                                         dx.ptr_on_device(),
                                         incx,
                                         dy.ptr_on_device(),
                                         incy,
                                         dA_1.ptr_on_device(),
                                         lda,
                                         batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_ger_batched<T, CONJ>(handle,
                                         M,
                                         N,
                                         &h_alpha,
                                         dx.ptr_on_device(),
                                         incx,
                                         dy.ptr_on_device(),
                                         incy,
                                         dA_1.ptr_on_device(),
                                         lda,
                                         batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * ger_gflop_count<T, CONJ>(M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * (2.0 * M * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,N,alpha,incx,incy,lda,batch_count,rocblas-Gflops,rocblas-GB/s";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << M << "," << N << "," << h_alpha << "," << incx << "," << incy << "," << lda
                  << "," << batch_count << "," << rocblas_gflops << "," << rocblas_bandwidth;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
