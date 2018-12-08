/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "flops.hpp"

template <typename T>
void testing_symv(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int lda  = arg.lda;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    T alpha = static_cast<T>(arg.alpha);
    T beta  = static_cast<T>(arg.beta);

    rocblas_fill uplo = char2rocblas_fill(arg.uplo);

    rocblas_int size_A = lda * N;
    rocblas_int size_X = N * incx;
    rocblas_int size_Y = N * incy;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N < 0 || lda < 0 || incx < 0 || incy < 0)
    {
        static const size_t safe_size = 100;
        device_vector<T> dA(safe_size);
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);
        if(!dA || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_symv<T>(handle, uplo, N, &alpha, dA, lda, dx, incx, &beta, dy, incy),
            rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_X);
    host_vector<T> hy(size_Y);
    host_vector<T> hz(size_Y);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;

    char char_fill = arg.uplo;

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_X);
    device_vector<T> dy(size_Y);
    if(!dA || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init_symmetric<T>(hA, N, lda);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, N, incy);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hz = hy;

    // copy data from CPU to device
    hipMemcpy(dA, hA, sizeof(T) * lda * N, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx, sizeof(T) * N * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy, sizeof(T) * N * incy, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        CHECK_ROCBLAS_ERROR(
            rocblas_symv<T>(handle, uplo, N, &alpha, dA, lda, dx, incx, &beta, dy, incy));
    }
    if(arg.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = symv_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
    }

    // copy output from device to CPU
    hipMemcpy(hy, dy, sizeof(T) * N * incy, hipMemcpyDeviceToHost);

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(arg.timing)
        {
            cpu_time_used = get_time_us();
        }

        cblas_symv<T>(uplo, N, alpha, hA, lda, hx, incx, beta, hz, incy);

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = symv_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incy, hz, hy);
        }

#if 0
        for(int i = 0; i < 32; i++)
        {
            printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hz[i], i, hy[i]);
        }
#endif

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, incy, hz, hy);
        }
    }

    if(arg.timing)
    {
        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N, lda, rocblas-Gflops (us) ";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops(us), norm-error";
        }
        std::cout << std::endl;

        std::cout << N << ',' << lda << ',' << rocblas_gflops << "(" << gpu_time_used << "),";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << "(" << cpu_time_used << "),";
            std::cout << rocblas_error;
        }

        std::cout << std::endl;
    }
}
