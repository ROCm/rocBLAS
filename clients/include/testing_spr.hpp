/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
void testing_spr_bad_arg()
{
    rocblas_fill         uplo  = rocblas_fill_upper;
    rocblas_int          N     = 100;
    rocblas_int          incx  = 1;
    T                    alpha = 0.6;
    rocblas_local_handle handle;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t size_A   = size_t(N) * (N + 1) / 2;
    size_t size_x   = size_t(N) * abs_incx;

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dx(size_x);
    CHECK_HIP_ERROR(dA_1.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_spr<T>(handle, rocblas_fill_full, N, &alpha, dx, incx, dA_1),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_spr<T>(handle, uplo, N, &alpha, nullptr, incx, dA_1),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_spr<T>(handle, uplo, N, &alpha, dx, incx, nullptr),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_spr<T>(nullptr, uplo, N, &alpha, dx, incx, dA_1),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_spr(const Arguments& arg)
{
    rocblas_int          N       = arg.N;
    rocblas_int          incx    = arg.incx;
    T                    h_alpha = arg.get_alpha<T>();
    rocblas_fill         uplo    = char2rocblas_fill(arg.uplo);
    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(N < 0 || !incx)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_spr<T>(handle, uplo, N, nullptr, nullptr, incx, nullptr),
                              rocblas_status_invalid_size);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t size_A   = size_t(N) * (N + 1) / 2;
    size_t size_x   = size_t(N) * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_1(size_A);
    host_vector<T> hA_2(size_A);
    host_vector<T> hA_gold(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> halpha(1);
    CHECK_HIP_ERROR(hA_1.memcheck());
    CHECK_HIP_ERROR(hA_2.memcheck());
    CHECK_HIP_ERROR(hA_gold.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(halpha.memcheck());

    halpha[0] = h_alpha;

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dA_2(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> d_alpha(1);
    CHECK_HIP_ERROR(dA_1.memcheck());
    CHECK_HIP_ERROR(dA_2.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(d_alpha.memcheck());

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initial Data on CPU
    rocblas_init(hA_1, true);
    rocblas_init(hx, false);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA_1.transfer_from(hA_1));
    CHECK_HIP_ERROR(dA_2.transfer_from(hA_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_spr<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_spr<T>(handle, uplo, N, d_alpha, dx, incx, dA_2));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_1));
        CHECK_HIP_ERROR(hA_2.transfer_from(dA_2));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_spr<T>(uplo, N, h_alpha, hx, incx, hA_gold);
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = spr_gflop_count<T>(N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_float_complex>{}
               || std::is_same<T, rocblas_double_complex>{})
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(1, size_A, 1, hA_gold, hA_1, tol);
                near_check_general<T>(1, size_A, 1, hA_gold, hA_2, tol);
            }
            else
            {
                unit_check_general<T>(1, size_A, 1, hA_gold, hA_1);
                unit_check_general<T>(1, size_A, 1, hA_gold, hA_2);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, size_A, 1, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, size_A, 1, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_spr<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_spr<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = spr_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = (2.0 * N * (N + 1)) / 2 * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,alpha,incx,rocblas-Gflops,rocblas-GB/s";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << N << "," << h_alpha << "," << incx << "," << rocblas_gflops << ","
                  << rocblas_bandwidth;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
