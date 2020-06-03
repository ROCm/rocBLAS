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

template <typename T, typename U = T>
void testing_scal_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_scal_fn = FORTRAN ? rocblas_scal<T, U, true> : rocblas_scal<T, U, false>;

    rocblas_int N     = 100;
    rocblas_int incx  = 1;
    U           alpha = (U)0.6;

    rocblas_local_handle handle;

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_scal_fn(handle, N, &alpha, nullptr, incx)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_scal_fn(handle, N, nullptr, dx, incx)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_scal_fn(nullptr, N, &alpha, dx, incx)),
                          rocblas_status_invalid_handle);
}

template <typename T, typename U = T>
void testing_scal(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_scal_fn = FORTRAN ? rocblas_scal<T, U, true> : rocblas_scal<T, U, false>;

    rocblas_int N       = arg.N;
    rocblas_int incx    = arg.incx;
    U           h_alpha = arg.get_alpha<U>();

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || incx <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_scal_fn(handle, N, nullptr, nullptr, incx)));
        return;
    }

    size_t size_x = N * size_t(incx);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx_1(size_x);
    host_vector<T> hx_2(size_x);
    host_vector<T> hy_gold(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx_1, 1, N, incx);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hx_2    = hx_1;
    hy_gold = hx_1;

    // allocate memory on device
    device_vector<T> dx_1(size_x);
    device_vector<T> dx_2(size_x);
    device_vector<U> d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dx_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1, sizeof(T) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1, sizeof(T) * size_x, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx_2, hx_2, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));

        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_scal_fn(handle, N, &h_alpha, dx_1, incx)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_scal_fn(handle, N, d_alpha, dx_2, incx)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_1, dx_1, sizeof(T) * N * incx, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_2, dx_2, sizeof(T) * N * incx, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_scal<T, U>(N, h_alpha, hy_gold, incx);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, hy_gold, hx_1);
            unit_check_general<T>(1, N, incx, hy_gold, hx_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, hy_gold, hx_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, hy_gold, hx_2);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_fn(handle, N, &h_alpha, dx_1, incx);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_fn(handle, N, &h_alpha, dx_1, incx);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx>{}.log_args<T>(rocblas_cout,
                                                          arg,
                                                          gpu_time_used,
                                                          scal_gflop_count<T, U>(N),
                                                          scal_gbyte_count<T>(N),
                                                          cpu_time_used,
                                                          rocblas_error_1,
                                                          rocblas_error_2);
    }
}
