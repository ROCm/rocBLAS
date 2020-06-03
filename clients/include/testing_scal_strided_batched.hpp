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
void testing_scal_strided_batched(const Arguments& arg)
{
    const bool FORTRAN                         = arg.fortran;
    auto       rocblas_scal_strided_batched_fn = FORTRAN ? rocblas_scal_strided_batched<T, U, true>
                                                   : rocblas_scal_strided_batched<T, U, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int stridex     = arg.stride_x;
    rocblas_int batch_count = arg.batch_count;
    U           h_alpha     = arg.get_alpha<U>();

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    // --- do no checking for stride_x ---
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS((rocblas_scal_strided_batched_fn)(
                                  handle, N, nullptr, nullptr, incx, stridex, batch_count),
                              rocblas_status_success);
        return;
    }

    size_t size_x = N * size_t(incx) + size_t(stridex) * size_t(batch_count - 1);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx_1(size_x);
    host_vector<T> hx_2(size_x);
    host_vector<T> hx_gold(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx_1, 1, N, incx, stridex, batch_count);

    // copy vector is easy in STL; hx_gold = hx: save a copy in hx_gold which will be output of CPU
    // BLAS
    hx_2    = hx_1;
    hx_gold = hx_1;

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
        CHECK_ROCBLAS_ERROR((rocblas_scal_strided_batched_fn(
            handle, N, &h_alpha, dx_1, incx, stridex, batch_count)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((
            rocblas_scal_strided_batched_fn(handle, N, d_alpha, dx_2, incx, stridex, batch_count)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_1, dx_1, sizeof(T) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_2, dx_2, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_scal<T, U>(N, h_alpha, hx_gold + i * stridex, incx);
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, stridex, hx_gold, hx_1, batch_count);
            unit_check_general<T>(1, N, incx, stridex, hx_gold, hx_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, N, incx, stridex, hx_gold, hx_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', 1, N, incx, stridex, hx_gold, hx_2, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_strided_batched_fn(handle, N, &h_alpha, dx_1, incx, stridex, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_strided_batched_fn(handle, N, &h_alpha, dx_1, incx, stridex, batch_count);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_stride_x, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            scal_gflop_count<T, U>(N),
            scal_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}

template <typename T, typename U = T>
void testing_scal_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN                         = arg.fortran;
    auto       rocblas_scal_strided_batched_fn = FORTRAN ? rocblas_scal_strided_batched<T, U, true>
                                                   : rocblas_scal_strided_batched<T, U, false>;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    U           h_alpha     = U(1.0);
    rocblas_int batch_count = 5;
    rocblas_int stridex     = 50;

    rocblas_local_handle handle;

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_strided_batched_fn)(handle, N, nullptr, dx, incx, stridex, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_strided_batched_fn)(handle, N, &h_alpha, nullptr, incx, stridex, batch_count),
        rocblas_status_invalid_pointer);
}
