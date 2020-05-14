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
void testing_scal_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_scal_batched_fn
        = FORTRAN ? rocblas_scal_batched<T, U, true> : rocblas_scal_batched<T, U, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    U           h_alpha     = arg.get_alpha<U>();
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N < 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            (rocblas_scal_batched_fn)(handle, N, nullptr, nullptr, incx, batch_count),
            rocblas_status_success);
        return;
    }

    size_t size_x = N * size_t(incx);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice

    // Device-arrays of pointers to device memory
    device_batch_vector<T> dx_1(N, incx, batch_count);
    device_batch_vector<T> dx_2(N, incx, batch_count);
    device_vector<U>       d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dx_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Host-arrays of pointers to host memory
    host_batch_vector<T> hx_1(N, incx, batch_count);
    host_batch_vector<T> hx_2(N, incx, batch_count);
    host_batch_vector<T> hx_gold(N, incx, batch_count);
    host_vector<U>       halpha(1);
    halpha[0] = h_alpha;

    // Initial Data on CPU
    rocblas_init(hx_1, true);
    hx_2.copy_from(hx_1);
    hx_gold.copy_from(hx_1);

    // copy data from CPU to device
    // 1. User intermediate arrays to access device memory from host
    CHECK_HIP_ERROR(dx_1.transfer_from(hx_1));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(dx_2.transfer_from(hx_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((
            rocblas_scal_batched_fn(handle, N, &h_alpha, dx_1.ptr_on_device(), incx, batch_count)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_scal_batched_fn(handle, N, d_alpha, dx_2.ptr_on_device(), incx, batch_count)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_1.transfer_from(dx_1));
        CHECK_HIP_ERROR(hx_2.transfer_from(dx_2));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_scal<T, U>(N, h_alpha, hx_gold[i], incx);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx_gold, hx_1, batch_count);
            unit_check_general<T>(1, N, incx, hx_gold, hx_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, hx_gold, hx_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, hx_gold, hx_2, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_batched_fn(handle, N, &h_alpha, dx_1.ptr_on_device(), incx, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_batched_fn(handle, N, &h_alpha, dx_1.ptr_on_device(), incx, batch_count);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_batch_count>{}.log_args<T>(rocblas_cout,
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
void testing_scal_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_scal_batched_fn
        = FORTRAN ? rocblas_scal_batched<T, U, true> : rocblas_scal_batched<T, U, false>;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    U           h_alpha     = U(1.0);
    rocblas_int batch_count = 5;

    rocblas_local_handle handle;

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_batch_vector<T> dx(N, incx, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_fn)(handle, N, nullptr, dx.ptr_on_device(), incx, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_fn)(handle, N, &h_alpha, nullptr, incx, batch_count),
        rocblas_status_invalid_pointer);
}
