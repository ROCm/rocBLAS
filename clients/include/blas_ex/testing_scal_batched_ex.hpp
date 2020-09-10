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

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_batched_ex_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_scal_batched_ex_fn
        = FORTRAN ? rocblas_scal_batched_ex_fortran : rocblas_scal_batched_ex;

    rocblas_datatype alpha_type     = rocblas_datatype_f32_r;
    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    Ta          h_alpha     = Ta(1.0);
    rocblas_int batch_count = 5;

    rocblas_local_handle handle(arg.atomics_mode);

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_batch_vector<Tx> dx(N, incx, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_scal_batched_ex_fn)(handle,
                                                       N,
                                                       nullptr,
                                                       alpha_type,
                                                       dx.ptr_on_device(),
                                                       x_type,
                                                       incx,
                                                       batch_count,
                                                       execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_ex_fn)(
            handle, N, &h_alpha, alpha_type, nullptr, x_type, incx, batch_count, execution_type),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_ex_fn)(
            nullptr, N, &h_alpha, alpha_type, dx, x_type, incx, batch_count, execution_type),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_scal_batched_ex_fn)(handle,
                                                       N,
                                                       nullptr,
                                                       rocblas_datatype_f32_r,
                                                       nullptr,
                                                       rocblas_datatype_f64_c,
                                                       incx,
                                                       batch_count,
                                                       rocblas_datatype_f64_c),
                          rocblas_status_not_implemented);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_batched_ex(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_scal_batched_ex_fn
        = FORTRAN ? rocblas_scal_batched_ex_fortran : rocblas_scal_batched_ex;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    Ta          h_alpha     = arg.get_alpha<Ta>();
    rocblas_int batch_count = arg.batch_count;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_local_handle handle(arg.atomics_mode);

    // argument sanity check before allocating invalid memory
    if(N < 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            (rocblas_scal_batched_ex_fn)(
                handle, N, nullptr, alpha_type, nullptr, x_type, incx, batch_count, execution_type),
            rocblas_status_success);
        return;
    }

    size_t size_x = N * size_t(incx);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice

    // Device-arrays of pointers to device memory
    device_batch_vector<Tx> dx_1(N, incx, batch_count);
    device_batch_vector<Tx> dx_2(N, incx, batch_count);
    device_vector<Ta>       d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dx_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Host-arrays of pointers to host memory
    host_batch_vector<Tx> hx_1(N, incx, batch_count);
    host_batch_vector<Tx> hx_2(N, incx, batch_count);
    host_batch_vector<Tx> hx_gold(N, incx, batch_count);
    host_vector<Ta>       halpha(1);
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
        CHECK_ROCBLAS_ERROR((rocblas_scal_batched_ex_fn(handle,
                                                        N,
                                                        &h_alpha,
                                                        alpha_type,
                                                        dx_1.ptr_on_device(),
                                                        x_type,
                                                        incx,
                                                        batch_count,
                                                        execution_type)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_scal_batched_ex_fn(handle,
                                                        N,
                                                        d_alpha,
                                                        alpha_type,
                                                        dx_2.ptr_on_device(),
                                                        x_type,
                                                        incx,
                                                        batch_count,
                                                        execution_type)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_1.transfer_from(dx_1));
        CHECK_HIP_ERROR(hx_2.transfer_from(dx_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_scal<Tx, Ta>(N, h_alpha, hx_gold[i], incx);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, incx, hx_gold, hx_1, batch_count);
            unit_check_general<Tx>(1, N, incx, hx_gold, hx_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx_1, batch_count);
            rocblas_error_2 = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx_2, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_batched_ex_fn(handle,
                                       N,
                                       &h_alpha,
                                       alpha_type,
                                       dx_1.ptr_on_device(),
                                       x_type,
                                       incx,
                                       batch_count,
                                       execution_type);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_batched_ex_fn(handle,
                                       N,
                                       &h_alpha,
                                       alpha_type,
                                       dx_1.ptr_on_device(),
                                       x_type,
                                       incx,
                                       batch_count,
                                       execution_type);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_batch_count>{}.log_args<Tx>(
            rocblas_cout,
            arg,
            gpu_time_used,
            scal_gflop_count<Tx, Ta>(N),
            scal_gbyte_count<Tx>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
