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
void testing_scal_ex_bad_arg(const Arguments& arg)
{
    const bool FORTRAN            = arg.fortran;
    auto       rocblas_scal_ex_fn = FORTRAN ? rocblas_scal_ex_fortran : rocblas_scal_ex;

    rocblas_datatype alpha_type     = rocblas_datatype_f32_r;
    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int N     = 100;
    rocblas_int incx  = 1;
    Ta          alpha = (Ta)0.6;

    rocblas_local_handle handle(arg.atomics_mode);

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_vector<Tx> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_ex_fn(handle, N, &alpha, alpha_type, nullptr, x_type, incx, execution_type)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_ex_fn(handle, N, nullptr, alpha_type, dx, x_type, incx, execution_type)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_ex_fn(nullptr, N, &alpha, alpha_type, dx, x_type, incx, execution_type)),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_scal_ex_fn(handle,
                                              N,
                                              nullptr,
                                              rocblas_datatype_f32_c,
                                              nullptr,
                                              rocblas_datatype_f64_r,
                                              incx,
                                              rocblas_datatype_f32_c)),
                          rocblas_status_not_implemented);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex(const Arguments& arg)
{
    const bool FORTRAN            = arg.fortran;
    auto       rocblas_scal_ex_fn = FORTRAN ? rocblas_scal_ex_fortran : rocblas_scal_ex;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int N       = arg.N;
    rocblas_int incx    = arg.incx;
    Ta          h_alpha = arg.get_alpha<Ta>();

    rocblas_local_handle handle(arg.atomics_mode);

    // argument sanity check before allocating invalid memory
    if(N <= 0 || incx <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_scal_ex_fn(
            handle, N, nullptr, alpha_type, nullptr, x_type, incx, execution_type)));
        return;
    }

    size_t size_x = N * size_t(incx);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx_1(size_x);
    host_vector<Tx> hx_2(size_x);
    host_vector<Tx> hy_gold(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<Tx>(hx_1, 1, N, incx);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hx_2    = hx_1;
    hy_gold = hx_1;

    // allocate memory on device
    device_vector<Tx> dx_1(size_x);
    device_vector<Tx> dx_2(size_x);
    device_vector<Ta> d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dx_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1, sizeof(Tx) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1, sizeof(Tx) * size_x, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx_2, hx_2, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_scal_ex_fn(
            handle, N, &h_alpha, alpha_type, dx_1, x_type, incx, execution_type)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_scal_ex_fn(
            handle, N, d_alpha, alpha_type, dx_2, x_type, incx, execution_type)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_1, dx_1, sizeof(Tx) * N * incx, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_2, dx_2, sizeof(Tx) * N * incx, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_scal<Tx, Ta>(N, h_alpha, hy_gold, incx);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, incx, hy_gold, hx_1);
            unit_check_general<Tx>(1, N, incx, hy_gold, hx_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<Tx>('F', 1, N, incx, hy_gold, hx_1);
            rocblas_error_2 = norm_check_general<Tx>('F', 1, N, incx, hy_gold, hx_2);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_ex_fn(handle, N, &h_alpha, alpha_type, dx_1, x_type, incx, execution_type);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_ex_fn(handle, N, &h_alpha, alpha_type, dx_1, x_type, incx, execution_type);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx>{}.log_args<Tx>(rocblas_cout,
                                                           arg,
                                                           gpu_time_used,
                                                           scal_gflop_count<Tx, Ta>(N),
                                                           scal_gbyte_count<Tx>(N),
                                                           cpu_time_used,
                                                           rocblas_error_1,
                                                           rocblas_error_2);
    }
}
