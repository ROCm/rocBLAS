/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
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

template <typename Tx, typename Tr>
void testing_nrm2_strided_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_nrm2_strided_batched_ex_fn
        = arg.fortran ? rocblas_nrm2_strided_batched_ex_fortran : rocblas_nrm2_strided_batched_ex;

    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype result_type    = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_stride stridex     = 1;
    rocblas_int    batch_count = 5;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_vector<Tx> dx(N, incx, stridex, batch_count);
    device_vector<Tr>               d_rocblas_result(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                             N,
                                                             nullptr,
                                                             x_type,
                                                             incx,
                                                             stridex,
                                                             batch_count,
                                                             d_rocblas_result,
                                                             result_type,
                                                             execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                             N,
                                                             dx,
                                                             x_type,
                                                             incx,
                                                             stridex,
                                                             batch_count,
                                                             nullptr,
                                                             result_type,
                                                             execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_strided_batched_ex_fn(nullptr,
                                                             N,
                                                             dx,
                                                             x_type,
                                                             incx,
                                                             stridex,
                                                             batch_count,
                                                             d_rocblas_result,
                                                             result_type,
                                                             execution_type),
                          rocblas_status_invalid_handle);
}

template <typename Tx, typename Tr>
void testing_nrm2_strided_batched_ex(const Arguments& arg)
{
    auto rocblas_nrm2_strided_batched_ex_fn
        = arg.fortran ? rocblas_nrm2_strided_batched_ex_fortran : rocblas_nrm2_strided_batched_ex;

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype result_type    = arg.b_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_stride stridex     = arg.stride_x;
    rocblas_int    batch_count = arg.batch_count;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_rocblas_result_0(std::max(1, batch_count));
        host_vector<Tr>   h_rocblas_result_0(std::max(1, batch_count));
        CHECK_HIP_ERROR(d_rocblas_result_0.memcheck());
        CHECK_HIP_ERROR(h_rocblas_result_0.memcheck());

        rocblas_init_nan(h_rocblas_result_0, 1, std::max(1, batch_count), 1);
        CHECK_HIP_ERROR(hipMemcpy(d_rocblas_result_0,
                                  h_rocblas_result_0,
                                  sizeof(Tr) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                               N,
                                                               nullptr,
                                                               x_type,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               d_rocblas_result_0,
                                                               result_type,
                                                               execution_type));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                               N,
                                                               nullptr,
                                                               x_type,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               h_rocblas_result_0,
                                                               result_type,
                                                               execution_type));

        if(batch_count > 0)
        {
            host_vector<Tr> cpu_0(batch_count);
            host_vector<Tr> gpu_0(batch_count);
            CHECK_HIP_ERROR(cpu_0.memcheck());
            CHECK_HIP_ERROR(gpu_0.memcheck());

            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_rocblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
            unit_check_general<Tr>(1, batch_count, 1, cpu_0, h_rocblas_result_0);
        }

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_strided_batch_vector<Tx> hx(N, incx, stridex, batch_count);
    host_vector<Tr>               rocblas_result_1(batch_count);
    host_vector<Tr>               rocblas_result_2(batch_count);
    host_vector<Tr>               cpu_result(batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hx.memcheck());

    // Allocate device memory
    device_strided_batch_vector<Tx> dx(N, incx, stridex, batch_count);
    device_vector<Tr>               d_rocblas_result_2(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                               N,
                                                               dx,
                                                               x_type,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               rocblas_result_1,
                                                               result_type,
                                                               execution_type));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                               N,
                                                               dx,
                                                               x_type,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               d_rocblas_result_2,
                                                               result_type,
                                                               execution_type));

        CHECK_HIP_ERROR(rocblas_result_2.transfer_from(d_rocblas_result_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_nrm2<Tx>(N, hx[b], incx, cpu_result + b);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        Tr abs_result = cpu_result[0] > 0 ? cpu_result[0] : -cpu_result[0];
        Tr abs_error;
        if(abs_result > 0)
        {
            abs_error = std::numeric_limits<Tr>::epsilon() * N * abs_result;
        }
        else
        {
            abs_error = std::numeric_limits<Tr>::epsilon() * N;
        }
        Tr tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
            //  If test fails, try decreasing n or increasing tolerance.
        abs_error *= tolerance;

        if(!rocblas_isnan(arg.alpha))
        {
            if(arg.unit_check)
            {
                near_check_general<Tr, Tr>(
                    batch_count, 1, 1, cpu_result, rocblas_result_1, abs_error);
                near_check_general<Tr, Tr>(
                    batch_count, 1, 1, cpu_result, rocblas_result_2, abs_error);
            }
        }

        if(arg.norm_check)
        {
            for(int b = 0; b < batch_count; ++b)
            {
                rocblas_error_1
                    += rocblas_abs((cpu_result[b] - rocblas_result_1[b]) / cpu_result[b]);
                rocblas_error_2
                    += rocblas_abs((cpu_result[b] - rocblas_result_2[b]) / cpu_result[b]);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_nrm2_strided_batched_ex_fn(handle,
                                               N,
                                               dx,
                                               x_type,
                                               incx,
                                               stridex,
                                               batch_count,
                                               d_rocblas_result_2,
                                               result_type,
                                               execution_type);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_nrm2_strided_batched_ex_fn(handle,
                                               N,
                                               dx,
                                               x_type,
                                               incx,
                                               stridex,
                                               batch_count,
                                               d_rocblas_result_2,
                                               result_type,
                                               execution_type);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_batch_count>{}.log_args<Tx>(
            rocblas_cout,
            arg,
            gpu_time_used,
            nrm2_gflop_count<Tx>(N),
            nrm2_gbyte_count<Tx>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
