/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_nrm2_strided_batched_ex_fn
        = FORTRAN ? rocblas_nrm2_strided_batched_ex_fortran : rocblas_nrm2_strided_batched_ex;

    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype result_type    = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int         N           = 100;
    rocblas_int         incx        = 1;
    rocblas_stride      stridex     = 1;
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 100;

    rocblas_local_handle handle{arg};

    device_vector<Tx> dx(safe_size);
    device_vector<Tr> d_rocblas_result(batch_count);
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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_nrm2_strided_batched_ex_fn
        = FORTRAN ? rocblas_nrm2_strided_batched_ex_fortran : rocblas_nrm2_strided_batched_ex;

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
        host_vector<Tr> res(std::max(1, std::abs(batch_count)));
        CHECK_HIP_ERROR(res.memcheck());
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_nrm2_strided_batched_ex_fn(handle,
                                                                 N,
                                                                 nullptr,
                                                                 x_type,
                                                                 incx,
                                                                 stridex,
                                                                 batch_count,
                                                                 res,
                                                                 result_type,
                                                                 execution_type),
                              rocblas_status_success);
        return;
    }

    Tr rocblas_result_1[batch_count];
    Tr rocblas_result_2[batch_count];
    Tr cpu_result[batch_count];

    size_t size_x = (size_t)stridex;

    // allocate memory on device
    device_vector<Tx> dx(batch_count * size_x);
    device_vector<Tr> d_rocblas_result_2(batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(batch_count * size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<Tx>(hx, 1, N, incx, stridex, batch_count);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x * batch_count, hipMemcpyHostToDevice));

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
        CHECK_HIP_ERROR(hipMemcpy(
            rocblas_result_2, d_rocblas_result_2, batch_count * sizeof(Tr), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int i = 0; i < batch_count; i++)
            cblas_nrm2<Tx>(N, hx + i * stridex, incx, cpu_result + i);
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
        if(arg.unit_check)
        {
            near_check_general<Tr, Tr>(batch_count, 1, 1, cpu_result, rocblas_result_1, abs_error);
            near_check_general<Tr, Tr>(batch_count, 1, 1, cpu_result, rocblas_result_2, abs_error);
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << cpu_result[0] << ", gpu_host_ptr=" << rocblas_result_1[0]
                         << ", gpu_dev_ptr=" << rocblas_result_2[0] << "\n";
            rocblas_error_1 = ((cpu_result[0] - rocblas_result_1[0]) / cpu_result[0]);
            rocblas_error_2 = ((cpu_result[0] - rocblas_result_2[0]) / cpu_result[0]);
            rocblas_error_1 = rocblas_error_1 < 0 ? -rocblas_error_1 : rocblas_error_1;
            rocblas_error_2 = rocblas_error_2 < 0 ? -rocblas_error_2 : rocblas_error_2;
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
