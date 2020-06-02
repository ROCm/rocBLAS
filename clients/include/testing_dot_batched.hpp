/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "bytes.hpp"
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

template <typename T, bool CONJ = false>
void testing_dot_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_dot_batched_fn
        = FORTRAN ? (CONJ ? rocblas_dotc_batched<T, true> : rocblas_dot_batched<T, true>)
                  : (CONJ ? rocblas_dotc_batched<T, false> : rocblas_dot_batched<T, false>);

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int stride_y    = incy * N;
    rocblas_int batch_count = 5;

    rocblas_local_handle   handle;
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_rocblas_result(batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS(
        (rocblas_dot_batched_fn)(
            handle, N, nullptr, incx, dy.ptr_on_device(), incy, batch_count, d_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_dot_batched_fn)(
            handle, N, dx.ptr_on_device(), incx, nullptr, incy, batch_count, d_rocblas_result),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_dot_batched_fn)(handle, N, dx, incx, dy, incy, batch_count, nullptr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_dot_batched_fn)(nullptr,
                                                   N,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   dy.ptr_on_device(),
                                                   incy,
                                                   batch_count,
                                                   d_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_dotc_batched_bad_arg(const Arguments& arg)
{
    testing_dot_batched_bad_arg<T, true>(arg);
}

template <typename T, bool CONJ = false>
void testing_dot_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_dot_batched_fn
        = FORTRAN ? (CONJ ? rocblas_dotc_batched<T, true> : rocblas_dot_batched<T, true>)
                  : (CONJ ? rocblas_dotc_batched<T, false> : rocblas_dot_batched<T, false>);

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    double               rocblas_error_1 = 0;
    double               rocblas_error_2 = 0;
    rocblas_local_handle handle;

    // check to prevent undefined memmory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<T> d_rocblas_result(std::max(batch_count, 1));
        CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR((rocblas_dot_batched_fn)(
            handle, N, nullptr, incx, nullptr, incy, batch_count, d_rocblas_result));
        return;
    }

    host_vector<T> cpu_result(batch_count);
    host_vector<T> rocblas_result_1(batch_count);
    host_vector<T> rocblas_result_2(batch_count);
    rocblas_int    abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int    abs_incy = incy >= 0 ? incy : -incy;
    size_t         size_x   = N * size_t(abs_incx);
    size_t         size_y   = N * size_t(abs_incy);

    //Device-arrays of pointers to device memory
    device_batch_vector<T> dx(N, incx ? incx : 1, batch_count);
    device_batch_vector<T> dy(N, incy ? incy : 1, batch_count);
    device_vector<T>       d_rocblas_result_2(batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx ? incx : 1, batch_count);
    host_batch_vector<T> hy(N, incy ? incy : 1, batch_count);

    // Initial Data on CPU
    rocblas_init(hx, true);
    rocblas_init(hy, false);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;

    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
    auto  dy_ptr = (arg.algo) ? dx.ptr_on_device() : dy.ptr_on_device();
    auto& hy_ptr = (arg.algo) ? hx : hy;
    if(arg.algo)
        incy = incx;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_dot_batched_fn)(
            handle, N, dx.ptr_on_device(), incx, dy_ptr, incy, batch_count, rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_dot_batched_fn)(
            handle, N, dx.ptr_on_device(), incx, dy_ptr, incy, batch_count, d_rocblas_result_2));

        CHECK_HIP_ERROR(rocblas_result_2.transfer_from(d_rocblas_result_2));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx[b], incx, hy_ptr[b], incy, &cpu_result[b]);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, 1, 1, 1, cpu_result, rocblas_result_1, batch_count);
            unit_check_general<T>(1, 1, 1, 1, cpu_result, rocblas_result_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << cpu_result << ", gpu_host_ptr=" << rocblas_result_1
                         << ", gpu_device_ptr=" << rocblas_result_2 << std::endl;

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
            (rocblas_dot_batched_fn)(
                handle, N, dx.ptr_on_device(), incx, dy_ptr, incy, batch_count, d_rocblas_result_2);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            (rocblas_dot_batched_fn)(
                handle, N, dx.ptr_on_device(), incx, dy_ptr, incy, batch_count, d_rocblas_result_2);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count, e_algo>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            dot_gflop_count<CONJ, T>(N),
            dot_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_1);
    }
}

template <typename T>
void testing_dotc_batched(const Arguments& arg)
{
    testing_dot_batched<T, true>(arg);
}
