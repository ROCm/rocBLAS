/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, typename R>
using rocblas_reduction_batched_t = rocblas_status (*)(rocblas_handle  handle,
                                                       rocblas_int     n,
                                                       const T* const* x,
                                                       rocblas_int     incx,
                                                       rocblas_int     batch_count,
                                                       R*              result);

template <typename T, typename R>
void template_testing_reduction_batched_bad_arg(const Arguments&                  arg,
                                                rocblas_reduction_batched_t<T, R> func)
{
    rocblas_int N = 100, incx = 1, batch_count = 5;

    rocblas_local_handle handle;

    //
    // allocate memory on device
    //
    device_batch_vector<T> dx(N, incx, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    R h_rocblas_result;

    EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, batch_count, &h_rocblas_result),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, batch_count, nullptr),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, batch_count, &h_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T, typename R>
void template_testing_reduction_batched(
    const Arguments&                  arg,
    rocblas_reduction_batched_t<T, R> func,
    void (*REFBLAS_FUNC)(rocblas_int, const T*, rocblas_int, R*))
{
    rocblas_int          N = arg.N, incx = arg.incx, batch_count = arg.batch_count;
    rocblas_stride       stride_x = arg.stride_x;
    double               rocblas_error_1, rocblas_error_2;
    rocblas_local_handle handle;

    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        host_vector<R> res(std::max(1, std::abs(batch_count)));
        CHECK_HIP_ERROR(res.memcheck());
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, batch_count, res),
                              rocblas_status_success);
        return;
    }

    host_vector<R> hr1(batch_count);
    CHECK_HIP_ERROR(hr1.memcheck());
    host_vector<R> hr2(batch_count);
    CHECK_HIP_ERROR(hr2.memcheck());
    host_vector<R> cpu_result(batch_count);
    CHECK_HIP_ERROR(cpu_result.memcheck());
    device_vector<R> dr(batch_count);
    CHECK_DEVICE_ALLOCATION(dr.memcheck());

    host_batch_vector<T> hx(N, incx, batch_count);
    CHECK_HIP_ERROR(hx.memcheck());
    device_batch_vector<T> dx(N, incx, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    //
    // Initialize.
    //
    rocblas_init(hx, true);

    //
    // Copy data from host to device.
    //
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    if(arg.unit_check || arg.norm_check)
    {
        //
        // GPU BLAS, rocblas_pointer_mode_host
        //
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, hr1));
        }

        //
        // GPU BLAS, rocblas_pointer_mode_device
        //
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, dr));
            CHECK_HIP_ERROR(hr2.transfer_from(dr));
        }

        //
        // CPU BLAS
        //
        {
            cpu_time_used = get_time_us();
            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                REFBLAS_FUNC(N, hx[batch_index], incx, cpu_result + batch_index);
            }
            cpu_time_used = get_time_us() - cpu_time_used;
        }

        if(arg.unit_check)
        {
            unit_check_general<R>(batch_count, 1, 1, cpu_result, hr1);
            unit_check_general<R>(batch_count, 1, 1, cpu_result, hr2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = 0.0;
            rocblas_error_2 = 0.0;
            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                double a1       = double(hr1[batch_index]);
                double a2       = double(hr2[batch_index]);
                double c        = double(cpu_result[batch_index]);
                rocblas_error_1 = std::max(rocblas_error_1, std::abs((c - a1) / c));
                rocblas_error_2 = std::max(rocblas_error_2, std::abs((c - a2) / c));
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hr2);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hr2);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "N,incx,batch_count,rocblas(us)";

        if(arg.norm_check)
            rocblas_cout << ",CPU(us),error_host_ptr,error_dev_ptr";

        rocblas_cout << std::endl;
        rocblas_cout << N << "," << incx << "," << batch_count << "," << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cpu_time_used << "," << rocblas_error_1 << ","
                         << rocblas_error_2;

        rocblas_cout << std::endl;
    }
}
