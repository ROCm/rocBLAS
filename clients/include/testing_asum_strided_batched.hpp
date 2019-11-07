/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "near.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T1, typename T2 = T1>
void testing_asum_strided_batched_bad_arg_template(const Arguments& arg)
{
    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_stride stridex     = N;
    rocblas_int    batch_count = 5;
    T2             h_rocblas_result[1];

    device_strided_batch_vector<T1> dx(N, incx, stridex, batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    rocblas_local_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    EXPECT_ROCBLAS_STATUS((rocblas_asum_strided_batched<T1, T2>(
                              handle, N, nullptr, incx, stridex, batch_count, h_rocblas_result)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_asum_strided_batched<T1, T2>(handle, N, dx, incx, stridex, batch_count, nullptr)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_asum_strided_batched<T1, T2>(
                              nullptr, N, dx, incx, stridex, batch_count, h_rocblas_result)),
                          rocblas_status_invalid_handle);
};

template <typename T1, typename T2 = T1>
void testing_asum_strided_batched_template(const Arguments& arg)
{

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_stride stridex     = arg.stride_x;
    rocblas_int    batch_count = arg.batch_count;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        device_strided_batch_vector<T1> dx(3, 1, 3, 3);
        CHECK_HIP_ERROR(dx.memcheck());
        device_vector<T2> dr(std::max(3, std::abs(batch_count)));
        CHECK_HIP_ERROR(dr.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS(
            (rocblas_asum_strided_batched<T1, T2>(handle, N, dx, incx, stridex, batch_count, dr)),
            (N > 0 && incx > 0 && batch_count < 0) ? rocblas_status_invalid_size
                                                   : rocblas_status_success);

        return;
    }

    // allocate memory on device
    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice

    host_strided_batch_vector<T1> hx(N, incx, stridex, batch_count);
    CHECK_HIP_ERROR(hx.memcheck());
    device_strided_batch_vector<T1> dx(N, incx, stridex, batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    device_vector<T2> dr(batch_count);
    CHECK_HIP_ERROR(dr.memcheck());
    host_vector<T2> hr1(batch_count);
    CHECK_HIP_ERROR(hr1.memcheck());
    host_vector<T2> hr(batch_count);
    CHECK_HIP_ERROR(hr.memcheck());

    //
    // Initialize the host vector.
    //
    rocblas_init(hx, true);

    //
    // copy data from CPU to device, does not work for incx != 1
    //
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            (rocblas_asum_strided_batched<T1, T2>(handle, N, dx, incx, stridex, batch_count, hr1)));

        // GPU BgdLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_asum_strided_batched<T1, T2>(handle, N, dx, incx, stridex, batch_count, dr)));

        CHECK_HIP_ERROR(hr.transfer_from(dr));

        // CPU BLAS
        T2 cpu_result[batch_count];

        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_asum<T1, T2>(N, hx + i * stridex, incx, cpu_result + i);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T2>(batch_count, 1, 1, cpu_result, hr1);
            unit_check_general<T2>(batch_count, 1, 1, cpu_result, hr);
        }

        if(arg.norm_check)
        {
            std::cout << "cpu=" << std::scientific << cpu_result[0] << ", gpu_host_ptr=" << hr1[0]
                      << ", gpu_dev_ptr=" << hr[0] << "\n";

            rocblas_error_1 = std::abs((cpu_result[0] - hr1[0]) / cpu_result[0]);
            rocblas_error_2 = std::abs((cpu_result[0] - hr[0]) / cpu_result[0]);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum_strided_batched<T1, T2>(handle, N, dx, incx, stridex, batch_count, hr);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum_strided_batched<T1, T2>(handle, N, dx, incx, stridex, batch_count, hr);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,stridex,batch_count,rocblas(us)";

        if(arg.norm_check)
            std::cout << ",CPU(us),error_host_ptr,error_dev_ptr";

        std::cout << std::endl;
        std::cout << N << "," << incx << "," << stridex << "," << batch_count << ","
                  << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}

template <typename T>
void testing_asum_strided_batched_bad_arg(const Arguments& arg)
{
    testing_asum_strided_batched_bad_arg_template<T>(arg);
}

template <>
void testing_asum_strided_batched_bad_arg<rocblas_float_complex>(const Arguments& arg)
{
    testing_asum_strided_batched_bad_arg_template<rocblas_float_complex, float>(arg);
}

template <>
void testing_asum_strided_batched_bad_arg<rocblas_double_complex>(const Arguments& arg)
{
    testing_asum_strided_batched_bad_arg_template<rocblas_double_complex, double>(arg);
}

template <typename T>
void testing_asum_strided_batched(const Arguments& arg)
{
    return testing_asum_strided_batched_template<T>(arg);
}

template <>
void testing_asum_strided_batched<rocblas_float_complex>(const Arguments& arg)
{
    return testing_asum_strided_batched_template<rocblas_float_complex, float>(arg);
}

template <>
void testing_asum_strided_batched<rocblas_double_complex>(const Arguments& arg)
{
    return testing_asum_strided_batched_template<rocblas_double_complex, double>(arg);
}
