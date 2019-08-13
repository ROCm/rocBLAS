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
void testing_asum_batched_bad_arg_template(const Arguments& arg)
{
    rocblas_int         N                = 100;
    rocblas_int         incx             = 1;
    rocblas_int         batch_count      = 5;
    static const size_t safe_size        = 100;
    T2                  rocblas_result   = 10;
    T2*                 h_rocblas_result = &rocblas_result;

    rocblas_local_handle handle;

    T1** dx;
    hipMalloc(&dx, safe_size * sizeof(T1*));
    if(!dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    EXPECT_ROCBLAS_STATUS(
        (rocblas_asum_batched<T1, T2>(handle, N, nullptr, incx, h_rocblas_result, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_asum_batched<T1, T2>(handle, N, dx, incx, nullptr, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_asum_batched<T1, T2>(nullptr, N, dx, incx, h_rocblas_result, batch_count)),
        rocblas_status_invalid_handle);
}

template <typename T1, typename T2 = T1>
void testing_asum_batched_template(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int batch_count = arg.batch_count;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle;

    if(batch_count <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to zero
        T1**                dx;
        hipMalloc(&dx, safe_size * sizeof(T1*));
        device_vector<T2> d_rocblas_result(std::max(batch_count, 1));
        if(!dx || !d_rocblas_result)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS(
            (rocblas_asum_batched<T1, T2>(handle, N, dx, incx, d_rocblas_result, batch_count)),
            rocblas_status_invalid_size);
        CHECK_HIP_ERROR(hipFree(dx));
        return;
    }

    T2 rocblas_result_1[batch_count];
    T2 rocblas_result_2[batch_count];
    T2 cpu_result[batch_count];

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        T1**                dx;
        hipMalloc(&dx, safe_size * sizeof(T1*));
        device_vector<T2> d_rocblas_result_2(batch_count);
        if(!dx || !d_rocblas_result_2)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_asum_batched<T1, T2>(handle, N, dx, incx, d_rocblas_result_2, batch_count)));
        CHECK_HIP_ERROR(hipFree(dx));
        return;
    }

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    //device_vector<T1> dx(size_x);
    device_vector<T2> d_rocblas_result_2(batch_count);
    if(!d_rocblas_result_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<T1> hx[batch_count];

    // Initial Data on CPU
    rocblas_seedrand();
    for(int i = 0; i < batch_count; i++)
    {
        hx[i] = host_vector<T1>(size_x);
        rocblas_init<T1>(hx[i], 1, N, incx);
    }
    //rocblas_init<T1>(hx, 1, N, incx);

    device_batch_vector<T1> dxvec(batch_count, size_x);
    /*
    T1** hdx = new T1*[batch_count]; // must create device ptr array on host
    */
    for(int i = 0; i < batch_count; i++)
    {
        //hipMalloc(&hdx[i], size_x * sizeof(T1));
        CHECK_HIP_ERROR(hipMemcpy(dxvec[i], hx[i], size_x * sizeof(T1), hipMemcpyHostToDevice));
    }

    // vector pointers on gpu
    T1** dx_pvec;
    CHECK_HIP_ERROR(hipMalloc(&dx_pvec, batch_count * sizeof(T1*)));
    if(!dx_pvec)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }
    // copy gpu vector pointers from host to device pointer array
    CHECK_HIP_ERROR(hipMemcpy(dx_pvec, dxvec, sizeof(T1*) * batch_count, hipMemcpyHostToDevice));

    // copy data from CPU to device, does not work for incx != 1
    //CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T1) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((
            rocblas_asum_batched<T1, T2>(handle, N, dx_pvec, incx, rocblas_result_1, batch_count)));

        // GPU BLAS rocblas_pointer_mode_device
        //CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T1) * size_x, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_asum_batched<T1, T2>(
            handle, N, dx_pvec, incx, d_rocblas_result_2, batch_count)));
        CHECK_HIP_ERROR(hipMemcpy(
            rocblas_result_2, d_rocblas_result_2, batch_count * sizeof(T2), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_asum<T1, T2>(N, hx[i], incx, cpu_result + i);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T2>(1, batch_count, 1, cpu_result, rocblas_result_1);
            unit_check_general<T2>(1, batch_count, 1, cpu_result, rocblas_result_2);
        }

        if(arg.norm_check)
        {
            std::cout << "cpu=" << std::scientific << cpu_result[0]
                      << ", gpu_host_ptr=" << rocblas_result_1[0]
                      << ", gpu_dev_ptr=" << rocblas_result_2[0] << "\n";

            rocblas_error_1 = std::abs((cpu_result[0] - rocblas_result_1[0]) / cpu_result[0]);
            rocblas_error_2 = std::abs((cpu_result[0] - rocblas_result_2[0]) / cpu_result[0]);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum_batched<T1, T2>(handle, N, dx_pvec, incx, rocblas_result_1, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum_batched<T1, T2>(handle, N, dx_pvec, incx, rocblas_result_1, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,batch_count,rocblas(us)";

        if(arg.norm_check)
            std::cout << ",CPU(us),error_host_ptr,error_dev_ptr";

        std::cout << std::endl;
        std::cout << N << "," << incx << "," << batch_count << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(dx_pvec));
}

template <typename T>
void testing_asum_batched_bad_arg(const Arguments& arg)
{
    testing_asum_batched_bad_arg_template<T>(arg);
}

template <>
void testing_asum_batched_bad_arg<rocblas_float_complex>(const Arguments& arg)
{
    testing_asum_batched_bad_arg_template<rocblas_float_complex, float>(arg);
}

template <>
void testing_asum_batched_bad_arg<rocblas_double_complex>(const Arguments& arg)
{
    testing_asum_batched_bad_arg_template<rocblas_double_complex, double>(arg);
}

template <typename T>
void testing_asum_batched(const Arguments& arg)
{
    return testing_asum_batched_template<T>(arg);
}

template <>
void testing_asum_batched<rocblas_float_complex>(const Arguments& arg)
{
    return testing_asum_batched_template<rocblas_float_complex, float>(arg);
}

template <>
void testing_asum_batched<rocblas_double_complex>(const Arguments& arg)
{
    return testing_asum_batched_template<rocblas_double_complex, double>(arg);
}
