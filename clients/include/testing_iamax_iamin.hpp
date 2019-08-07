/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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

template <typename T,
          rocblas_status (&FUNC)(rocblas_handle, rocblas_int, const T*, rocblas_int, rocblas_int*)>
void testing_iamax_iamin_bad_arg(const Arguments& arg)
{
    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    static const size_t safe_size = 100;

    rocblas_local_handle handle;
    device_vector<T>     dx(safe_size);
    if(!dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocblas_int h_rocblas_result;

    EXPECT_ROCBLAS_STATUS(FUNC(handle, N, nullptr, incx, &h_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(FUNC(handle, N, dx, incx, nullptr), rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(FUNC(nullptr, N, dx, incx, &h_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_iamax_bad_arg(const Arguments& arg)
{
    testing_iamax_iamin_bad_arg<T, rocblas_iamax<T>>(arg);
}

template <typename T>
void testing_iamin_bad_arg(const Arguments& arg)
{
    testing_iamax_iamin_bad_arg<T, rocblas_iamin<T>>(arg);
}

template <typename T,
          rocblas_status (&FUNC)(rocblas_handle, rocblas_int, const T*, rocblas_int, rocblas_int*),
          void CBLAS_FUNC(rocblas_int, const T*, rocblas_int, rocblas_int*)>
void testing_iamax_iamin(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;

    rocblas_int h_rocblas_result_1;
    rocblas_int h_rocblas_result_2;

    rocblas_int rocblas_error_1;
    rocblas_int rocblas_error_2;

    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        static const size_t safe_size = 100; // arbritrarily set to 100
        device_vector<T>    dx(safe_size);
        if(!dx)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }
        CHECK_ROCBLAS_ERROR(FUNC(handle, N, dx, incx, &h_rocblas_result_1));

#ifdef GOOGLE_TEST
        EXPECT_EQ(h_rocblas_result_1, 0);
#endif
        return;
    }

    size_t size_x = size_t(N) * incx;

    // allocate memory on device
    device_vector<T>           dx(size_x);
    device_vector<rocblas_int> d_rocblas_result(1);
    if(!dx || !d_rocblas_result)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz
    // follow this practice
    host_vector<T> hx(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, incx);

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(FUNC(handle, N, dx, incx, &h_rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(FUNC(handle, N, dx, incx, d_rocblas_result));
        CHECK_HIP_ERROR(hipMemcpy(
            &h_rocblas_result_2, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        rocblas_int cpu_result;
        CBLAS_FUNC(N, hx, incx, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;
        cpu_result += 1; // make index 1 based as in Fortran BLAS, not 0 based as in CBLAS

        if(arg.unit_check)
        {
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_1);
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = h_rocblas_result_1 - cpu_result;
            rocblas_error_2 = h_rocblas_result_2 - cpu_result;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            FUNC(handle, N, dx, incx, d_rocblas_result);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            FUNC(handle, N, dx, incx, d_rocblas_result);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,rocblas-us";

        if(arg.norm_check)
            std::cout << ",cpu_time_used,rocblas_error_host_ptr,rocblas_error_dev_ptr";

        std::cout << std::endl;

        std::cout << (int)N << "," << incx << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}

// CBLAS does not have a cblas_iamin function, so we write our own version of it
namespace rocblas_cblas
{
    template <typename T>
    T asum(T x)
    {
        return x < 0 ? -x : x;
    }

    rocblas_half asum(rocblas_half x)
    {
        return x & 0x7fff;
    }

    template <typename T>
    bool lessthan(T x, T y)
    {
        return x < y;
    }

    bool lessthan(rocblas_half x, rocblas_half y)
    {
        return half_to_float(x) < half_to_float(y);
    }

    template <typename T>
    void cblas_iamin(rocblas_int N, const T* X, rocblas_int incx, rocblas_int* result)
    {
        rocblas_int minpos = -1;
        if(N > 0 && incx > 0)
        {
            auto min = asum(X[0]);
            minpos   = 0;
            for(size_t i = 1; i < N; ++i)
            {
                auto a = asum(X[i * incx]);
                if(lessthan(a, min))
                {
                    min    = a;
                    minpos = i;
                }
            }
        }
        *result = minpos;
    }

} // namespace rocblas_cblas

template <typename T>
void testing_iamax(const Arguments& arg)
{
    testing_iamax_iamin<T, rocblas_iamax<T>, cblas_iamax<T>>(arg);
}

template <typename T>
void testing_iamin(const Arguments& arg)
{
    testing_iamax_iamin<T, rocblas_iamin<T>, rocblas_cblas::cblas_iamin<T>>(arg);
}
