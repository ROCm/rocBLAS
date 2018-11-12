/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <cmath>

#include "rocblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "near.h"
#include "unit.h"
#include <complex.h>

using namespace std;

template <typename T1, typename T2>
rocblas_status testing_nrm2_bad_arg()
{
    rocblas_int N         = 100;
    rocblas_int incx      = 1;
    rocblas_int safe_size = 100;

    rocblas_status status;

    rocblas_local_handle handle;

    device_vector<T1> dx(safe_size);
    device_vector<T2> d_rocblas_result(1);
    if(!dx || !d_rocblas_result)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // test if (nullptr == dx)
    {
        T1* dx_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_nrm2<T1, T2>(handle, N, dx_null, incx, d_rocblas_result);

        verify_rocblas_status_invalid_pointer(status, "Error: x or result is nullptr");
    }
    // test if (nullptr == d_rocblas_result)
    {
        T2* d_rocblas_result_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_nrm2<T1, T2>(handle, N, dx, incx, d_rocblas_result_null);

        verify_rocblas_status_invalid_pointer(status, "Error: x or result is nullptr");
    }
    // test if (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_nrm2<T1, T2>(handle_null, N, dx, incx, d_rocblas_result);

        verify_rocblas_status_invalid_handle(status);
    }
    return rocblas_status_success;
}

template <typename T1, typename T2>
rocblas_status testing_nrm2(Arguments argus)
{
    rocblas_int N         = argus.N;
    rocblas_int incx      = argus.incx;
    rocblas_int safe_size = 100; //  arbitrarily set to zero

    T2 rocblas_result_1;
    T2 rocblas_result_2;
    T2 cpu_result;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_status status;
    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<T1> dx(safe_size);
        device_vector<T2> d_rocblas_result(1);
        if(!dx || !d_rocblas_result)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_nrm2<T1, T2>(handle, N, dx, incx, d_rocblas_result);
        nrm2_dot_arg_check<T2>(status, d_rocblas_result);
        return status;
    }

    rocblas_int size_x = N * incx;

    // allocate memory on device
    device_vector<T1> dx(size_x);
    device_vector<T2> d_rocblas_result_2(1);
    if(!dx || !d_rocblas_result_2)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<T1> hx(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T1>(hx, 1, N, incx);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T1) * N * incx, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_nrm2<T1, T2>(handle, N, dx, incx, &rocblas_result_1)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_nrm2<T1, T2>(handle, N, dx, incx, d_rocblas_result_2)));
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result_2, d_rocblas_result_2, sizeof(T2), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_nrm2<T1, T2>(N, hx, incx, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;

        //      allowable error is sqrt of precision. This is based on nrm2 calculating the
        //      square root of a sum. It is assumed that the sum will have accuracy =approx=
        //      precision, so nrm2 will have accuracy =approx= sqrt(precision)
        T2 abs_error = pow(10.0, -(std::numeric_limits<T2>::digits10 / 2.0)) * cpu_result;
        T2 tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
                            //  If test fails, try decreasing n or increasing tolerance.
        abs_error = abs_error * tolerance;
        if(argus.unit_check)
        {
            near_check_general<T1, T2>(1, 1, 1, &cpu_result, &rocblas_result_1, abs_error);
            near_check_general<T1, T2>(1, 1, 1, &cpu_result, &rocblas_result_2, abs_error);
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            printf("cpu=%e, gpu_host_ptr=%e, gpu_dev_ptr=%e\n",
                   cpu_result,
                   rocblas_result_1,
                   rocblas_result_2);
            rocblas_error_1 = fabs((cpu_result - rocblas_result_1) / cpu_result);
            rocblas_error_2 = fabs((cpu_result - rocblas_result_2) / cpu_result);
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_nrm2<T1, T2>(handle, N, dx, incx, &rocblas_result_2);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_nrm2<T1, T2>(handle, N, dx, incx, &rocblas_result_2);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        cout << "N,incx,rocblas(us)";

        if(argus.norm_check)
            cout << ",CPU(us),error_host_ptr,error_dev_ptr";

        cout << endl;
        cout << N << "," << incx << "," << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        cout << endl;
    }

    return rocblas_status_success;
}
