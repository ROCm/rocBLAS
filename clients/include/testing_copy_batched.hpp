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

template <typename T>
void testing_copy_batched_bad_arg(const Arguments& arg)
{
    rocblas_int       N           = 100;
    rocblas_int       incx        = 1;
    rocblas_int       incy        = 1;
    const rocblas_int batch_count = 5;

    rocblas_local_handle handle;

    // allocate memory on device
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_copy_batched<T>(handle, N, nullptr, incx, dy.ptr_on_device(), incy, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_copy_batched<T>(handle, N, dx.ptr_on_device(), incx, nullptr, incy, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_copy_batched<T>(
            nullptr, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_copy_batched(const Arguments& arg)
{
    rocblas_int          N    = arg.N;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_local_handle handle;
    rocblas_int          batch_count = arg.batch_count;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        size_t                 safe_size = 100;
        device_batch_vector<T> dx(safe_size, 1, 1);
        device_batch_vector<T> dy(safe_size, 1, 1);
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        EXPECT_ROCBLAS_STATUS(
            rocblas_copy_batched<T>(
                handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count),
            N > 0 && batch_count < 0 ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    //Device-arrays of pointers to device memory
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hy_gold(N, incy, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);

    // Initial Data on CPU
    rocblas_init(hx, true);
    rocblas_init(hy, false);
    hy_gold.copy_from(hy);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_copy_batched<T>(
            handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));

        CHECK_HIP_ERROR(hy.transfer_from(dy));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_copy<T>(N, hx[b], incx, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, hy_gold, hy);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, batch_count, hy_gold, hy);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_copy_batched<T>(
                handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_copy_batched<T>(
                handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "N,incx,incy,batch_count,rocblas-us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-us,error";

        rocblas_cout << std::endl;

        rocblas_cout << N << "," << incx << "," << incy << "," << batch_count << ","
                     << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cpu_time_used << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}
