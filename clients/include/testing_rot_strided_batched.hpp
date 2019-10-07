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

template <typename T, typename U = T, typename V = T>
void testing_rot_strided_batched_bad_arg(const Arguments& arg)
{
    rocblas_int         N           = 100;
    rocblas_int         incx        = 1;
    rocblas_stride      stride_x    = 1;
    rocblas_int         incy        = 1;
    rocblas_stride      stride_y    = 1;
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 100;

    rocblas_local_handle handle;
    device_vector<T>     dx(safe_size);
    device_vector<T>     dy(safe_size);
    device_vector<U>     dc(1);
    device_vector<V>     ds(1);
    if(!dx || !dy || !dc || !ds)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_strided_batched<T, U, V>(
            nullptr, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count)),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_strided_batched<T, U, V>(
            handle, N, nullptr, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_strided_batched<T, U, V>(
            handle, N, dx, incx, stride_x, nullptr, incy, stride_y, dc, ds, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_strided_batched<T, U, V>(
            handle, N, dx, incx, stride_x, dy, incy, stride_y, nullptr, ds, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_strided_batched<T, U, V>(
            handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, nullptr, batch_count)),
        rocblas_status_invalid_pointer);
}

template <typename T, typename U = T, typename V = T>
void testing_rot_strided_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int stride_x    = arg.stride_x;
    rocblas_int stride_y    = arg.stride_y;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || batch_count <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        device_vector<T>    dy(safe_size);
        device_vector<U>    dc(1);
        device_vector<V>    ds(1);
        if(!dx || !dy || !dc || !ds)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched<T, U, V>)(handle,
                                                                         N,
                                                                         dx,
                                                                         incx,
                                                                         stride_x,
                                                                         dy,
                                                                         incy,
                                                                         stride_y,
                                                                         dc,
                                                                         ds,
                                                                         batch_count),
                                  rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR((rocblas_rot_strided_batched<T, U, V>(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count)));
        return;
    }

    size_t size_x = N * size_t(incx) + size_t(stride_x) * size_t(batch_count - 1);
    size_t size_y = N * size_t(incy) + size_t(stride_y) * size_t(batch_count - 1);

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<U> dc(1);
    device_vector<V> ds(1);
    if(!dx || !dy || !dc || !ds)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<U> hc(1);
    host_vector<V> hs(1);
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, incx, stride_x, batch_count);
    rocblas_init<T>(hy, 1, N, incy, stride_y, batch_count);

    // Random alpha (0 - 10)
    host_vector<rocblas_int> alpha(1);
    rocblas_init<rocblas_int>(alpha, 1, 1, 1);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<T> cx = hx;
    host_vector<T> cy = hy;
    // cblas_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    cpu_time_used = get_time_us();
    for(int b = 0; b < batch_count; b++)
    {
        cblas_rot<T, U, V>(N, cx + b * stride_x, incx, cy + b * stride_y, incy, hc, hs);
    }
    cpu_time_used = get_time_us() - cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // Test rocblas_pointer_mode_host
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR((rocblas_rot_strided_batched<T, U, V>(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, hc, hs, batch_count)));
            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, batch_count, incx, stride_x, cx, rx);
                unit_check_general<T>(1, N, batch_count, incy, stride_y, cy, ry);
            }
            if(arg.norm_check)
            {
                norm_error_host_x
                    = norm_check_general<T>('F', 1, N, incx, stride_x, batch_count, cx, rx);
                norm_error_host_y
                    = norm_check_general<T>('F', 1, N, incy, stride_x, batch_count, cy, ry);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR((rocblas_rot_strided_batched<T, U, V>(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count)));
            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, batch_count, incx, stride_x, cx, rx);
                unit_check_general<T>(1, N, batch_count, incy, stride_y, cy, ry);
            }
            if(arg.norm_check)
            {
                norm_error_device_x
                    = norm_check_general<T>('F', 1, N, incx, stride_x, batch_count, cx, rx);
                norm_error_device_y
                    = norm_check_general<T>('F', 1, N, incy, stride_y, batch_count, cy, ry);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rot_strided_batched<T, U, V>(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, hc, hs, batch_count);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rot_strided_batched<T, U, V>(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, hc, hs, batch_count);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,rocblas(us),cpu(us)";
        if(arg.norm_check)
            std::cout
                << ",norm_error_host_x,norm_error_host_y,norm_error_device_x,norm_error_device_y";
        std::cout << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << gpu_time_used << ","
                  << cpu_time_used;
        if(arg.norm_check)
            std::cout << ',' << norm_error_host_x << ',' << norm_error_host_y << ","
                      << norm_error_device_x << "," << norm_error_device_y;
        std::cout << std::endl;
    }
}
