/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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

template <typename T>
void testing_rotm_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_rotm_fn = FORTRAN ? rocblas_rotm<T, true> : rocblas_rotm<T, false>;

    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    static const size_t safe_size = 100;

    rocblas_local_handle handle;
    device_vector<T>     dx(safe_size);
    device_vector<T>     dy(safe_size);
    device_vector<T>     dparam(5);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(nullptr, N, dx, incx, dy, incy, dparam),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(handle, N, nullptr, incx, dy, incy, dparam),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(handle, N, dx, incx, nullptr, incy, dparam),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_rotm_fn(handle, N, dx, incx, dy, incy, nullptr),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotm(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_rotm_fn = FORTRAN ? rocblas_rotm<T, true> : rocblas_rotm<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    rocblas_local_handle handle;
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;
    const T rel_error          = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_rotm_fn(handle, N, nullptr, incx, nullptr, incy, nullptr));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> dparam(5);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hdata(4);
    host_vector<T> hparam(5);
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);
    rocblas_init<T>(hdata, 1, 4, 1);

    // CPU BLAS reference data
    cblas_rotmg<T>(&hdata[0], &hdata[1], &hdata[2], &hdata[3], hparam);
    const int FLAG_COUNT        = 4;
    const T   FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};
    for(int i = 0; i < FLAG_COUNT; ++i)
    {
        hparam[0]         = FLAGS[i];
        host_vector<T> cx = hx;
        host_vector<T> cy = hy;
        cpu_time_used     = get_time_us();
        cblas_rotm<T>(N, cx, incx, cy, incy, hparam);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check || arg.norm_check)
        {
            // Test rocblas_pointer_mode_host
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                CHECK_ROCBLAS_ERROR(rocblas_rotm_fn(handle, N, dx, incx, dy, incy, hparam));
                host_vector<T> rx(size_x);
                host_vector<T> ry(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
                if(arg.unit_check)
                {
                    near_check_general<T>(1, N, abs_incx, cx, rx, rel_error);
                    near_check_general<T>(1, N, abs_incy, cy, ry, rel_error);
                }
                if(arg.norm_check)
                {
                    norm_error_host_x = norm_check_general<T>('F', 1, N, abs_incx, cx, rx);
                    norm_error_host_y = norm_check_general<T>('F', 1, N, abs_incy, cy, ry);
                }
            }

            // Test rocblas_pointer_mode_device
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dparam, hparam, sizeof(T) * 5, hipMemcpyHostToDevice));
                CHECK_ROCBLAS_ERROR(rocblas_rotm_fn(handle, N, dx, incx, dy, incy, dparam));
                host_vector<T> rx(size_x);
                host_vector<T> ry(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
                if(arg.unit_check)
                {
                    near_check_general<T>(1, N, abs_incx, cx, rx, rel_error);
                    near_check_general<T>(1, N, abs_incy, cy, ry, rel_error);
                }
                if(arg.norm_check)
                {
                    norm_error_device_x = norm_check_general<T>('F', 1, N, abs_incx, cx, rx);
                    norm_error_device_y = norm_check_general<T>('F', 1, N, abs_incy, cy, ry);
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotm_fn(handle, N, dx, incx, dy, incy, hparam);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotm_fn(handle, N, dx, incx, dy, incy, hparam);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "N,incx,incy,rocblas(us),cpu(us)";
        if(arg.norm_check)
            rocblas_cout
                << ",norm_error_host_x,norm_error_host_y,norm_error_device_x,norm_error_device_y";
        rocblas_cout << std::endl;
        rocblas_cout << N << "," << incx << "," << incy << "," << gpu_time_used << ","
                     << cpu_time_used;
        if(arg.norm_check)
            rocblas_cout << ',' << norm_error_host_x << ',' << norm_error_host_y << ","
                         << norm_error_device_x << "," << norm_error_device_y;
        rocblas_cout << std::endl;
    }
}
