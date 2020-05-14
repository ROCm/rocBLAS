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
void testing_swap_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_swap_fn = FORTRAN ? rocblas_swap<T, true> : rocblas_swap<T, false>;

    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_swap_fn(handle, N, nullptr, incx, dy, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_fn(handle, N, dx, incx, nullptr, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_fn(nullptr, N, dx, incx, dy, incy),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_swap(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_swap_fn = FORTRAN ? rocblas_swap<T, true> : rocblas_swap<T, false>;

    rocblas_int          N    = arg.N;
    rocblas_int          incx = arg.incx;
    rocblas_int          incy = arg.incy;
    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_swap_fn(handle, N, nullptr, incx, nullptr, incy));
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hx_gold(size_x);
    host_vector<T> hy_gold(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx);
    // make hy different to hx
    for(size_t i = 0; i < N; i++)
    {
        hy[i * abs_incy] = hx[i * abs_incx] + 1.0;
    };

    // swap vector is easy in STL; hy_gold = hx: save a swap in hy_gold which will be output of CPU
    // BLAS
    hx_gold = hx;
    hy_gold = hy;

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_swap_fn(handle, N, dx, incx, dy, incy));
        CHECK_HIP_ERROR(hipMemcpy(hx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_swap<T>(N, hx_gold, incx, hy_gold, incy);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, hx_gold, hx);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_gold, hx);
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap_fn(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_fn(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "N,incx,incy,rocblas-us" << std::endl;
        rocblas_cout << N << "," << incx << "," << incy << "," << gpu_time_used << std::endl;
    }
}
