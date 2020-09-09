/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */
template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex_bad_arg(const Arguments& arg)
{
    const bool FORTRAN            = arg.fortran;
    auto       rocblas_axpy_ex_fn = FORTRAN ? rocblas_axpy_ex_fortran : rocblas_axpy_ex;

    rocblas_datatype alpha_type     = rocblas_datatype_f32_r;
    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype y_type         = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         incy      = 1;
    static const size_t safe_size = 100;
    Ta                  alpha     = 0.6;

    rocblas_local_handle handle(arg.atomics_mode);
    device_vector<Tx>    dx(safe_size);
    device_vector<Ty>    dy(safe_size);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_ex_fn(
            handle, N, &alpha, alpha_type, nullptr, x_type, incx, dy, y_type, incy, execution_type),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_ex_fn(
            handle, N, &alpha, alpha_type, dx, x_type, incx, nullptr, y_type, incy, execution_type),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_ex_fn(
            handle, N, nullptr, alpha_type, dx, x_type, incx, dy, y_type, incy, execution_type),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_ex_fn(
            nullptr, N, &alpha, alpha_type, dx, x_type, incx, dy, y_type, incy, execution_type),
        rocblas_status_invalid_handle);
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex(const Arguments& arg)
{
    const bool FORTRAN            = arg.fortran;
    auto       rocblas_axpy_ex_fn = FORTRAN ? rocblas_axpy_ex_fortran : rocblas_axpy_ex;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype y_type         = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int          N       = arg.N;
    rocblas_int          incx    = arg.incx;
    rocblas_int          incy    = arg.incy;
    Ta                   h_alpha = arg.get_alpha<Ta>();
    rocblas_local_handle handle(arg.atomics_mode);

    bool special_compute_test = N == 1 && h_alpha == -1.001;

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_ex_fn(handle,
                                               N,
                                               nullptr,
                                               alpha_type,
                                               nullptr,
                                               x_type,
                                               incx,
                                               nullptr,
                                               y_type,
                                               incy,
                                               execution_type));
        return;
    }

    rocblas_int abs_incx = incx > 0 ? incx : -incx;
    rocblas_int abs_incy = incy > 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx>  hx(size_x);
    host_vector<Ty>  hy_1(size_y);
    host_vector<Ty>  hy_2(size_y);
    host_vector<Ty>  hy_gold(size_y);
    host_vector<Tex> hy_gold_ex(size_y);
    host_vector<Tex> hx_ex(size_x);

    // Initial Data on CPU
    // TODO: add NaN testing when roblas_isnan(arg.alpha) returns true.
    rocblas_seedrand();
    rocblas_init(hx, true);
    rocblas_init(hy_1, false);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hy_2    = hy_1;
    hy_gold = hy_1;

    for(size_t i = 0; i < size_y; i++)
        hy_gold_ex[i] = (Tex)hy_gold[i];

    for(size_t i = 0; i < size_x; i++)
        hx_ex[i] = (Tex)hx[i];

    Tex h_alpha_ex = (Tex)h_alpha;

    // allocate memory on device
    device_vector<Tx> dx(size_x);
    device_vector<Ty> dy_1(size_y);
    device_vector<Ty> dy_2(size_y);
    device_vector<Ta> d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // This is to test that we are using the correct
    // compute type (avoiding overflow in this case)
    if(special_compute_test)
    {
        // max half value
        hx[0]   = 65504;
        hy_1[0] = 65504;
        hy_2    = hy_1;
        hy_gold = hy_1;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(Ty) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(Ty) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

        // ROCBLAS pointer mode host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_ex_fn(
            handle, N, &h_alpha, alpha_type, dx, x_type, incx, dy_1, y_type, incy, execution_type));

        // ROCBLAS pointer mode device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_ex_fn(
            handle, N, d_alpha, alpha_type, dx, x_type, incx, dy_2, y_type, incy, execution_type));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        // cblas_axpy<Ta>(N, h_alpha, hx, incx, hy_gold, incy);
        cblas_axpy<Tex>(N, h_alpha_ex, hx_ex, incx, hy_gold_ex, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        for(size_t i = 0; i < size_y; i++)
            hy_gold[i] = (Ty)hy_gold_ex[i];

        if(special_compute_test)
            hy_gold[0] = Tex(h_alpha + 1) * Tex(65504);

        // No accumulation in axpy, hard to check if we're using the right
        // compute_type
        if(arg.unit_check)
        {
            unit_check_general<Ty>(1, N, abs_incy, hy_gold, hy_1);
            unit_check_general<Ty>(1, N, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<Ty>('F', 1, N, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<Ty>('F', 1, N, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_axpy_ex_fn(handle,
                               N,
                               &h_alpha,
                               alpha_type,
                               dx,
                               x_type,
                               incx,
                               dy_1,
                               y_type,
                               incy,
                               execution_type);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy_ex_fn(handle,
                               N,
                               &h_alpha,
                               alpha_type,
                               dx,
                               x_type,
                               incx,
                               dy_1,
                               y_type,
                               incy,
                               execution_type);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy>{}.log_args<Ta>(rocblas_cout,
                                                                   arg,
                                                                   gpu_time_used,
                                                                   axpy_gflop_count<Ta>(N),
                                                                   axpy_gbyte_count<Ta>(N),
                                                                   cpu_time_used,
                                                                   rocblas_error_1,
                                                                   rocblas_error_2);
    }
}
