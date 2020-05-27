/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_tpsv_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_tpsv_fn = FORTRAN ? rocblas_tpsv<T, true> : rocblas_tpsv<T, false>;

    const rocblas_int       N      = 100;
    const rocblas_int       incx   = 1;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_fill      uplo   = rocblas_fill_lower;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle;

    size_t size_A = N * size_t(N);
    size_t size_x = N * size_t(incx);

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    //
    // Checks.
    //
    EXPECT_ROCBLAS_STATUS(rocblas_tpsv_fn(handle, rocblas_fill_full, transA, diag, N, dA, dx, incx),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocblas_tpsv_fn(handle, uplo, transA, diag, N, nullptr, dx, incx),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_tpsv_fn(handle, uplo, transA, diag, N, dA, nullptr, incx),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_tpsv_fn(nullptr, uplo, transA, diag, N, dA, dx, incx),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_tpsv(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_tpsv_fn = FORTRAN ? rocblas_tpsv<T, true> : rocblas_tpsv<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || !incx;
    if(invalid_size || !N)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_tpsv_fn(handle, uplo, transA, diag, N, nullptr, nullptr, incx),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t size_A   = size_t(N) * N;
    size_t size_AP  = tri_count(N);
    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = N * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hAP(size_AP);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);
    host_vector<T> my_cpu_x_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error;
    double error_eps_multiplier    = 40.0;
    double residual_eps_multiplier = 20.0;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // allocate memory on device
    device_vector<T> dAP(size_AP);
    device_vector<T> dx_or_b(size_x);
    CHECK_DEVICE_ALLOCATION(dAP.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    rocblas_init<T>(hA, true);

    prepare_triangular_solve((T*)hA, N, (T*)AAT, N, char_uplo);
    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, (T*)hA, N, N);
    }

    rocblas_init<T>(hx, 1, N, abs_incx);
    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, N, hA, N, hb, incx);
    cpu_x_or_b    = hb; // cpuXorB <- B
    hx_or_b_1     = hb;
    hx_or_b_2     = hb;
    my_cpu_x_or_b = hb;

    regular_to_packed(uplo == rocblas_fill_upper, (T*)hA, (T*)hAP, N);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAP.transfer_from(hAP));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;
    double max_res_1 = 0.0;
    double max_res_2 = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_tpsv_fn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx));
        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_2));

        CHECK_ROCBLAS_ERROR(rocblas_tpsv_fn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx));
        CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));

        max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx, hx_or_b_1));
        max_err_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx, hx_or_b_2));

        trsm_err_res_check(max_err_1, N, error_eps_multiplier, eps);
        trsm_err_res_check(max_err_2, N, error_eps_multiplier, eps);

        cblas_trmv<T>(uplo, transA, diag, N, hA, N, hx_or_b_1, incx);
        cblas_trmv<T>(uplo, transA, diag, N, hA, N, hx_or_b_2, incx);
        // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
        //                                                  = hx_or_b - hb
        // res is the one norm of the scaled residual for each column

        max_res_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_1, hb));
        max_res_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_2, hb));

        trsm_err_res_check(max_res_1, N, residual_eps_multiplier, eps);
        trsm_err_res_check(max_res_2, N, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        hx_or_b_1 = cpu_x_or_b;
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_tpsv_fn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx);

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_tpsv_fn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx);

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = tpsv_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = tpsv_gbyte_count<T>(N) / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        if(arg.norm_check)
            cblas_tpsv<T>(uplo, transA, diag, N, hAP, cpu_x_or_b, incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = tpsv_gflop_count<T>(N) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "N,incx,uplo,transA,diag,rocblas-Gflops,rocblas-GB/s,us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << N << ',' << incx << ',' << char_uplo << ',' << char_transA << ','
                     << char_diag << ',' << rocblas_gflops << "," << rocblas_bandwidth << ","
                     << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                         << max_err_2;

        rocblas_cout << std::endl;
    }
}
