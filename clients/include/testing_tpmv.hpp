/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
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
void testing_tpmv_bad_arg(const Arguments& arg)
{
    const rocblas_int       M      = 100;
    const rocblas_int       incx   = 1;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_fill      uplo   = rocblas_fill_lower;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle;

    size_t size_A = (M * (M + 1)) / 2;
    size_t size_x = M * size_t(incx);

    host_vector<T> hA((rocblas_int)size_A, (rocblas_int)1);
    CHECK_HIP_ERROR(hA.memcheck());
    host_vector<T> hx((rocblas_int)size_x, (rocblas_int)1);
    CHECK_HIP_ERROR(hx.memcheck());
    device_vector<T> dA(size_A);
    CHECK_HIP_ERROR(dA.memcheck());
    device_vector<T> dx(size_x);
    CHECK_HIP_ERROR(dx.memcheck());

    //
    // Checks.
    //
    EXPECT_ROCBLAS_STATUS(rocblas_tpmv<T>(handle, uplo, transA, diag, M, nullptr, dx, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv<T>(handle, uplo, transA, diag, M, dA, nullptr, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv<T>(nullptr, uplo, transA, diag, M, dA, dx, incx),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_tpmv(const Arguments& arg)
{
    rocblas_int M = arg.M, incx = arg.incx;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill         uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation    transA = char2rocblas_operation(char_transA);
    rocblas_diagonal     diag   = char2rocblas_diagonal(char_diag);
    rocblas_local_handle handle;

    if(M < 0 || !incx)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA1(safe_size);
        CHECK_HIP_ERROR(dA1.memcheck());
        device_vector<T> dx1(safe_size);
        CHECK_HIP_ERROR(dx1.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_tpmv<T>(handle, uplo, transA, diag, M, dA1, dx1, incx),
                              rocblas_status_invalid_size);

        return;
    }

    if(M == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_tpmv<T>(handle, uplo, transA, diag, M, nullptr, nullptr, incx),
            rocblas_status_success);
        return;
    }

    rocblas_int size_A = (M * (M + 1)) / 2;
    rocblas_int size_x, dim_x, abs_incx;
    dim_x    = M;
    abs_incx = incx >= 0 ? incx : -incx;
    size_x   = dim_x * abs_incx;

    host_vector<T> hA(size_A, 1);
    CHECK_HIP_ERROR(hA.memcheck());
    host_vector<T> hx(M, incx);
    CHECK_HIP_ERROR(hx.memcheck());
    device_vector<T> dA(size_A);
    CHECK_HIP_ERROR(dA.memcheck());
    device_vector<T> dx(size_x);
    CHECK_HIP_ERROR(dx.memcheck());
    host_vector<T> hres(M, incx);
    CHECK_HIP_ERROR(hres.memcheck());

    //
    // Initialize.
    //
    rocblas_init(hA, true);
    rocblas_init(hx, false);

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {

        //
        // ROCBLAS
        //
        CHECK_ROCBLAS_ERROR(rocblas_tpmv<T>(handle, uplo, transA, diag, M, dA, dx, incx));
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        //
        // CPU BLAS
        //
        {
            cpu_time_used = get_time_us();
            cblas_tpmv<T>(uplo, transA, diag, M, hA, hx, incx);
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = tpmv_gflop_count<T>(M) / cpu_time_used * 1e6;
        }

        //
        // Unit check.
        //
        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_x, abs_incx, hx, hres);
        }

        //
        // Norm check.
        //
        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, dim_x, abs_incx, hx, hres);
        }
    }

    if(arg.timing)
    {
        //
        // Warmup
        //
        {
            int number_cold_calls = 2;
            for(int iter = 0; iter < number_cold_calls; iter++)
            {
                rocblas_tpmv<T>(handle, uplo, transA, diag, M, dA, dx, incx);
            }
        }

        //
        // Go !
        //
        {
            gpu_time_used        = get_time_us(); // in microseconds
            int number_hot_calls = arg.iters;
            for(int iter = 0; iter < number_hot_calls; iter++)
            {
                rocblas_tpmv<T>(handle, uplo, transA, diag, M, dA, dx, incx);
            }
            gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;
        }

        //
        // Evaluate performance.
        //
        rocblas_gflops    = tpmv_gflop_count<T>(M) / gpu_time_used * 1e6;
        rocblas_bandwidth = (double((M * (M + 1)) / 2) * sizeof(T)) / gpu_time_used * 1e-3;

        //
        // Display.
        //
        std::cout << "M,incx,uplo,transA,diag,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error";
        }
        std::cout << std::endl;
        std::cout << M << "," << incx << "," << incx << "," << incx << "," << incx << ","
                  << char_uplo << ',' << char_transA << ',' << char_diag << ',' << rocblas_gflops
                  << "," << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error;
        }

        std::cout << std::endl;
    }
}
