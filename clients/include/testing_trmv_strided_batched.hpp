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
void testing_trmv_strided_batched_bad_arg(const Arguments& arg)
{
    const rocblas_int       M           = 100;
    const rocblas_int       lda         = 100;
    const rocblas_int       incx        = 1;
    const rocblas_int       batch_count = 1;
    const rocblas_stride    stride_a    = M * lda;
    const rocblas_stride    stride_x    = M;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle;

    size_t size_A = lda * size_t(M);

    host_strided_batch_vector<T> hA(size_A, 1, stride_a, batch_count);
    CHECK_HIP_ERROR(hA.memcheck());

    host_strided_batch_vector<T> hx(M, incx, stride_x, batch_count);
    CHECK_HIP_ERROR(hx.memcheck());

    device_strided_batch_vector<T> dA(size_A, 1, stride_a, batch_count);
    CHECK_HIP_ERROR(dA.memcheck());

    device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    //
    // Checks.
    //
    EXPECT_ROCBLAS_STATUS(
        rocblas_trmv_strided_batched<T>(
            handle, uplo, transA, diag, M, nullptr, lda, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmv_strided_batched<T>(
            handle, uplo, transA, diag, M, dA, lda, stride_a, nullptr, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmv_strided_batched<T>(
            nullptr, uplo, transA, diag, M, dA, lda, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_trmv_strided_batched(const Arguments& arg)
{
    rocblas_int    M = arg.M, lda = arg.lda, incx = arg.incx, batch_count = arg.batch_count;
    rocblas_stride stride_a = arg.stride_a, stride_x = arg.stride_x;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(M < 0 || lda < M || lda < 1 || !incx || batch_count < 0)
    {
        device_strided_batch_vector<T> dA1(10, 1, 10, 2);
        CHECK_HIP_ERROR(dA1.memcheck());
        device_strided_batch_vector<T> dx1(10, 1, 10, 2);
        CHECK_HIP_ERROR(dx1.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_trmv_strided_batched<T>(handle,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              M,
                                                              dA1,
                                                              lda,
                                                              stride_a,
                                                              dx1,
                                                              incx,
                                                              stride_x,
                                                              batch_count),
                              rocblas_status_invalid_size);

        return;
    }

    if(!M || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_trmv_strided_batched<T>(handle,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              M,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              batch_count),
                              rocblas_status_success);
        return;
    }

    size_t size_A   = lda * size_t(M);
    size_t abs_incx = incx >= 0 ? incx : -incx;

    host_strided_batch_vector<T> hA(size_A, 1, stride_a, batch_count);
    CHECK_HIP_ERROR(hA.memcheck());

    host_strided_batch_vector<T> hx(M, incx, stride_x, batch_count);
    CHECK_HIP_ERROR(hx.memcheck());

    host_strided_batch_vector<T> hres(M, incx, stride_x, batch_count);
    CHECK_HIP_ERROR(hres.memcheck());

    device_strided_batch_vector<T> dA(size_A, 1, stride_a, batch_count);
    CHECK_HIP_ERROR(dA.memcheck());

    device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    //
    // Initialize.
    //
    rocblas_init(hA, true);
    rocblas_init(hx);

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used, rocblas_gflops, cblas_gflops, rocblas_bandwidth,
        rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        //
        // GPU BLAS
        //
        CHECK_ROCBLAS_ERROR(rocblas_trmv_strided_batched<T>(
            handle, uplo, transA, diag, M, dA, lda, stride_a, dx, incx, stride_x, batch_count));
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        //
        // CPU BLAS
        //
        {
            cpu_time_used = get_time_us();
            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                cblas_trmv<T>(uplo, transA, diag, M, hA[batch_index], lda, hx[batch_index], incx);
            }

            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = (double(batch_count) * trmv_gflop_count<T>(M)) / cpu_time_used * 1e6;
        }

        //
        // Unit check.
        //
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, abs_incx, stride_x, hx, hres);
        }

        //
        // Norm check.
        //
        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, M, batch_count, abs_incx, stride_x, hx, hres);
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
                rocblas_trmv_strided_batched<T>(handle,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                dA,
                                                lda,
                                                stride_a,
                                                dx,
                                                incx,
                                                stride_x,
                                                batch_count);
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
                rocblas_trmv_strided_batched<T>(handle,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                dA,
                                                lda,
                                                stride_a,
                                                dx,
                                                incx,
                                                stride_x,
                                                batch_count);
            }
            gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;
        }

        //
        // Evaluate performance.
        //
        rocblas_gflops = (double(batch_count) * trmv_gflop_count<T>(M)) / gpu_time_used * 1e6;
        rocblas_bandwidth
            = (double((M * (M + 1)) / 2 + 2 * M) * double(batch_count) * double(sizeof(T)))
              / gpu_time_used * 1e-3;

        //
        // Display.
        //
        std::cout << "M,lda,stride_a,incx,stride_x,batch_count, "
                     "uplo,transA,diag,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error";
        }
        std::cout << std::endl;
        std::cout << M << "," << lda << "," << stride_a << "," << incx << "," << stride_x << ","
                  << batch_count << "," << char_uplo << ',' << char_transA << ',' << char_diag
                  << ',' << rocblas_gflops << "," << rocblas_bandwidth << ",";
        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error;
        }
        std::cout << std::endl;
    }
}
