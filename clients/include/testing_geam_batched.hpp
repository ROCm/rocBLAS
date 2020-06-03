/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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

/* ============================================================================================ */

template <typename T>
void testing_geam_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_geam_batched_fn
        = FORTRAN ? rocblas_geam_batched<T, true> : rocblas_geam_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const rocblas_int batch_count = 5;

    const T h_alpha = 1.0;
    const T h_beta  = 1.0;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_local_handle handle;

    size_t size_A = size_t(lda) * (transA == rocblas_operation_none ? N : M);
    size_t size_B = size_t(ldb) * (transB == rocblas_operation_none ? N : M);
    size_t size_C = size_t(lda) * N;

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dB(size_B, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  nullptr,
                                                  lda,
                                                  &h_beta,
                                                  dB,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  dA,
                                                  lda,
                                                  &h_beta,
                                                  nullptr,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  dA,
                                                  lda,
                                                  &h_beta,
                                                  dB,
                                                  ldb,
                                                  nullptr,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam_batched_fn(
            handle, transA, transB, M, N, nullptr, dA, lda, &h_beta, dB, ldb, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  dA,
                                                  lda,
                                                  nullptr,
                                                  dB,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(nullptr,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  dA,
                                                  lda,
                                                  &h_beta,
                                                  dB,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_geam_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_geam_batched_fn
        = FORTRAN ? rocblas_geam_batched<T, true> : rocblas_geam_batched<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int lda         = arg.lda;
    rocblas_int ldb         = arg.ldb;
    rocblas_int ldc         = arg.ldc;
    rocblas_int batch_count = arg.batch_count;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    rocblas_int A_row, A_col, B_row, B_col;
    rocblas_int inc1_A, inc2_A, inc1_B, inc2_B;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error_1 = std::numeric_limits<T>::max();
    T rocblas_error_2 = std::numeric_limits<T>::max();
    T rocblas_error   = std::numeric_limits<T>::max();

    rocblas_local_handle handle;

    if(transA == rocblas_operation_none)
    {
        A_row  = M;
        A_col  = N;
        inc1_A = 1;
        inc2_A = lda;
    }
    else
    {
        A_row  = N;
        A_col  = M;
        inc1_A = lda;
        inc2_A = 1;
    }
    if(transB == rocblas_operation_none)
    {
        B_row  = M;
        B_col  = N;
        inc1_B = 1;
        inc2_B = ldb;
    }
    else
    {
        B_row  = N;
        B_col  = M;
        inc1_B = ldb;
        inc2_B = 1;
    }

    size_t size_A = size_t(lda) * size_t(A_col);
    size_t size_B = size_t(ldb) * size_t(B_col);
    size_t size_C = size_t(ldc) * size_t(N);

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                      transA,
                                                      transB,
                                                      M,
                                                      N,
                                                      nullptr,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      nullptr,
                                                      ldb,
                                                      nullptr,
                                                      ldc,
                                                      batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>       h_alpha(1);
    host_vector<T>       h_beta(1);
    host_batch_vector<T> hA(size_A, 1, batch_count), hA_copy(size_A, 1, batch_count);
    host_batch_vector<T> hB(size_B, 1, batch_count), hB_copy(size_B, 1, batch_count);
    host_batch_vector<T> hC_1(size_C, 1, batch_count);
    host_batch_vector<T> hC_2(size_C, 1, batch_count);
    host_batch_vector<T> hC_gold(size_C, 1, batch_count);
    CHECK_HIP_ERROR(h_alpha.memcheck());
    CHECK_HIP_ERROR(h_beta.memcheck());
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;
    rocblas_seedrand();
    rocblas_init<T>(hA);
    rocblas_init<T>(hB);

    hA_copy.copy_from(hA);
    hB_copy.copy_from(hB);

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dB(size_B, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
    device_batch_vector<T> dC_in_place(size_C, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
    CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    &alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    &beta,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        rocblas_init<T>(hC_2);
        CHECK_HIP_ERROR(dC.transfer_from(hC_2));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR(rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    d_alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    d_beta,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        // reference calculation for golden result
        cpu_time_used = get_time_us();

        for(int b = 0; b < batch_count; b++)
        {
            auto hA_copy_p = hA_copy[b];
            auto hB_copy_p = hB_copy[b];
            auto hC_gold_p = hC_gold[b];

            cblas_geam(transA,
                       transB,
                       M,
                       N,
                       (T*)h_alpha,
                       (T*)hA_copy_p,
                       lda,
                       (T*)h_beta,
                       (T*)hB_copy_p,
                       ldb,
                       (T*)hC_gold_p,
                       ldc);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = geam_gflop_count<T>(M, N) * batch_count / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
            unit_check_general<T>(M, N, ldc, hC_gold, hC_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2, batch_count);
        }

        // inplace check for dC == dA
        {
            if((lda == ldc) && (transA == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hA));
            //
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    &alpha,
                                                    dC_in_place.ptr_on_device(),
                                                    lda,
                                                    &beta,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    dC_in_place.ptr_on_device(),
                                                    ldc,
                                                    batch_count);

            if(lda != ldc || transA != rocblas_operation_none)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_HIP_ERROR(hC_1.transfer_from(dC));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(dA.transfer_from(hA));

                // reference calculation
                for(int b = 0; b < batch_count; b++)
                {
                    auto hA_copy_p = hA_copy[b];
                    auto hB_copy_p = hB_copy[b];
                    auto hC_gold_p = hC_gold[b];

                    cblas_geam(transA,
                               transB,
                               M,
                               N,
                               (T*)h_alpha,
                               (T*)hA_copy_p,
                               lda,
                               (T*)h_beta,
                               (T*)hB_copy_p,
                               ldb,
                               (T*)hC_gold_p,
                               ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error
                        = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
                }
            }
        }

        // inplace check for dC == dB
        {
            if((ldb == ldc) && (transB == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hB));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    &alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    &beta,
                                                    dC_in_place.ptr_on_device(),
                                                    ldb,
                                                    dC_in_place.ptr_on_device(),
                                                    ldc,
                                                    batch_count);

            if(ldb != ldc || transB != rocblas_operation_none)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_ROCBLAS_ERROR(status_h);

                CHECK_HIP_ERROR(hC_1.transfer_from(dC_in_place));

                // reference calculation
                for(int b = 0; b < batch_count; b++)
                {
                    auto hA_copy_p = hA_copy[b];
                    auto hB_copy_p = hB_copy[b];
                    auto hC_gold_p = hC_gold[b];

                    cblas_geam(transA,
                               transB,
                               M,
                               N,
                               (T*)h_alpha,
                               (T*)hA_copy_p,
                               lda,
                               (T*)h_beta,
                               (T*)hB_copy_p,
                               ldb,
                               (T*)hC_gold_p,
                               ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error
                        = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
                }
            }
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_geam_batched_fn(handle,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    &alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    &beta,
                                    dB.ptr_on_device(),
                                    ldb,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_geam_batched_fn(handle,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    &alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    &beta,
                                    dB.ptr_on_device(),
                                    ldb,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = geam_gflop_count<T>(M, N) * number_hot_calls * batch_count / gpu_time_used * 1e6;

        rocblas_cout << "transA,transB,M,N,alpha,lda,beta,ldb,ldc,batch_"
                        "count,rocblas-Gflops,us";
        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << ",CPU-Gflops,us,norm_error_ptr_host,norm_error_ptr_dev";
        }
        rocblas_cout << std::endl;

        rocblas_cout << arg.transA << arg.transB << "," << M << "," << N << "," << alpha << ","
                     << lda << "," << beta << "," << ldb << "," << ldc << "," << batch_count << ","
                     << rocblas_gflops << "," << gpu_time_used / number_hot_calls << ",";

        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << cblas_gflops << "," << cpu_time_used << ",";
            rocblas_cout << rocblas_error_1 << "," << rocblas_error_2;
        }
        rocblas_cout << std::endl;
    }
}
