/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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

#define DEBUG_PRINT 0

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_gemm_batched_ex_fn
        = FORTRAN ? rocblas_gemm_batched_ex_fortran : rocblas_gemm_batched_ex;

    const rocblas_int M = 100;
    const rocblas_int N = 100;
    const rocblas_int K = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;
    const rocblas_int ldd = 100;

    const rocblas_int batch_count = 1;

    rocblas_datatype a_type       = rocblas_datatype_f32_r;
    rocblas_datatype b_type       = rocblas_datatype_f32_r;
    rocblas_datatype c_type       = rocblas_datatype_f32_r;
    rocblas_datatype d_type       = rocblas_datatype_f32_r;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    const float alpha_float = 1.0;
    const float beta_float  = 1.0;

    rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
    int32_t           solution_index = 0;
    rocblas_int       flags          = 0;

    const size_t safe_size = 100;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<float> dA(safe_size);
    device_vector<float> dB(safe_size);
    device_vector<float> dC(safe_size);
    device_vector<float> dD(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha_float,
                                                     nullptr,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     &beta_float,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha_float,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     nullptr,
                                                     b_type,
                                                     ldb,
                                                     &beta_float,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha_float,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     &beta_float,
                                                     nullptr,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha_float,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     &beta_float,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     nullptr,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     nullptr,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     &beta_float,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha_float,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     nullptr,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(nullptr,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha_float,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     &beta_float,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     batch_count,
                                                     compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags),
                          rocblas_status_invalid_handle);
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_gemm_batched_ex_fn
        = FORTRAN ? rocblas_gemm_batched_ex_fortran : rocblas_gemm_batched_ex;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool nantest    = rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai);
    Tc   h_alpha_Tc = arg.get_alpha<Tc>();
    Tc   h_beta_Tc  = arg.get_beta<Tc>();

    double               gpu_time_used, cpu_time_used;
    double               rocblas_gflops, cblas_gflops;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle;
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row       = transA == rocblas_operation_none ? M : K;
    auto                 A_col       = transA == rocblas_operation_none ? K : M;
    auto                 B_row       = transB == rocblas_operation_none ? K : N;
    auto                 B_col       = transB == rocblas_operation_none ? N : K;
    auto                 batch_count = arg.batch_count;

    // Quick-return or error sizes
    // Note: K==0 is not an early exit, since we still must multiply C by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;
    bool int8_invalid = (std::is_same<Ti, int8_t>{}
                         && (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0)));

    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         &h_alpha_Tc,
                                                         nullptr,
                                                         arg.a_type,
                                                         lda,
                                                         nullptr,
                                                         arg.b_type,
                                                         ldb,
                                                         nullptr,
                                                         nullptr,
                                                         arg.c_type,
                                                         ldc,
                                                         nullptr,
                                                         arg.d_type,
                                                         ldd,
                                                         batch_count,
                                                         arg.compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }
    if(int8_invalid)
    {
        // This check is currently done below the invalid_pointer checks, so we can't pass in nullptrs.
        device_batch_vector<Ti> dA(1, 1, 1);
        device_batch_vector<Ti> dB(1, 1, 1);
        device_batch_vector<To> dC(1, 1, 1);
        device_batch_vector<To> dD(1, 1, 1);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         &h_alpha_Tc,
                                                         dA.ptr_on_device(),
                                                         arg.a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         arg.b_type,
                                                         ldb,
                                                         &h_beta_Tc,
                                                         dC.ptr_on_device(),
                                                         arg.c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         arg.d_type,
                                                         ldd,
                                                         batch_count,
                                                         arg.compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_size);
        return;
    }

    size_t size_one_a
        = transA == rocblas_operation_none ? size_t(K) * size_t(lda) : size_t(M) * size_t(lda);
    size_t size_one_b
        = transB == rocblas_operation_none ? size_t(N) * size_t(ldb) : size_t(K) * size_t(ldb);
    size_t size_one_c = N * ldc;
    size_t size_one_d = N * ldd;
    size_t size_a     = size_one_a;
    size_t size_b     = size_one_b;
    size_t size_c     = size_one_c;
    size_t size_d     = size_one_d;

    // allocate memory on device
    device_batch_vector<Ti> dA(size_a, 1, batch_count);
    device_batch_vector<Ti> dB(size_b, 1, batch_count);
    device_batch_vector<To> dC(size_c, 1, batch_count);
    device_batch_vector<To> dD(size_d, 1, batch_count);
    device_vector<Tc>       d_alpha_Tc(1);
    device_vector<Tc>       d_beta_Tc(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    using To_hpa = std::conditional_t<std::is_same<To, rocblas_bfloat16>{}, float, To>;
    host_batch_vector<Ti>     hA(size_a, 1, batch_count);
    host_batch_vector<Ti>     hB(size_b, 1, batch_count);
    host_batch_vector<To>     hC(size_c, 1, batch_count);
    host_batch_vector<To>     hD_1(size_d, 1, batch_count);
    host_batch_vector<To>     hD_2(size_d, 1, batch_count);
    host_batch_vector<To_hpa> hD_gold(size_d, 1, batch_count);

    // Initial Data on CPU
    rocblas_seedrand();
    for(int b = 0; b < batch_count; b++)
    {
        rocblas_init<Ti>(hA[b], A_row, A_col, lda);
        rocblas_init_alternating_sign<Ti>(hB[b], B_row, B_col, ldb);
        if(nantest)
            rocblas_init_nan<To>(hC[b], M, N, ldc);
        else
            rocblas_init<To>(hC[b], M, N, ldc);
        rocblas_init<To>(hD_1[b], M, N, ldd);
    }

    hD_2.copy_from(hD_1);
    for(int b = 0; b < batch_count; b++)
    {
        for(size_t i = 0; i < size_d; i++)
        {
            hD_gold[b][i] = hD_1[b][i];
        }
    }

#if 0 // Copied from testing_gemm_ex.hpp
    if(std::is_same<To, rocblas_half>{} && std::is_same<Tc, float>{})
    {
        // half precision IEEE has max and lowest values 65504 and -65504,
        // foat precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
        // the following will overflow to inf in half arithmetic,
        // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
        //
        // set matrix A and matrix B upper left block to values below to cause
        // inf overflow with 16 bit arithmetic, but no overflow for 32 bit arithmetic
        //
        // 65500 65500             2   -2
        // 65500 65500            -2    2
        //
        const rocblas_half ieee_half_near_max(65504.0 - 4.0);
        const rocblas_half positive_two      (2.0);
        const rocblas_half negative_two      (-2.0);
        if(M >= 2 && N >= 2 && K >= 2)
        {
            hA[0]       = ieee_half_near_max;
            hA[1]       = ieee_half_near_max;
            hA[lda]     = ieee_half_near_max;
            hA[lda + 1] = ieee_half_near_max;
            hB[0]       = positive_two;
            hB[1]       = negative_two;
            hB[ldb]     = negative_two;
            hB[ldb + 1] = positive_two;
        }
    }
#endif

    // copy data from CPU to device
    if(std::is_same<Ti, int8_t>{} && transA == rocblas_operation_none)
    {
        host_batch_vector<Ti> hA_packed(size_a, 1, batch_count);
        hA_packed.copy_from(hA);

        for(int b = 0; b < batch_count; b++)
            rocblas_packInt8(hA_packed[b], hA[b], M, K, lda);

        CHECK_HIP_ERROR(dA.transfer_from(hA_packed));
    }
    else
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }

    if(std::is_same<Ti, int8_t>{} && transB != rocblas_operation_none)
    {
        host_batch_vector<Ti> hB_packed(size_b, 1, batch_count);
        hB_packed.copy_from(hB);

        for(int b = 0; b < batch_count; b++)
            rocblas_packInt8(hB_packed[b], hB[b], N, K, ldb);

        CHECK_HIP_ERROR(dB.transfer_from(hB_packed));
    }
    else
    {
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dD.transfer_from(hD_1));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha_Tc,
                                                       dA.ptr_on_device(),
                                                       arg.a_type,
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       arg.b_type,
                                                       ldb,
                                                       &h_beta_Tc,
                                                       dC.ptr_on_device(),
                                                       arg.c_type,
                                                       ldc,
                                                       dD.ptr_on_device(),
                                                       arg.d_type,
                                                       ldd,
                                                       batch_count,
                                                       arg.compute_type,
                                                       algo,
                                                       solution_index,
                                                       flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_1.transfer_from(dD));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dD.transfer_from(hD_2));

        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       d_alpha_Tc,
                                                       dA.ptr_on_device(),
                                                       arg.a_type,
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       arg.b_type,
                                                       ldb,
                                                       d_beta_Tc,
                                                       dC.ptr_on_device(),
                                                       arg.c_type,
                                                       ldc,
                                                       dD.ptr_on_device(),
                                                       arg.d_type,
                                                       ldd,
                                                       batch_count,
                                                       arg.compute_type,
                                                       algo,
                                                       solution_index,
                                                       flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_2.transfer_from(dD));

        // CPU BLAS
        // copy C matrix into D matrix
        if(batch_count > 0 && N > 0 && M > 0)
            for(int b = 0; b < batch_count; b++)
                for(int i2 = 0; i2 < N; i2++)
                    for(int i1 = 0; i1 < M; i1++)
                    {
                        hD_gold[b][i1 + (i2 * ldd)] = hC[b][i1 + (i2 * ldc)];
                    }
        cpu_time_used = get_time_us();

        for(rocblas_int b = 0; b < batch_count; b++)
        {
            cblas_gemm<Ti, To_hpa>(transA,
                                   transB,
                                   M,
                                   N,
                                   K,
                                   h_alpha_Tc,
                                   hA[b],
                                   lda,
                                   hB[b],
                                   ldb,
                                   h_beta_Tc,
                                   hD_gold[b],
                                   ldd);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<To>(M, N, K) * batch_count / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count, tol);
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count);
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            auto err1
                = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1, batch_count));
            auto err2
                = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_2, batch_count));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dD.transfer_from(hD_1));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(
                handle,
                transA,
                transB,
                M,
                N,
                K,
                &h_alpha_Tc,
                dA.ptr_on_device(),
                arg.a_type,
                lda,
                dB.ptr_on_device(),
                arg.b_type,
                ldb,
                &h_beta_Tc,
                dC.ptr_on_device(),
                arg.c_type,
                ldc,
                arg.c_noalias_d ? dD.ptr_on_device() : dC.ptr_on_device(),
                arg.c_noalias_d ? arg.d_type : arg.c_type,
                arg.c_noalias_d ? ldd : ldc,
                batch_count,
                arg.compute_type,
                algo,
                solution_index,
                flags));
        }

        int number_hot_calls = arg.iters;
        gpu_time_used        = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_batched_ex_fn(handle,
                                       transA,
                                       transB,
                                       M,
                                       N,
                                       K,
                                       &h_alpha_Tc,
                                       dA.ptr_on_device(),
                                       arg.a_type,
                                       lda,
                                       dB.ptr_on_device(),
                                       arg.b_type,
                                       ldb,
                                       &h_beta_Tc,
                                       dC.ptr_on_device(),
                                       arg.c_type,
                                       ldc,
                                       arg.c_noalias_d ? dD.ptr_on_device() : dC.ptr_on_device(),
                                       arg.c_noalias_d ? arg.d_type : arg.c_type,
                                       arg.c_noalias_d ? ldd : ldc,
                                       batch_count,
                                       arg.compute_type,
                                       algo,
                                       solution_index,
                                       flags);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = gemm_gflop_count<To>(M, N, K) * batch_count * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,ldd,"
                        "batch_count,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            rocblas_cout << ",CPU-Gflops(us),norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB)
                     << "," << M << "," << N << "," << K << "," << arg.alpha << "," << lda << ","
                     << ldb << "," << arg.beta << "," << ldc << "," << ldd << "," << batch_count
                     << "," << rocblas_gflops << "," << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        rocblas_cout << std::endl;
    }
}
