/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "handle.hpp"
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

template <typename Ti, typename To, typename Tc, typename To_hpa>
void reference_gemm_ext2(rocblas_int    M,
                         rocblas_int    N,
                         rocblas_int    K,
                         Tc             alpha,
                         const Ti*      A,
                         rocblas_stride row_stride_a,
                         rocblas_stride col_stride_a,
                         const Ti*      B,
                         rocblas_stride row_stride_b,
                         rocblas_stride col_stride_b,
                         Tc             beta,
                         const To*      C,
                         rocblas_stride row_stride_c,
                         rocblas_stride col_stride_c,
                         To_hpa*        D,
                         rocblas_stride row_stride_d,
                         rocblas_stride col_stride_d)
{
    for(rocblas_int row = 0; row < M; row++)
        for(rocblas_int col = 0; col < N; col++)
        {
            Tc t(0);
            if(alpha)
                for(rocblas_int k = 0; k < K; k++)
                    t += Tc(A[row * row_stride_a + k * col_stride_a])
                         * Tc(B[k * row_stride_b + col * col_stride_b]);
            D[row * row_stride_d + col * col_stride_d]
                = beta ? beta * C[row * row_stride_c + col * col_stride_c] + alpha * t : alpha * t;
        }
}

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_ext2_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_ext2_fn = arg.fortran ? rocblas_gemm_ext2_fortran : rocblas_gemm_ext2;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 100;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;
        const rocblas_int ldd = 100;

        const rocblas_stride row_stride_a = 1, col_stride_a = lda;
        const rocblas_stride row_stride_b = 1, col_stride_b = ldb;
        const rocblas_stride row_stride_c = 0, col_stride_c = ldc;
        const rocblas_stride row_stride_d = 1, col_stride_d = ldd;

        const rocblas_datatype a_type       = rocblas_datatype_f32_r;
        const rocblas_datatype b_type       = rocblas_datatype_f32_r;
        const rocblas_datatype c_type       = rocblas_datatype_f32_r;
        const rocblas_datatype d_type       = rocblas_datatype_f32_r;
        const rocblas_datatype compute_type = rocblas_datatype_f32_r;

        const float alpha_float = 1.0;
        const float beta_float  = 1.0;

        const rocblas_gemm_algo algo      = rocblas_gemm_algo_standard;
        static const size_t     safe_size = 100;

        int32_t     solution_index = 0;
        rocblas_int flags          = 0;

        rocblas_local_handle handle(arg.atomics_mode);
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // allocate memory on device
        device_vector<float> dA(safe_size);
        device_vector<float> dB(safe_size);
        device_vector<float> dC(safe_size);
        device_vector<float> dD(safe_size);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha_float,
                                                   nullptr,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   &beta_float,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha_float,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   nullptr,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   &beta_float,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha_float,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   &beta_float,
                                                   nullptr,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha_float,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   &beta_float,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   nullptr,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   nullptr,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   &beta_float,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha_float,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   nullptr,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(nullptr,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha_float,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   &beta_float,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_handle);
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_ext2(const Arguments& arg)
{
    auto rocblas_gemm_ext2_fn = arg.fortran ? rocblas_gemm_ext2_fortran : rocblas_gemm_ext2;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool nantest = rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai);
    if(!std::is_same<To, float>{} && !std::is_same<To, double>{}
       && !std::is_same<To, rocblas_half>{} && !is_complex<To> && nantest)
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    rocblas_local_handle handle(arg.atomics_mode);

    auto transA = char2rocblas_operation(arg.transA);
    auto transB = char2rocblas_operation(arg.transB);
    auto M = arg.M, N = arg.N, K = arg.K;
    auto lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto A_row = transA == rocblas_operation_none ? M : K;
    auto A_col = transA == rocblas_operation_none ? K : M;
    auto B_row = transB == rocblas_operation_none ? K : N;
    auto B_col = transB == rocblas_operation_none ? N : K;

    // Row and column strides are based on transpose and leading dimensions
    rocblas_stride row_stride_a = transA == rocblas_operation_none ? 1 : lda;
    rocblas_stride col_stride_a = transA == rocblas_operation_none ? lda : 1;
    rocblas_stride row_stride_b = transB == rocblas_operation_none ? 1 : ldb;
    rocblas_stride col_stride_b = transB == rocblas_operation_none ? ldb : 1;
    rocblas_stride row_stride_c = 1;
    rocblas_stride col_stride_c = arg.ldc;
    rocblas_stride row_stride_d = 1;
    rocblas_stride col_stride_d = arg.ldd;

    // For now, rocblas_gemm_ext2 only supports row_stride_c == 0
    // We do not want to flood the Arguments structure with arbitrary strides,
    // So we artificially set row_stride_c = 0 here, leaving the other strides
    // the same as a normal GEMM
    row_stride_c = 0;

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0;

    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   nullptr,
                                                   nullptr,
                                                   arg.a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   nullptr,
                                                   arg.b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   nullptr,
                                                   nullptr,
                                                   arg.c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   nullptr,
                                                   arg.c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   arg.compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_size);
        return;
    }

    const size_t size_A = size_t(col_stride_a) * size_t(A_col);
    const size_t size_B = size_t(col_stride_b) * size_t(B_col);
    const size_t size_C = size_t(col_stride_c) * size_t(N);
    const size_t size_D = size_t(col_stride_d) * size_t(N);

    // allocate memory on device
    device_vector<Ti> dA(size_A);
    device_vector<Ti> dB(size_B);
    device_vector<To> dC(size_C);
    device_vector<To> dD(size_D);
    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);

    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti> hA(size_A);
    host_vector<Ti> hB(size_B);
    host_vector<To> hC(size_C);
    host_vector<To> hC_1(size_C);
    host_vector<To> hC_2(size_C);
    host_vector<To> hD_1(size_D);
    host_vector<To> hD_2(size_D);
    using To_hpa = std::conditional_t<std::is_same<To, rocblas_bfloat16>{}, float, To>;
    host_vector<To_hpa> hD_gold(size_D);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<Ti>(hA, A_row, A_col, lda);
    rocblas_init_alternating_sign<Ti>(hB, B_row, B_col, ldb);
    if(nantest)
        rocblas_init_nan<To>(hC, M, N, ldc);
    else
        rocblas_init<To>(hC, M, N, ldc);
    rocblas_init<To>(hD_1, M, N, ldd);

    if(std::is_same<To, rocblas_half>{} && std::is_same<Tc, float>{})
    {
        // half precision IEEE has max and lowest values 65504 and -65504,
        // float precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
        // the following will overflow to inf in half arithmetic,
        // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
        //
        // set matrix A and matrix B so reduction sum has 65504 * 2 - 65504 * 2
        //
        const rocblas_half ieee_half_near_max(65504.0 - 4.0);
        const rocblas_half positive_two(2.0);
        const rocblas_half negative_two(-2.0);
        if(M >= 2 && N >= 2 && K >= 2)
        {
            if(transA == rocblas_operation_none)
            {
                hA[0]   = Ti(ieee_half_near_max);
                hA[lda] = Ti(ieee_half_near_max);
            }
            else
            {
                hA[0] = Ti(ieee_half_near_max);
                hA[1] = Ti(ieee_half_near_max);
            }
            if(transB == rocblas_operation_none)
            {
                for(int j = 0; j < N; j++)
                {
                    hB[j * ldb]     = j % 2 == 0 ? Ti(positive_two) : Ti(negative_two);
                    hB[1 + j * ldb] = j % 2 == 0 ? Ti(negative_two) : Ti(positive_two);
                }
            }
            else
            {
                for(int j = 0; j < N; j++)
                {
                    hB[j]       = j % 2 == 0 ? Ti(positive_two) : Ti(negative_two);
                    hB[ldb + j] = j % 2 == 0 ? Ti(negative_two) : Ti(positive_two);
                }
            }
        }
    }

    hD_2 = hD_1;

    // copy data from CPU to device
    // if int8 and A not transposed and valid case, pack A
    if(std::is_same<Ti, int8_t>{} && transA == rocblas_operation_none)
    {
        host_vector<Ti> hA_packed(hA);

        rocblas_packInt8(hA_packed, M, K, lda);
        CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }

    // if int8 and B transposed and valid case, pack B
    if(std::is_same<Ti, int8_t>{} && transB != rocblas_operation_none)
    {
        host_vector<Ti> hB_packed(hB);

        rocblas_packInt8(hB_packed, N, K, ldb);
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_packed, sizeof(Ti) * size_B, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Ti) * size_B, hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(To) * size_C, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dD, hD_1, sizeof(To) * size_D, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha_Tc,
                                                 dA,
                                                 arg.a_type,
                                                 row_stride_a,
                                                 col_stride_a,
                                                 dB,
                                                 arg.b_type,
                                                 row_stride_b,
                                                 col_stride_b,
                                                 &h_beta_Tc,
                                                 dC,
                                                 arg.c_type,
                                                 row_stride_c,
                                                 col_stride_c,
                                                 dD,
                                                 arg.d_type,
                                                 row_stride_d,
                                                 col_stride_d,
                                                 arg.compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags));

        CHECK_HIP_ERROR(hipMemcpy(hD_1, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dD, hD_2, sizeof(To) * size_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                 M,
                                                 N,
                                                 K,
                                                 d_alpha_Tc,
                                                 dA,
                                                 arg.a_type,
                                                 row_stride_a,
                                                 col_stride_a,
                                                 dB,
                                                 arg.b_type,
                                                 row_stride_b,
                                                 col_stride_b,
                                                 d_beta_Tc,
                                                 dC,
                                                 arg.c_type,
                                                 row_stride_c,
                                                 col_stride_c,
                                                 dD,
                                                 arg.d_type,
                                                 row_stride_d,
                                                 col_stride_d,
                                                 arg.compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags));

        CHECK_HIP_ERROR(hipMemcpy(hD_2, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        reference_gemm_ext2<Ti, To, Tc, To_hpa>(M,
                                                N,
                                                K,
                                                h_alpha_Tc,
                                                hA,
                                                row_stride_a,
                                                col_stride_a,
                                                hB,
                                                row_stride_b,
                                                col_stride_b,
                                                h_beta_Tc,
                                                hC,
                                                row_stride_c,
                                                col_stride_c,
                                                hD_gold,
                                                row_stride_d,
                                                col_stride_d);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<To>(M, N, K) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1);
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1));
            auto err2     = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_2));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                     M,
                                                     N,
                                                     K,
                                                     &h_alpha_Tc,
                                                     dA,
                                                     arg.a_type,
                                                     row_stride_a,
                                                     col_stride_a,
                                                     dB,
                                                     arg.b_type,
                                                     row_stride_b,
                                                     col_stride_b,
                                                     &h_beta_Tc,
                                                     dC,
                                                     arg.c_type,
                                                     row_stride_c,
                                                     col_stride_c,
                                                     arg.c_noalias_d ? dD : dC,
                                                     arg.c_noalias_d ? arg.d_type : arg.c_type,
                                                     arg.c_noalias_d ? row_stride_d : row_stride_c,
                                                     arg.c_noalias_d ? col_stride_d : col_stride_c,
                                                     arg.compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_ext2_fn(handle,
                                 M,
                                 N,
                                 K,
                                 &h_alpha_Tc,
                                 dA,
                                 arg.a_type,
                                 row_stride_a,
                                 col_stride_a,
                                 dB,
                                 arg.b_type,
                                 row_stride_b,
                                 col_stride_b,
                                 &h_beta_Tc,
                                 dC,
                                 arg.c_type,
                                 row_stride_c,
                                 col_stride_c,
                                 arg.c_noalias_d ? dD : dC,
                                 arg.c_noalias_d ? arg.d_type : arg.c_type,
                                 arg.c_noalias_d ? row_stride_d : row_stride_c,
                                 arg.c_noalias_d ? col_stride_d : row_stride_d,
                                 arg.compute_type,
                                 algo,
                                 solution_index,
                                 flags);
        }
        gpu_time_used  = get_time_us_sync(stream) - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "M,N,K,alpha,row_stride_a,col_stride_a,row_stride_b,col_stride_b,beta,row_"
                        "stride_c,col_stride_c,row_stride_d,col_stride_d,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            rocblas_cout << ",CPU-Gflops(us),norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << M << "," << N << "," << K << "," << arg.alpha << "," << row_stride_a << ","
                     << col_stride_a << "," << row_stride_b << "," << col_stride_b << ","
                     << arg.beta << "," << row_stride_c << "," << col_stride_c << ","
                     << row_stride_d << "," << col_stride_d << "," << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        rocblas_cout << std::endl;
    }
}
