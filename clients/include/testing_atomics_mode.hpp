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

// Check to see if rocblas_set_atomics_mode is working. This is done by:
// - Calling rocblas_set_atomics_mode to not allow atomics
// - Calling rocblas_sgemm for a size that would normally use a Tensile
//   kernel with GlobalSplitU > 1
// - Initializing matrices with random rational numbers, and checking
//   that the rocblas_sgemm is deterministic
// - If atomics are allowed in this case, the result is not deterministic.

template <typename T>
void testing_atomics_mode(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gemm_fn = FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    double               rocblas_gflops, cblas_gflops;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              N,
                                              K,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              nullptr,
                                              ldc),
                              rocblas_status_invalid_size);

        return;
    }

    const auto size_A = size_t(lda) * size_t(A_col);
    const auto size_B = size_t(ldb) * size_t(B_col);
    const auto size_C = size_t(ldc) * size_t(N);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_gold(size_C);
    host_vector<T> hC_input(size_C);

    rocblas_seedrand();
    rocblas_init_hpl<T>(hA, A_row, A_col, lda);
    rocblas_init_hpl<T>(hB, B_row, B_col, ldb);
    rocblas_init_hpl<T>(hC_input, M, N, ldc);

    // ROCBLAS rocblas_pointer_mode_host
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));

    //  From kernel selection file /library/src/blas3/Tensile/Logic/asm_full/vega20_Cijk_Ailk_Bljk_SB.yaml
    //  we know that for gchArch == 906 and [m,n,batch_count,k] = [1024, 16, 1, 500000] an
    //  algorithm with globalSplitU == 32  will be called. For this architecture and size result
    //  should not be deterministic.
    //  If this test fails, check to see if kernel selection file listed above has changed.

    std::string arch_name = rocblas_get_arch_name();
    if(arch_name == "gfx906")
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_allowed));

        // calculate reference result
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_input, sizeof(T) * size_C, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_gold, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // verify arbitary number of tests are not deterministic
        double err1                     = 0;
        int    arbitary_number_of_tests = 10;
        for(int i = 0; i < arbitary_number_of_tests; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(dC, hC_input, sizeof(T) * size_C, hipMemcpyHostToDevice));

            CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

            CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

            // compare hC_1 and hC_gold
            err1 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
            if(err1 != 0)
                break;
        }
        EXPECT_NE(err1, 0);
    }

    //  Verify that result is deterministic for size [m,n,batch_count,k] = [1024, 16, 1, 500000].
    //  We know that for this size gcnArch == 906 will use globalSplitU == 32. For other architecures
    //  we suspect globalSplitU > 32 will be used
    CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed));

    // calculate reference result
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_input, sizeof(T) * size_C, hipMemcpyHostToDevice));

    CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
        handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

    CHECK_HIP_ERROR(hipMemcpy(hC_gold, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

    // verify arbitary number of tests are deterministic
    double err2                     = 0;
    int    arbitary_number_of_tests = 10;
    for(int i = 0; i < arbitary_number_of_tests; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_input, sizeof(T) * size_C, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // compare hC_1 and hC_gold
        err2 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
        if(err2 != 0)
            break;
    }

    EXPECT_EQ(err2, 0);
}
