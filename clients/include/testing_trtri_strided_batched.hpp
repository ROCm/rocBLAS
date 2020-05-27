/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
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
void testing_trtri_strided_batched(const Arguments& arg)
{
    const bool FORTRAN                          = arg.fortran;
    auto       rocblas_trtri_strided_batched_fn = FORTRAN ? rocblas_trtri_strided_batched<T, true>
                                                    : rocblas_trtri_strided_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int lda         = arg.lda;
    rocblas_int batch_count = arg.batch_count;

    rocblas_stride stride_a = lda * N;
    size_t         size_A   = size_t(stride_a) * batch_count;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    // char_uplo = 'U';
    rocblas_fill     uplo = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || batch_count < 0;
    if(invalid_size || N == 0 || batch_count == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_trtri_strided_batched_fn(
                handle, uplo, diag, N, nullptr, lda, stride_a, nullptr, lda, stride_a, batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hB(size_A);
    host_vector<T> hA;
    host_vector<T> hA_2(size_A);
    hA.reserve(size_A);

    // Initial Data on CPU
    rocblas_seedrand();
    host_vector<T> hA_sub(stride_a);

    for(size_t b = 0; b < batch_count; b++)
    {
        rocblas_init_symmetric<T>(hA_sub, N, lda);
        for(size_t i = 0; i < N; i++)
        {
            for(size_t j = 0; j < N; j++)
            {
                hA_sub[i + j * lda] *= 0.01;

                if(j % 2)
                    hA_sub[i + j * lda] *= -1;
                if(uplo == rocblas_fill_lower
                   && j > i) // need to explicitly set unsused side to 0 if using it for temp storage
                    hA_sub[i + j * lda] = 0.0f;
                else if(uplo == rocblas_fill_upper && j < i)
                    hA_sub[i + j * lda] = 0.0f;
                if(i == j)
                {
                    if(diag == rocblas_diagonal_unit)
                        hA_sub[i + j * lda] = 1.0; // need to preprocess matrix for clbas_trtri
                    else
                        hA_sub[i + j * lda] *= 100.0;
                }
            }
        }
        hA.insert(std::end(hA), std::begin(hA_sub), std::end(hA_sub));
    }

    hB = hA;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    device_vector<T> dA(size_A);
    device_vector<T> dinvA(size_A);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched_fn(
        handle, uplo, diag, N, dA, lda, stride_a, dinvA, lda, stride_a, batch_count));

    // Test in place
    CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched_fn(
        handle, uplo, diag, N, dA, lda, stride_a, dA, lda, stride_a, batch_count));

    if(arg.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = batch_count * trtri_gflop_count<T>(N) / gpu_time_used * 1e6;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA, dinvA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hA_2, dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(arg.timing)
        {
            cpu_time_used = get_time_us();
        }

        for(size_t i = 0; i < batch_count; i++)
        {
            rocblas_int info = cblas_trtri<T>(char_uplo, char_diag, N, hB + i * stride_a, lda);
            if(info != 0)
            {
#ifdef GOOGLE_TEST
                FAIL() << "error in cblas_trtri";
#else
                rocblas_cerr << "error in cblas_trtri" << std::endl;
#endif
            }
        }
        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = batch_count * trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

#if 0
        rocblas_print_matrix(hB, hA, N, N, lda, 1);
#endif

        if(arg.unit_check)
        {
            const double rel_error = get_epsilon<T>() * 1000;
            near_check_general<T>(N, N, lda, stride_a, hB, hA, batch_count, rel_error);
            near_check_general<T>(N, N, lda, stride_a, hB, hA_2, batch_count, rel_error);
        }

        if(arg.norm_check)
        {
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error
                    = fmax(rocblas_error,
                           norm_check_symmetric<T>(
                               'F', char_uplo, N, lda, hB + i * stride_a, hA + i * stride_a));
            }
            rocblas_error = 0.0;
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error
                    = fmax(rocblas_error,
                           norm_check_symmetric<T>(
                               'F', char_uplo, N, lda, hB + i * stride_a, hA_2 + i * stride_a));
            }
        }
    } // end of norm_check

    if(arg.timing)
    {
        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "batch, N, lda, rocblas-Gflops (us) ";
        if(arg.norm_check)
        {
            rocblas_cout << "CPU-Gflops(us), norm-error";
        }
        rocblas_cout << std::endl;

        rocblas_cout << batch_count << ',' << N << ',' << lda << ',' << rocblas_gflops << "("
                     << gpu_time_used << "),";

        if(arg.norm_check)
        {
            rocblas_cout << cblas_gflops << "(" << cpu_time_used << "),";
            rocblas_cout << rocblas_error;
        }

        rocblas_cout << std::endl;
    }
}
