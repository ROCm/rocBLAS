/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "near.hpp"
#include "flops.hpp"

template <typename T>
void testing_trtri_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int lda         = arg.lda;
    rocblas_int batch_count = arg.batch_count;

    rocblas_int bsa = lda * N;
    size_t size_A   = static_cast<size_t>(bsa) * batch_count;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    // char_uplo = 'U';
    rocblas_fill uplo     = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || lda < 0 || lda < N || batch_count < 0)
    {
        static const size_t safe_size = 100;
        device_vector<T> dA(safe_size);
        device_vector<T> dinvA(safe_size);
        if(!dA || !dinvA)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_trtri_batched<T>(
                handle, uplo, diag, N, dA, lda, bsa, dinvA, lda, bsa, batch_count),
            rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hB(size_A);
    host_vector<T> hA;
    host_vector<T> hA_2(size_A);
    hA.reserve(size_A);

    // Initial Data on CPU
    rocblas_seedrand();
    host_vector<T> hA_sub(bsa);

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
                if(uplo == rocblas_fill_lower &&
                   j > i) // need to explicitly set unsused side to 0 if using it for temp storage
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
    if(!dA || !dinvA)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

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

    CHECK_ROCBLAS_ERROR(rocblas_trtri_batched<T>(
        handle, uplo, diag, N, dA, lda, bsa, dinvA, lda, bsa, batch_count));

    // Test in place
    CHECK_ROCBLAS_ERROR(
        rocblas_trtri_batched<T>(handle, uplo, diag, N, dA, lda, bsa, dA, lda, bsa, batch_count));

    if(arg.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trtri_gflop_count<T>(N) / gpu_time_used * 1e6;
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
            rocblas_int info = cblas_trtri<T>(char_uplo, char_diag, N, hB + i * bsa, lda);
            if(info != 0)
                printf("error in cblas_trtri\n");
        }
        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

#if 0
        rocblas_print_matrix(hB, hA, N, N, lda, 1);
#endif

        if(arg.unit_check)
        {
            T rel_error = std::numeric_limits<T>::epsilon() * 1000;
            near_check_general<T>(N, N * batch_count, lda, hB, hA, rel_error);
            near_check_general<T>(N, N * batch_count, lda, hB, hA_2, rel_error);
        }

        if(arg.norm_check)
        {
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error = fmax(
                    rocblas_error,
                    norm_check_symmetric<T>('F', char_uplo, N, lda, hB + i * bsa, hA + i * bsa));
                // printf("error=%f, %lu\n", rocblas_error, i);
            }
            rocblas_error = 0.0;
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error = fmax(
                    rocblas_error,
                    norm_check_symmetric<T>('F', char_uplo, N, lda, hB + i * bsa, hA_2 + i * bsa));
                // printf("error=%f, %lu\n", rocblas_error, i);
            }
        }
    } // end of norm_check

    if(arg.timing)
    {
        // only norm_check return an norm error, unit check won't return anything
        std::cout << "batch, N, lda, rocblas-Gflops (us) ";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops(us), norm-error";
        }
        std::cout << std::endl;

        std::cout << batch_count << ',' << N << ',' << lda << ',' << rocblas_gflops << "("
                  << gpu_time_used << "),";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << "(" << cpu_time_used << "),";
            std::cout << rocblas_error;
        }

        std::cout << std::endl;
    }
}
