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
    if(N < 0 || lda < 0 || batch_count < 0)
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

    // Initial Data on CPU
    rocblas_seedrand();
    host_vector<T> hA_sub(bsa);
    for(size_t i = 0; i < batch_count; i++)
    {
        rocblas_init_symmetric<T>(hA_sub, N, lda);
        for(int j = 0; j < bsa; j++)
        {
            hA.push_back(hA_sub[j]);
        }
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

    if(arg.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trtri_gflop_count<T>(N) / gpu_time_used * 1e6;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA, dinvA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

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

        if(arg.unit_check)
        {
            T rel_error = std::numeric_limits<T>::epsilon() * 1000;
            near_check_general<T>(N, N * batch_count, lda, hB, hA, rel_error);
        }

#if 0
        for(int i = 0; i < 32; i++)
        {
            printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hB[i], i, hA[i]);
        }
#endif

        if(arg.norm_check)
        {
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error = fmax(
                    rocblas_error,
                    norm_check_symmetric<T>('F', char_uplo, N, lda, hB + i * bsa, hA + i * bsa));
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
