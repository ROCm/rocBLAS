/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
void testing_trtri_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int lda         = arg.lda;
    rocblas_int batch_count = arg.batch_count;

    size_t size_A = size_t(lda) * N;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    // char_uplo = 'U';
    rocblas_fill     uplo = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    if(N < 0 || lda < 0 || lda < N || batch_count <= 0)
    {
        static constexpr size_t safe_size = 100;
        device_vector<T*, 0, T> dA(1);
        device_vector<T*, 0, T> dInv(1);

        if(!dA || !dInv)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_trtri_batched<T>(handle, uplo, diag, N, dA, lda, dInv, lda, batch_count),
            N < 0 || lda < 0 || lda < N || batch_count < 0 ? rocblas_status_invalid_size
                                                           : rocblas_status_success);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hB[batch_count];
    host_vector<T> hA[batch_count];
    host_vector<T> hA_2[batch_count];
    for(int b = 0; b < batch_count; b++)
    {
        hB[b]   = host_vector<T>(size_A);
        hA[b]   = host_vector<T>(size_A);
        hA_2[b] = host_vector<T>(size_A);
    }

    // Initial Data on CPU
    rocblas_seedrand();
    host_vector<T> hA_sub(size_A);

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
        // hA[b].insert(std::end(hA[b]), std::begin(hA_sub), std::end(hA_sub));
        hA[b] = hA_sub;
        hB[b] = hA[b];
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    device_batch_vector<T>  Av(batch_count, size_A);
    device_batch_vector<T>  invAv(batch_count, size_A);
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dinvA(batch_count);
    int                     last = batch_count - 1;
    if((!Av[last] && size_A) || (!invAv[last] && size_A) || !dA || !dinvA)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(hipMemcpy(Av[b], hA[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(invAv[b], hA[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, Av, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, invAv, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    CHECK_ROCBLAS_ERROR(
        rocblas_trtri_batched<T>(handle, uplo, diag, N, dA, lda, dinvA, lda, batch_count));

    // Test in place
    CHECK_ROCBLAS_ERROR(
        rocblas_trtri_batched<T>(handle, uplo, diag, N, dA, lda, dA, lda, batch_count));

    if(arg.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = batch_count * trtri_gflop_count<T>(N) / gpu_time_used * 1e6;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(hipMemcpy(hA[b], invAv[b], sizeof(T) * size_A, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hA_2[b], Av[b], sizeof(T) * size_A, hipMemcpyDeviceToHost));
    }

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
            rocblas_int info = cblas_trtri<T>(char_uplo, char_diag, N, hB[i], lda);
            if(info != 0)
                printf("error in cblas_trtri\n");
        }
        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = batch_count * trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

#if 0
        rocblas_print_matrix(hB[b], hA[b], N, N, lda, 1);
#endif

        if(arg.unit_check)
        {
            T rel_error = std::numeric_limits<T>::epsilon() * 1000;
            near_check_general<T>(N, N, batch_count, lda, hB, hA, rel_error);
            near_check_general<T>(N, N, batch_count, lda, hB, hA_2, rel_error);
        }

        if(arg.norm_check)
        {
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error = fmax(rocblas_error,
                                     norm_check_symmetric<T>('F', char_uplo, N, lda, hB[i], hA[i]));
                // printf("error=%f, %lu\n", rocblas_error, i);
            }
            rocblas_error = 0.0;
            for(size_t i = 0; i < batch_count; i++)
            {
                rocblas_error = fmax(
                    rocblas_error, norm_check_symmetric<T>('F', char_uplo, N, lda, hB[i], hA_2[i]));
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
