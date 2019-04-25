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
void testing_trtri(const Arguments& arg)
{
    rocblas_int N = arg.N;
    rocblas_int lda;
    rocblas_int ldinvA;
    ldinvA = lda = arg.lda;

    size_t size_A = static_cast<size_t>(lda) * N;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    rocblas_fill uplo     = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N)
    {
        static const size_t safe_size = 100;
        device_vector<T> dA(safe_size);
        device_vector<T> dinvA(safe_size);
        if(!dA || !dinvA)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_trtri<T>(handle, uplo, diag, N, dA, lda, dinvA, ldinvA),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_A);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;

    device_vector<T> dA(size_A);
    device_vector<T> dinvA(size_A);
    if(!dA || !dinvA)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init_symmetric<T>(hA, N, lda);

    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            hA[i + j * lda] *= 0.01;

            if(j % 2)
                hA[i + j * lda] *= -1;
            if(uplo == rocblas_fill_lower &&
               j > i) // need to explicitly set unsused side to 0 if using it for temp storage
                hA[i + j * lda] = 0.0f;
            else if(uplo == rocblas_fill_upper && j < i)
                hA[i + j * lda] = 0.0f;
            if(i == j)
            {
                if(diag == rocblas_diagonal_unit)
                    hA[i + j * lda] = 1.0; // need to preprocess matrix for clbas_trtri
                else
                    hA[i + j * lda] *= 100.0;
            }
        }
    }

    hB = hA;

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

    CHECK_ROCBLAS_ERROR(rocblas_trtri<T>(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));

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

        rocblas_int info = cblas_trtri<T>(char_uplo, char_diag, N, hB, lda);

        if(info != 0)
            printf("error in cblas_trtri\n");

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            T rel_error = std::numeric_limits<T>::epsilon() * 1000;
            near_check_general<T>(N, N, lda, hB, hA, rel_error);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_symmetric<T>('F', char_uplo, N, lda, hB, hA);
        }
    }

    if(arg.timing)
    {
        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N, lda, uplo, diag, rocblas-Gflops (us) ";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops(us), norm-error";
        }
        std::cout << std::endl;

        std::cout << N << ',' << lda << ',' << char_uplo << ',' << char_diag << ','
                  << rocblas_gflops << "(" << gpu_time_used << "),";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << "(" << cpu_time_used << "),";
            std::cout << rocblas_error;
        }

        std::cout << std::endl;
    }
}
