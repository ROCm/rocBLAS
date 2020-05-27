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
void testing_trtri(const Arguments& arg)
{
    const bool FORTRAN          = arg.fortran;
    auto       rocblas_trtri_fn = FORTRAN ? rocblas_trtri<T, true> : rocblas_trtri<T, false>;

    rocblas_int N = arg.N;
    rocblas_int lda;
    rocblas_int ldinvA;
    ldinvA = lda = arg.lda;

    size_t size_A = size_t(lda) * N;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    rocblas_fill     uplo = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    bool invalid_size = lda < 0 || lda < N;
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_trtri_fn(handle, uplo, diag, N, nullptr, lda, nullptr, ldinvA),
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
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

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
            if(uplo == rocblas_fill_lower
               && j > i) // need to explicitly set unsused side to 0 if using it for temp storage
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

    CHECK_ROCBLAS_ERROR(rocblas_trtri_fn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));

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
        {
#ifdef GOOGLE_TEST
            FAIL() << "error in cblas_trtri";
#else
            rocblas_cerr << "error in cblas_trtri" << std::endl;
#endif
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            const double rel_error = get_epsilon<T>() * 1000;
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
        rocblas_cout << "N, lda, uplo, diag, rocblas-Gflops (us) ";
        if(arg.norm_check)
        {
            rocblas_cout << "CPU-Gflops(us), norm-error";
        }
        rocblas_cout << std::endl;

        rocblas_cout << N << ',' << lda << ',' << char_uplo << ',' << char_diag << ','
                     << rocblas_gflops << "(" << gpu_time_used << "),";

        if(arg.norm_check)
        {
            rocblas_cout << cblas_gflops << "(" << cpu_time_used << "),";
            rocblas_cout << rocblas_error;
        }

        rocblas_cout << std::endl;
    }
}
