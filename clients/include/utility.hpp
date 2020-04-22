/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "cblas_interface.hpp"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_vector.hpp"
#include "utility.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/*!\file
 * \brief provide common utilities
 */

// We use rocblas_cout and rocblas_cerr instead of std::cout, std::cerr, stdout and stderr,
// for thread-safe IO.
//
// All stdio and std::ostream functions related to stdout and stderr are poisoned, as are
// functions which can create buffer overflows, or which are inherently thread-unsafe.
//
// This must come after the header #includes above, to avoid poisoning system headers.
//
// This is only enabled for rocblas-test and rocblas-bench.
//
// If you are here because of a poisoned identifier error, here is the rationale for each
// included identifier:
//
// cout, stdout: rocblas_cout should be used instead, for thread-safe and atomic line buffering
// cerr, stderr: rocblas_cerr should be used instead, for thread-safe and atomic line buffering
// clog: C++ stream which should not be used
// gets: Always unsafe; buffer-overflows; removed from later versions of the language; use fgets
// puts, putchar, fputs, printf, fprintf, vprintf, vfprintf: Use rocblas_cout or rocblas_cerr
// sprintf, vsprintf: Possible buffer overflows; us snprintf or vsnprintf instead
// strerror: Thread-unsafe; use snprintf / dprintf with %m or strerror_* alternatives
// strtok: Thread-unsafe; use strtok_r
// gmtime, ctime, asctime, localtime: Thread-unsafe
// tmpnam: Thread-unsafe; use mkstemp or related functions instead
// putenv: Use setenv instead
// clearenv, fcloseall, ecvt, fcvt: Miscellaneous thread-unsafe functions
// sleep: Might interact with signals by using alarm(); use nanosleep() instead
// abort: Does not abort as cleanly as rocblas_abort, and can be caught by a signal handler

#if defined(GOOGLE_TEST) || defined(ROCBLAS_BENCH)
#undef stdout
#undef stderr
#pragma GCC poison cout cerr clog stdout stderr gets puts putchar fputs fprintf printf sprintf    \
    vfprintf vprintf vsprintf perror strerror strtok gmtime ctime asctime localtime tmpnam putenv \
        clearenv fcloseall ecvt fcvt sleep abort
#else
// Suppress warnings about hipMalloc(), hipFree() except in rocblas-test and rocblas-bench
#undef hipMalloc
#undef hipFree
#endif

static constexpr char LIMITED_MEMORY_STRING[]
    = "Error: Attempting to allocate more memory than available.";

// TODO: This is dependent on internal gtest behaviour.
// Compared with result.message() when a test ended. Note that "Succeeded\n" is
// added to the beginning of the message automatically by gtest, so this must be compared.
static constexpr char LIMITED_MEMORY_STRING_GTEST[]
    = "Succeeded\nError: Attempting to allocate more memory than available.";

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class rocblas_local_handle
{
    rocblas_handle handle;

public:
    rocblas_local_handle()
    {
        rocblas_create_handle(&handle);
    }
    ~rocblas_local_handle()
    {
        rocblas_destroy_handle(handle);
    }

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&()
    {
        return handle;
    }
    operator const rocblas_handle&() const
    {
        return handle;
    }
};

/* ============================================================================================ */
/*  device query and print out their ID and name */
rocblas_int query_device_property();

/*  set current device to device_id */
void set_device(rocblas_int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocblas sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/* ============================================================================================ */
// Return path of this executable
std::string rocblas_exepath();

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void rocblas_print_matrix(
    std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
        {
            rocblas_cout << "matrix  col " << i << ", row " << j
                         << ", CPU result=" << CPU_result[j + i * lda]
                         << ", GPU result=" << GPU_result[j + i * lda] << "\n";
        }
}

template <typename T>
void rocblas_print_matrix(const char* name, T* A, size_t m, size_t n, size_t lda)
{
    rocblas_cout << "---------- " << name << " ----------\n";
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
            rocblas_cout << A[i + j * lda] << " ";
        rocblas_cout << std::endl;
    }
}

/* ============================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a banded matrix. */
template <typename T>
inline void regular_to_banded(
    bool upper, const T* A, rocblas_int lda, T* AB, rocblas_int ldab, rocblas_int n, rocblas_int k)
{
    // convert regular hA matrix to banded hAB matrix
    for(int j = 0; j < n; j++)
    {
        rocblas_int min1 = upper ? std::max(0, j - k) : j;
        rocblas_int max1 = upper ? j : std::min(n - 1, j + k);
        rocblas_int m    = upper ? k - j : -j;

        // Move bands of hA into new banded hAB format.
        for(int i = min1; i <= max1; i++)
            AB[j * ldab + (m + i)] = A[j * lda + i];

        min1 = upper ? k + 1 : std::min(k + 1, n - j);
        max1 = ldab - 1;

        // fill in bottom with random data to ensure we aren't using it.
        // for !upper, fill in bottom right triangle as well.
        for(int i = min1; i <= max1; i++)
            rocblas_init<T>(AB + j * ldab + i, 1, 1, 1);

        // for upper, fill in top left triangle with random data to ensure
        // we aren't using it.
        if(upper)
        {
            for(int i = 0; i < m; i++)
                rocblas_init<T>(AB + j * ldab + i, 1, 1, 1);
        }
    }
}

/* =============================================================================== */
/*! \brief For testing purposes, zeros out elements not needed in a banded matrix. */
template <typename T>
inline void banded_matrix_setup(bool upper, T* A, rocblas_int lda, rocblas_int n, rocblas_int k)
{
    // Made A a banded matrix with k sub/super-diagonals
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(upper && (j > k + i || i > j))
                A[j * n + i] = T(0);
            else if(!upper && (i > k + j || j > i))
                A[j * n + i] = T(0);
        }
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a packed matrix.                  */
template <typename T>
inline void regular_to_packed(bool upper, const T* A, T* AP, rocblas_int n)
{
    int index = 0;
    if(upper)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = i; j < n; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, makes a matrix hA into a unit_diagonal matrix and               *
 *         randomly initialize the diagonal.                                                     */
template <typename T>
void make_unit_diagonal(rocblas_fill uplo, T* hA, rocblas_int lda, rocblas_int N)
{
    if(uplo == rocblas_fill_lower)
    {
        for(int i = 0; i < N; i++)
        {
            T diag = hA[i + i * N];
            for(int j = 0; j <= i; j++)
                hA[i + j * lda] = hA[i + j * lda] / diag;
        }
    }
    else // rocblas_fill_upper
    {
        for(int j = 0; j < N; j++)
        {
            T diag = hA[j + j * lda];
            for(int i = 0; i <= j; i++)
                hA[i + j * lda] = hA[i + j * lda] / diag;
        }
    }

    // randomly initalize diagonal to ensure we aren't using it's values for tests.
    for(int i = 0; i < N; i++)
    {
        rocblas_init<T>(hA + i * lda + i, 1, 1, 1);
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, prepares matrix hA for a triangular solve.                      *
 *         Makes hA strictly diagonal dominant (SPD), then calculates Cholesky factorization     *
 *         of hA.                                                                                */
template <typename T>
void prepare_triangular_solve(T* hA, rocblas_int lda, T* AAT, rocblas_int N, char char_uplo)
{
    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T, T>(rocblas_operation_none,
                     rocblas_operation_conjugate_transpose,
                     N,
                     N,
                     N,
                     T(1.0),
                     hA,
                     lda,
                     hA,
                     lda,
                     T(0.0),
                     AAT,
                     lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += rocblas_abs(AAT[i + j * lda]);
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, N, hA, lda);
}

template <typename T>
void print_strided_batched(const char* name,
                           T*          A,
                           rocblas_int n1,
                           rocblas_int n2,
                           rocblas_int n3,
                           rocblas_int s1,
                           rocblas_int s2,
                           rocblas_int s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    rocblas_cout << "---------- " << name << " ----------\n";
    int max_size = 8;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                rocblas_cout << A[(i1 * s1) + (i2 * s2) + (i3 * s3)] << "|";
            }
            rocblas_cout << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            rocblas_cout << "\n";
    }
    rocblas_cout << std::flush;
}

template <typename T>
void print_batched_matrix(const char*           name,
                          host_batch_vector<T>& A,
                          rocblas_int           n1,
                          rocblas_int           n2,
                          rocblas_int           s1,
                          rocblas_int           s2,
                          rocblas_int           batch_count)
{
    // n1, n2 are matrix dimensions, sometimes called m, n
    // s1, s2 are matrix strides, sometimes called 1, lda
    int max_size = 8;

    for(int i3 = 0; i3 < A.batch_count() && i3 < max_size; i3++)
    {
        auto A_p = A[i3];
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                rocblas_cout << A_p[(i1 * s1) + (i2 * s2)] << "|";
            }
            rocblas_cout << "\n";
        }
        if(i3 < (batch_count - 1) && i3 < (max_size - 1))
            rocblas_cout << "\n";
    }
    rocblas_cout << std::flush;
}

#endif
