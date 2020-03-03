/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "cblas_interface.hpp"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_test.hpp"
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

// Passed into gtest's SUCCEED macro when skipping a test.
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

class rocblas_print_helper
{
public:
    /************************************************************************************
     * Print values
     ************************************************************************************/
    // Default output
    template <typename T>
    static void print_value(std::ostream& os, const T& x)
    {
        os << x;
    }

    // Floating-point output
    static void print_value(std::ostream& os, double x)
    {
        if(std::isnan(x))
            os << ".nan";
        else if(std::isinf(x))
            os << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcpy(end, ".0");
            os << s;
        }
    }

    // Complex output
    template <typename T>
    static void print_value(std::ostream& os, const rocblas_complex_num<T>& x)
    {
        os << "'(";
        print_value(os, std::real(x));
        os << ",";
        print_value(os, std::imag(x));
        os << ")'";
    }
};

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void rocblas_print_matrix(
    std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
        {
            printf("matrix  col %d, row %d, CPU result=%f, GPU result=%f\n",
                   i,
                   j,
                   CPU_result[j + i * lda],
                   GPU_result[j + i * lda]);
        }
}

template <typename T>
void rocblas_print_matrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            rocblas_print_helper::print_value(std::cout, A[i + j * lda]);
            printf(" ");
        }
        printf("\n");
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a packed matrix.                  */
template <typename T>
inline void pack_matrix(bool upper, const T* A, T* AP, rocblas_int n)
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

template <typename T>
inline void
    regular_to_packed(bool upper, const host_vector<T>& A, host_vector<T>& AP, rocblas_int n)
{
    pack_matrix(upper, (const T*)A, (T*)AP, n);
}

template <typename T>
inline void regular_to_packed(bool                        upper,
                              const host_batch_vector<T>& A,
                              host_batch_vector<T>&       AP,
                              rocblas_int                 n,
                              rocblas_int                 batch_count)
{
    for(int b = 0; b < batch_count; b++)
    {
        pack_matrix(upper, (const T*)(A[b]), (T*)(AP[b]), n);
    }
}

template <typename T>
inline void regular_to_packed(bool                                upper,
                              const host_strided_batch_vector<T>& A,
                              host_strided_batch_vector<T>&       AP,
                              rocblas_int                         n,
                              rocblas_int                         batch_count)
{
    for(int b = 0; b < batch_count; b++)
    {
        pack_matrix(upper, (const T*)(A[b]), (T*)(AP[b]), n);
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
    printf("---------- %s ----------\n", name);
    int max_size = 8;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                rocblas_print_helper::print_value(std::cout, A[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
                printf("|");
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

#endif
