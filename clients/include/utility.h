
/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <immintrin.h>
#include <typeinfo>
#include <fstream>
#include <iterator>
#include <cerrno>
#include <boost/iterator/filter_iterator.hpp>
#include <functional>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include "rocblas.h"
#include <memory>
#include <random>
#include <limits>
#include <type_traits>
#include <cinttypes>
#include "arg_check.h"
#include <gtest/gtest.h>

using namespace std;

typedef rocblas_half half;

/*!\file
 * \brief provide data initialization, timing, rocblas type <-> lapack char conversion utilities.
 */

#define CHECK_HIP_ERROR(error)                    \
    do                                            \
    {                                             \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    do                                                            \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    } while(0)

#define CHECK_ROCBLAS_ERROR(error)                                  \
    do                                                              \
    {                                                               \
        if(error != rocblas_status_success)                         \
        {                                                           \
            fprintf(stderr, "rocBLAS error: ");                     \
            if(error == rocblas_status_invalid_handle)              \
            {                                                       \
                fprintf(stderr, "rocblas_status_invalid_handle");   \
            }                                                       \
            else if(error == rocblas_status_not_implemented)        \
            {                                                       \
                fprintf(stderr, " rocblas_status_not_implemented"); \
            }                                                       \
            else if(error == rocblas_status_invalid_pointer)        \
            {                                                       \
                fprintf(stderr, "rocblas_status_invalid_pointer");  \
            }                                                       \
            else if(error == rocblas_status_invalid_size)           \
            {                                                       \
                fprintf(stderr, "rocblas_status_invalid_size");     \
            }                                                       \
            else if(error == rocblas_status_memory_error)           \
            {                                                       \
                fprintf(stderr, "rocblas_status_memory_error");     \
            }                                                       \
            else if(error == rocblas_status_internal_error)         \
            {                                                       \
                fprintf(stderr, "rocblas_status_internal_error");   \
            }                                                       \
            else                                                    \
            {                                                       \
                fprintf(stderr, "rocblas_status error");            \
            }                                                       \
            fprintf(stderr, "\n");                                  \
            return error;                                           \
        }                                                           \
    } while(0)

#define BLAS_1_RESULT_PRINT                           \
    do                                                \
    {                                                 \
        if(argus.timing)                              \
        {                                             \
            cout << "N, rocblas (us), ";              \
            if(argus.norm_check)                      \
            {                                         \
                cout << "CPU (us), error";            \
            }                                         \
            cout << endl;                             \
            cout << N << ',' << gpu_time_used << ','; \
            if(argus.norm_check)                      \
            {                                         \
                cout << cpu_time_used << ',';         \
                cout << rocblas_error;                \
            }                                         \
            cout << endl;                             \
        }                                             \
    } while(0)

// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline rocblas_half float_to_half(float val)
{
    // return static_cast<rocblas_half>( _mm_cvtsi128_si32( _mm_cvtps_ph( _mm_set_ss( val ), 0 ) )
    // );
    return _cvtss_sh(val, 0);
}

// Helper routine to convert halfs into their floats equivalent; uses F16C instructions
inline float half_to_float(rocblas_half val)
{
    // return static_cast<rocblas_half>(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(val), 0)));
    return _cvtsh_ss(val);
}

// Random number generator
typedef mt19937 rocblas_rng_t;
extern rocblas_rng_t rocblas_rng, rocblas_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand() { rocblas_rng = rocblas_seed; }

/* ============================================================================================ */
/*! \brief  Memory guard detects memory coruption */
template <class T, size_t PAD>
class memory_guard
{
    T guard[PAD];

    // Generate random NaN values
    template <typename U, typename UINT_T, int SIG, int EXP>
    static U random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(U), "Type sizes do not match");
        union
        {
            UINT_T u;
            U fp;
        } x;
        do
            x.u = uniform_int_distribution<UINT_T>()(rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG;   // Exponent = all 1's
        return x.fp;                              // NaN with random bits
    }

    // Random integer
    template <typename U>
    static typename enable_if<is_integral<U>::value>::type random_data(U* data, size_t size = 1)
    {
        for(size_t i = 0; i < size; ++i)
            data[i]  = uniform_int_distribution<U>()(rocblas_rng);
    }

    // Random NaN double
    static void random_data(double* data, size_t size = 1)
    {
        for(size_t i = 0; i < size; ++i)
            data[i]  = random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    static void random_data(float* data, size_t size = 1)
    {
        for(size_t i = 0; i < size; ++i)
            data[i]  = random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half (takes priority over templated integer function above)
    static void random_data(rocblas_half* data, size_t size = 1)
    {
        for(size_t i = 0; i < size; ++i)
            data[i]  = random_nan_data<rocblas_half, uint16_t, 10, 5>();
    }

    public:
    // Constructor initializes random data and saves it for later verification
    explicit memory_guard(T* data)
    {
        if(data)
        {
            // Fill with random data
            random_data(data, PAD);

            // Save random data in guard
            memcpy(guard, data, sizeof(guard));
        }
    }

    // Verify that random data has not been modified
    void check(const T* data) const { EXPECT_EQ(memcmp(data, guard, sizeof(guard)), 0); }
};

/* ============================================================================================ */
/*! \brief  pseudo-vector class which uses device memory */
template <typename T, size_t PAD = 4096>
class device_vector
{
#ifdef GOOGLE_TEST

    // Declaration order is important
    T host_front[PAD], host_back[PAD], *data;
    size_t size;
    bool empty;
    memory_guard<T, PAD> front, back;

    public:
    // Allocate device memory and initialize host_front and host_back with random NaN Data
    explicit device_vector(size_t size)
        : size(size),
          empty(size == 0 || hipMalloc(&data, (size + PAD * 2) * sizeof(T)) != hipSuccess),
          front(empty ? nullptr : host_front),
          back(empty ? nullptr : host_back)
    {
        if(empty)
        {
            data = nullptr;
        }
        else
        {
            // Copy host_front to device memory
            CHECK_HIP_ERROR(hipMemcpy(data, host_front, PAD * sizeof(T), hipMemcpyHostToDevice));

            // Point to allocated block
            data += PAD;

            // Copy host_back to device memory
            CHECK_HIP_ERROR(
                hipMemcpy(data + size, host_back, PAD * sizeof(T), hipMemcpyHostToDevice));
        }
    }

    ~device_vector()
    {
        if(!empty)
        {
            // Copy device memory to host_back
            CHECK_HIP_ERROR(
                hipMemcpy(host_back, data + size, PAD * sizeof(T), hipMemcpyDeviceToHost));

            // Point to beginning of device memory
            data -= PAD;

            // Copy device memory to host_front
            CHECK_HIP_ERROR(hipMemcpy(host_front, data, PAD * sizeof(T), hipMemcpyDeviceToHost));

            // Check integrity of host_front and host_back
            front.check(host_front);
            back.check(host_back);

            // Free device memory
            CHECK_HIP_ERROR(hipFree(data));
        }
    }

#else // GOOGLE_TEST

    // Code without memory guards
    T* data;

    public:
    explicit device_vector(size_t size)
    {
        if(size == 0 || hipMalloc(&data, size * sizeof(T)) != hipSuccess)
            data = nullptr;
    }

    ~device_vector()
    {
        if(data != nullptr)
            CHECK_HIP_ERROR(hipFree(data));
    }

#endif // GOOGLE_TEST

    // Decay into pointer wherever pointer is expected
    operator T*() { return data; }
    operator const T*() const { return data; }

    // Tell whether malloc failed
    explicit operator bool() const { return data != nullptr; }

    // Disallow copying or assigning
    device_vector(const device_vector&) = delete;
    device_vector& operator=(const device_vector&) = delete;
};

/* ============================================================================================ */
/*! \brief  pseudo-vector class which uses host memory */
template <typename T>
struct host_vector : vector<T>
{
    // Inherit constructors
    using vector<T>::vector;

    // Decay into pointer wherever pointer is expected
    operator T*() { return this->data(); }
    operator const T*() const { return this->data(); }
};

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class rocblas_local_handle
{
    rocblas_handle handle;

    public:
    rocblas_local_handle()
    {
        auto status = rocblas_create_handle(&handle);
        verify_rocblas_status_success(status, "ERROR: rocblas_local_handle constructor");
    }

    ~rocblas_local_handle()
    {
        auto status = rocblas_destroy_handle(handle);
        verify_rocblas_status_success(status, "ERROR: rocblas_local_handle destructor");
    }

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&() { return handle; }
    operator const rocblas_handle&() const { return handle; }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return uniform_int_distribution<int>(1, 10)(rocblas_rng);
}

// for rocblas_half, generate float, and convert to rocblas_half
/*! \brief  generate a random number in range [1,2,3] */
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return float_to_half(uniform_int_distribution<int>(1, 3)(rocblas_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return uniform_int_distribution<int8_t>(1, 3)(rocblas_rng);
};

/*! \brief  generate a random number in range [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1] */
template <typename T>
inline T random_generator_negative()
{
    return uniform_int_distribution<int>(-10, -1)(rocblas_rng);
};

/*! \brief  generate a random number in range [-3,-2,-1] */
template <>
inline rocblas_half random_generator_negative<rocblas_half>()
{
    return float_to_half(uniform_int_distribution<int>(-3, -1)(rocblas_rng));
};

/*! \brief  generate a random number in range [-3,-2,-1] */
template <>
inline int8_t random_generator_negative<int8_t>()
{
    return uniform_int_distribution<int8_t>(-3, -1)(rocblas_rng);
};

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// initializing vector with a constant value passed as a parameter
template <typename T>
void rocblas_init(vector<T>& A, rocblas_int M, rocblas_int N, rocblas_int lda, double value)
{
    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            A[i + j * lda] = value;
        }
    }
};

template <typename T>
void rocblas_init(vector<T>& A, rocblas_int M, rocblas_int N, rocblas_int lda)
{
    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            A[i + j * lda] = random_generator<T>();
        }
    }
};

// initialize strided_batched matrix
template <typename T>
void rocblas_init(vector<T>& A,
                  rocblas_int M,
                  rocblas_int N,
                  rocblas_int lda,
                  rocblas_int stride,
                  rocblas_int batch_count)
{
    for(rocblas_int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(rocblas_int i = 0; i < M; ++i)
        {
            for(rocblas_int j = 0; j < N; ++j)
            {
                A[i + j * lda + i_batch * stride] = random_generator<T>();
            }
        }
    }
};

template <typename T>
void rocblas_init_alternating_sign(vector<T>& A, rocblas_int M, rocblas_int N, rocblas_int lda)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            if(j % 2 ^ i % 2)
            {
                A[i + j * lda] = random_generator<T>();
            }
            else
            {
                A[i + j * lda] = random_generator_negative<T>();
            }
        }
    }
};

template <typename T>
void rocblas_init_alternating_sign(vector<T>& A,
                                   rocblas_int M,
                                   rocblas_int N,
                                   rocblas_int lda,
                                   rocblas_int stride,
                                   rocblas_int batch_count)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(rocblas_int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(rocblas_int i = 0; i < M; ++i)
        {
            for(rocblas_int j = 0; j < N; ++j)
            {
                if(j % 2 ^ i % 2)
                {
                    A[i + j * lda + i_batch * stride] = random_generator<T>();
                }
                else
                {
                    A[i + j * lda + i_batch * stride] = random_generator_negative<T>();
                }
            }
        }
    }
};

template <typename T>
void rocblas_init_alternating_sign(vector<T>& A,
                                   rocblas_int M,
                                   rocblas_int N,
                                   rocblas_int lda,
                                   rocblas_int stride,
                                   rocblas_int batch_count,
                                   double value)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(rocblas_int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(rocblas_int i = 0; i < M; ++i)
        {
            for(rocblas_int j = 0; j < N; ++j)
            {
                if(j % 2 ^ i % 2)
                {
                    A[i + j * lda + i_batch * stride] = value;
                }
                else
                {
                    A[i + j * lda + i_batch * stride] = -value;
                }
            }
        }
    }
};

template <typename T>
void rocblas_init(vector<T>& A,
                  rocblas_int M,
                  rocblas_int N,
                  rocblas_int lda,
                  rocblas_int stride_a,
                  rocblas_int batch_count,
                  double value)
{
    for(rocblas_int k = 0; k < batch_count; ++k)
    {
        for(rocblas_int i = 0; i < M; ++i)
        {
            for(rocblas_int j = 0; j < N; ++j)
            {
                A[i + j * lda + k * stride_a] = value;
            }
        }
    }
};

template <>
inline void
rocblas_init(vector<rocblas_half>& A, rocblas_int M, rocblas_int N, rocblas_int lda, double value)
{
    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            A[i + j * lda] = float_to_half(static_cast<float>(value));
        }
    }
};

template <>
inline void rocblas_init(vector<rocblas_half>& A,
                         rocblas_int M,
                         rocblas_int N,
                         rocblas_int lda,
                         rocblas_int stride_a,
                         rocblas_int batch_count,
                         double value)
{
    for(rocblas_int k = 0; k < batch_count; ++k)
    {
        for(rocblas_int i = 0; i < M; ++i)
        {
            for(rocblas_int j = 0; j < N; ++j)
            {
                A[i + j * lda + k * stride_a] = float_to_half(static_cast<float>(value));
            }
        }
    }
};

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
void rocblas_init_symmetric(vector<T>& A, rocblas_int N, rocblas_int lda)
{
    for(rocblas_int i = 0; i < N; ++i)
    {
        for(rocblas_int j = 0; j <= i; ++j)
        {
            A[j + i * lda] = A[i + j * lda] = random_generator<T>();
        }
    }
};

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
void rocblas_init_hermitian(vector<T>& A, rocblas_int N, rocblas_int lda)
{
    for(rocblas_int i = 0; i < N; ++i)
    {
        for(rocblas_int j = 0; j <= i; ++j)
        {
            A[j + i * lda] = A[i + j * lda] = random_generator<T>();
            if(i == j)
                A[j + i * lda].y = 0.0;
        }
    }
};

/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// initializing vector with a constant value passed as a parameter
template <typename T>
void rocblas_print_vector(vector<T>& A, rocblas_int M, rocblas_int N, rocblas_int lda)
{
    if(typeid(T) == typeid(float))
        cout << "vec[float]: ";
    else if(typeid(T) == typeid(double))
        cout << "vec[double]: ";
    else if(typeid(T) == typeid(rocblas_half))
        cout << "vec[rocblas_half]: ";

    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            if(typeid(T) == typeid(rocblas_half))
                printf("%04x,", A[i + j * lda]);
            else
                cout << A[i + j * lda] << ", ";
        }
    }
    cout << endl;
};

/* ============================================================================================ */
/*! \brief  Packs matricies into groups of 4 in N */
template <typename T>
inline void rocblas_packInt8(host_vector<T>& A, rocblas_int M, rocblas_int N, rocblas_int lda)
{
    /* Assumes original matrix provided in column major order, where N is a multiple of 4

        ---------- N ----------
   |  | 00 05 10 15 20 25 30 35      |00 05 10 15|20 25 30 35|
   |  | 01 06 11 16 21 26 31 36      |01 06 11 16|21 26 31 36|
   l  M 02 07 12 17 22 27 32 37  --> |02 07 12 17|22 27 32 37|
   d  | 03 08 13 18 23 28 33 38      |03 08 13 18|23 28 33 38|
   a  | 04 09 14 19 24 29 34 39      |04 09 14 19|24 29 34 39|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|

     Input :  00 01 02 03 04 ** ** 05   ...  38 39 ** **
     Output:  00 05 10 15 01 06 11 16   ...  ** ** ** **

   */

    if(N % 4 != 0)
    {
        std::cerr << "ERROR: dimension must be a multiple of 4 in order to pack" << std::endl;
    }

    host_vector<T> temp(A);

    size_t idx = 0;
    for(int colBase = 0; colBase < N; colBase += 4)
    {
        for(int row = 0; row < lda; row++)
        {
            for(int colOffset = 0; colOffset < 4; colOffset++)
            {
                A[idx++] = temp[(colBase + colOffset) * lda + row];
            }
        }
    }
}

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', rocblas_float_complex -> 'c', rocblas_double_complex
 * -> 'z' */
template <typename T>
char type2char();

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
void print_matrix(
    vector<T> CPU_result, vector<T> GPU_result, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
        {
            printf("matrix  col %d, row %d, CPU result=%f, GPU result=%f\n",
                   i,
                   j,
                   CPU_result[j + i * lda],
                   GPU_result[j + i * lda]);
        }
}

/* ============================================================================================ */
/*! \brief  Return normalized test name to conform to Google Tests */
// Note: forward<STRING> optimizes by eliminating copies of temporary strings
// Note: hit is used to disambigutate duplicates
template <class STRING>
string normalized_test_name(STRING&& prefix, unordered_map<string, size_t>& hit)
{
    auto p = hit.find(prefix);
    string str;

    // If parameters are repeated, append an incrementing suffix
    if(p != hit.end())
    {
        str = forward<STRING>(prefix) + "_t" + to_string(++p->second);
    }
    else
    {
        hit[prefix] = 1;
        str         = forward<STRING>(prefix);
    }

    // Replace non-alphanumeric characters with letters
    replace(str.begin(), str.end(), '-', 'n');
    replace(str.begin(), str.end(), '.', 'p');
    return str;
}

#ifdef __cplusplus
extern "C" {
#endif

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
/*  Convert rocblas constants to lapack char. */

char rocblas2char_operation(rocblas_operation value);

char rocblas2char_fill(rocblas_fill value);

char rocblas2char_diagonal(rocblas_diagonal value);

char rocblas2char_side(rocblas_side value);

char rocblas_datatype2char(rocblas_datatype value);

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

rocblas_operation char2rocblas_operation(char value);

rocblas_fill char2rocblas_fill(char value);

rocblas_diagonal char2rocblas_diagonal(char value);

rocblas_side char2rocblas_side(char value);

rocblas_datatype char2rocblas_datatype(char value);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */
struct Arguments
{
    rocblas_int M = 128;
    rocblas_int N = 128;
    rocblas_int K = 128;

    rocblas_int lda = 128;
    rocblas_int ldb = 128;
    rocblas_int ldc = 128;
    rocblas_int ldd = 128;

    rocblas_datatype a_type       = rocblas_datatype_f32_r;
    rocblas_datatype b_type       = rocblas_datatype_f32_r;
    rocblas_datatype c_type       = rocblas_datatype_f32_r;
    rocblas_datatype d_type       = rocblas_datatype_f32_r;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    rocblas_int incx = 1;
    rocblas_int incy = 1;
    rocblas_int incd = 1;
    rocblas_int incb = 1;

    double alpha = 1.0;
    double beta  = 0.0;

    char transA_option = 'N';
    char transB_option = 'N';
    char side_option   = 'L';
    char uplo_option   = 'L';
    char diag_option   = 'N';

    rocblas_int apiCallCount = 1;
    rocblas_int batch_count  = 10;

    rocblas_int stride_a = 128 * 128; //  stride_a > transA_option == 'N' ? lda * K : lda * M
    rocblas_int stride_b = 128 * 128; //  stride_b > transB_option == 'N' ? ldb * N : ldb * K
    rocblas_int stride_c = 128 * 128; //  stride_c > ldc * N
    rocblas_int stride_d = 128 * 128; //  stride_d > ldd * N

    rocblas_int norm_check = 0;
    rocblas_int unit_check = 1;
    rocblas_int timing     = 0;
    rocblas_int iters      = 10;

    uint32_t algo          = 0;
    int32_t solution_index = 0;
    uint32_t flags         = 0;
    size_t workspace_size  = 0;

    char function[32] = "";
    char namex[32]    = "";
    char category[32] = "";

    // Function to read Structures data from stream
    friend istream& operator>>(istream& s, Arguments& arg)
    {
        s.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return s;
    }

    // Function to print Structures data out to stream (for debugging)
    friend ostream& operator<<(ostream& o, const Arguments& arg)
    {
        return o << "{ 'transA': '" << arg.transA_option << "', 'transB': '" << arg.transB_option
                 << "', 'M': '" << arg.M << "', 'N': '" << arg.N << "', 'K': '" << arg.K
                 << "', 'lda': '" << arg.lda << "', 'ldb': '" << arg.ldb << "', 'ldc': '" << arg.ldc
                 << "', 'alpha': " << arg.alpha << ", 'beta': " << arg.beta << " }\n";
    }
};

enum rocblas_data_class
{
    rocblas_test_data,
    rocblas_perf_data,
};

// Class used to read Arguments data into the tests
template <rocblas_data_class>
struct RocBLAS_Data
{
    // filter iterator
    typedef boost::filter_iterator<function<bool(const Arguments&)>, istream_iterator<Arguments>>
        iterator;

    // Initialize class
    static void init(const string& file) { datafile = file; }

    // begin() iterator which accepts an optional filter.
    static iterator begin(function<bool(const Arguments&)> filter = [](const Arguments&) {
        return true;
    })
    {
        auto& ifs = get().ifs;

        // We re-seek the file back to position 0
        ifs.clear();
        ifs.seekg(0);

        // We create a filter iterator which will choose only those test cases
        // we want right now. This is to preserve Gtest output structure while
        // not creating no-op tests which "always pass".
        return iterator(filter, istream_iterator<Arguments>(ifs));
    }

    // end() iterator
    static iterator end() { return iterator(); }

    private:
    // We define this function to generate a single instance of the class on
    // first use so that we don't depend on the static initialization order.
    static RocBLAS_Data& get()
    {
        static RocBLAS_Data singleton;
        return singleton;
    }

    // Constructor which opens file
    RocBLAS_Data()
    {
        ifs.open(datafile, ifstream::binary);
        if(ifs.fail())
        {
            cerr << "Cannot open " << datafile << ": " << strerror(errno) << endl;
            throw ifstream::failure("Cannot open " + datafile);
        }
    }

    static string datafile;
    ifstream ifs;
};

template <rocblas_data_class C>
string RocBLAS_Data<C>::datafile =
    "(Uninitialized data. RocBLAS_Data<...>::init needs to be called first.)";

typedef RocBLAS_Data<rocblas_test_data> RocBLAS_TestData;
typedef RocBLAS_Data<rocblas_perf_data> RocBLAS_PerfData;

#endif
