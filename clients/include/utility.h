
/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <sys/time.h>
#include <sys/param.h>
#include <immintrin.h>
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
#include <cctype>
#include <locale.h>

/*!\file
 * \brief provide data initialization, timing, rocblas type <-> lapack char conversion utilities.
 */

#ifdef GOOGLE_TEST

#define EXPECT_ROCBLAS_STATUS EXPECT_EQ

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#else // GOOGLE_TEST

inline const char* rocblas_status_to_string(rocblas_status status)
{
    switch(status)
    {
    case rocblas_status_success: return "rocblas_status_success";
    case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
    case rocblas_status_memory_error: return "rocblas_status_memory_error";
    case rocblas_status_internal_error: return "rocblas_status_internal_error";
    default: return "<undefined rocblas_status value>";
    }
}

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
        std::cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                  << ", received " << rocblas_status_to_string(status) << std::endl;
}

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#define CHECK_HIP_ERROR(ERROR)                    \
    do                                            \
    {                                             \
        auto error = ERROR;                       \
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

#endif // GOOGLE_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline rocblas_half float_to_half(float val) { return _cvtss_sh(val, 0); }

// Helper routine to convert halfs into their floats equivalent; uses F16C instructions
inline float half_to_float(rocblas_half val) { return _cvtsh_ss(val); }

/* ============================================================================================ */
/*! \brief  returns true if value is NaN */

template <typename T>
inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <>
inline bool rocblas_isnan(rocblas_half arg)
{
    return (~arg & 0x7c00) == 0 && (arg & 0x3ff) != 0;
}

/* ============================================================================================ */
/*! \brief is_complex<T> returns true iff T is complex */

template <typename>
constexpr bool is_complex = false;

template <>
constexpr bool is_complex<rocblas_double_complex> = true;

template <>
constexpr bool is_complex<rocblas_float_complex> = true;

/* ============================================================================================ */
// Random number generator
using rocblas_rng_t = std::mt19937;
extern rocblas_rng_t rocblas_rng, rocblas_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand() { rocblas_rng = rocblas_seed; }

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */
class rocblas_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union
        {
            UINT_T u;
            T fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>()(rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG;   // Exponent = all 1's
        return x.fp;                              // NaN with random bits
    }

    public:
    // Random integer
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>()(rocblas_rng);
    }

    // Random NaN double
    explicit operator double() { return random_nan_data<double, uint64_t, 52, 11>(); }

    // Random NaN float
    explicit operator float() { return random_nan_data<float, uint32_t, 23, 8>(); }

    // Random NaN half (non-template rocblas_half takes precedence over integer template above)
    explicit operator rocblas_half() { return random_nan_data<rocblas_half, uint16_t, 10, 5>(); }
};

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where apppropriate */

template <typename T>
inline void rocblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i]     = static_cast<T>(rocblas_nan_rng());
}

/* ============================================================================================ */
/*! \brief  pseudo-vector class which uses device memory */

template <typename T, size_t PAD = 4096>
class device_vector
{
#ifdef GOOGLE_TEST

    T guard[PAD];

    void device_vector_setup()
    {
        if(hipMalloc(&data, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", bytes, bytes >> 30);
            data = nullptr;
        }
        else
        {
            // Initialize guard with random data
            rocblas_init_nan(guard, PAD);

            // Copy guard to device memory before allocated memory
            CHECK_HIP_ERROR(hipMemcpy(data, guard, sizeof(guard), hipMemcpyHostToDevice));

            // Point to allocated block
            data += PAD;

            // Copy guard to device memory after allocated memory
            CHECK_HIP_ERROR(hipMemcpy(data + size, guard, sizeof(guard), hipMemcpyHostToDevice));
        }
    }

    void device_vector_teardown()
    {
        if(data != nullptr)
        {
            T host[PAD];

            // Copy device memory after allocated memory to host
            CHECK_HIP_ERROR(hipMemcpy(host, data + size, sizeof(guard), hipMemcpyDeviceToHost));

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Point to guard before allocated memory
            data -= PAD;

            // Copy device memory after allocated memory to host
            CHECK_HIP_ERROR(hipMemcpy(host, data, sizeof(guard), hipMemcpyDeviceToHost));

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Free device memory
            CHECK_HIP_ERROR(hipFree(data));
        }
    }

    public:
    // Must wrap constructor and destructor in functions to allow Google Test macros to work
    explicit device_vector(size_t size) : size(size), bytes((size + PAD * 2) * sizeof(T))
    {
        device_vector_setup();
    }

    ~device_vector() { device_vector_teardown(); }

#else // GOOGLE_TEST

    // Code without memory guards

    public:
    explicit device_vector(size_t size) : size(size), bytes(size ? size * sizeof(T) : sizeof(T))
    {
        if(hipMalloc(&data, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%'zu GB)\n", bytes, bytes >> 30);
            data = nullptr;
        }
    }

    ~device_vector()
    {
        if(data != nullptr)
            CHECK_HIP_ERROR(hipFree(data));
    }

#endif // GOOGLE_TEST

    public:
    // Decay into pointer wherever pointer is expected
    operator T*() { return data; }
    operator const T*() const { return data; }

    // Tell whether malloc failed
    explicit operator bool() const { return data != nullptr; }

    // Disallow copying or assigning
    device_vector(const device_vector&) = delete;
    device_vector& operator=(const device_vector&) = delete;

    private:
    T* data;
    const size_t size, bytes;
};

/* ============================================================================================ */
/*! \brief  pseudo-vector class which uses host memory */
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

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
    rocblas_local_handle() { CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle)); }

    ~rocblas_local_handle() { CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle)); }

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
    return std::uniform_int_distribution<int>(1, 10)(rocblas_rng);
}

// for rocblas_half, generate float, and convert to rocblas_half
/*! \brief  generate a random number in range [1,2,3] */
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return float_to_half(std::uniform_int_distribution<int>(1, 3)(rocblas_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return std::uniform_int_distribution<int8_t>(1, 3)(rocblas_rng);
};

/* ============================================================================================ */
/*! \brief  negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half x)
{
    return x ^ 0x8000;
}

/* ============================================================================================ */
/*! \brief  print vector */
template <typename T>
inline void rocblas_print_vector(std::vector<T>& A, size_t M, size_t N, size_t lda)
{
    if(std::is_same<T, float>::value)
        std::cout << "vec[float]: ";
    else if(std::is_same<T, double>::value)
        std::cout << "vec[double]: ";
    else if(std::is_same<T, rocblas_half>::value)
        std::cout << "vec[rocblas_half]: ";

    for(size_t i = 0; i < M; ++i)
        for(size_t j = 0; j < N; ++j)
            std::cout << (std::is_same<T, rocblas_half>::value ? half_to_float(A[i + j * lda])
                                                               : A[i + j * lda])
                      << ", ";

    std::cout << std::endl;
}

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
inline void rocblas_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j                          = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
inline void rocblas_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
inline void rocblas_init_symmetric(std::vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value = random_generator<T>();
            // Warning: It's undefined behavior to assign to the
            // same array element twice in same statement (i==j)
            A[j + i * lda] = value;
            A[i + j * lda] = value;
        }
}

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
inline void rocblas_init_hermitian(std::vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value     = random_generator<T>();
            A[j + i * lda] = value;
            value.y        = (i == j) ? 0 : negate(value.y);
            A[i + j * lda] = value;
        }
}

/* ============================================================================================ */
/*! \brief  Packs matricies into groups of 4 in N */
template <typename T>
inline void rocblas_packInt8(host_vector<T>& A, size_t M, size_t N, size_t lda)
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
    for(size_t colBase = 0; colBase < N; colBase += 4)
    {
        for(size_t row = 0; row < lda; row++)
        {
            for(size_t colOffset = 0; colOffset < 4; colOffset++)
            {
                A[(colBase * lda + 4 * row) + colOffset] = temp[(colBase + colOffset) * lda + row];
            }
        }
    }
}

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void
print_matrix(std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
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

constexpr char rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none: return 'N';
    case rocblas_operation_transpose: return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return '\0';
}

constexpr char rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full: return 'F';
    }
    return '\0';
}

constexpr char rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit: return 'U';
    case rocblas_diagonal_non_unit: return 'N';
    }
    return '\0';
}

constexpr char rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left: return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both: return 'B';
    }
    return '\0';
}

constexpr char rocblas_datatype2char(rocblas_datatype value)
{
    switch(value)
    {
    case rocblas_datatype_f16_r: return 'h';
    case rocblas_datatype_f32_r: return 's';
    case rocblas_datatype_f64_r: return 'd';
    case rocblas_datatype_f16_c: return 'k';
    case rocblas_datatype_f32_c: return 'c';
    case rocblas_datatype_f64_c: return 'z';
    default:
        return 'e'; // todo, handle integer types
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n': return rocblas_operation_none;
    case 'T':
    case 't': return rocblas_operation_transpose;
    case 'C':
    case 'c': return rocblas_operation_conjugate_transpose;
    default: return static_cast<rocblas_operation>(-1);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_fill_upper;
    case 'L':
    case 'l': return rocblas_fill_lower;
    default: return static_cast<rocblas_fill>(-1);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_diagonal_unit;
    case 'N':
    case 'n': return rocblas_diagonal_non_unit;
    default: return static_cast<rocblas_diagonal>(-1);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l': return rocblas_side_left;
    case 'R':
    case 'r': return rocblas_side_right;
    default: return static_cast<rocblas_side>(-1);
    }
}

constexpr rocblas_datatype char2rocblas_datatype(char value)
{
    switch(value)
    {
    case 'H':
    case 'h': return rocblas_datatype_f16_r;
    case 'S':
    case 's': return rocblas_datatype_f32_r;
    case 'D':
    case 'd': return rocblas_datatype_f64_r;
    case 'C':
    case 'c': return rocblas_datatype_f32_c;
    case 'Z':
    case 'z': return rocblas_datatype_f64_c;
    default: return static_cast<rocblas_datatype>(-1);
    }
}

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */
struct Arguments
{
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_int lda;
    rocblas_int ldb;
    rocblas_int ldc;
    rocblas_int ldd;

    rocblas_datatype a_type;
    rocblas_datatype b_type;
    rocblas_datatype c_type;
    rocblas_datatype d_type;
    rocblas_datatype compute_type;

    rocblas_int incx;
    rocblas_int incy;
    rocblas_int incd;
    rocblas_int incb;

    double alpha;
    double beta;

    char transA_option;
    char transB_option;
    char side_option;
    char uplo_option;
    char diag_option;

    rocblas_int apiCallCount;
    rocblas_int batch_count;

    rocblas_int stride_a; //  stride_a > transA_option == 'N' ? lda * K : lda * M
    rocblas_int stride_b; //  stride_b > transB_option == 'N' ? ldb * N : ldb * K
    rocblas_int stride_c; //  stride_c > ldc * N
    rocblas_int stride_d; //  stride_d > ldd * N

    rocblas_int norm_check;
    rocblas_int unit_check;
    rocblas_int timing;
    rocblas_int iters;

    uint32_t algo;
    int32_t solution_index;
    uint32_t flags;
    size_t workspace_size;

    char function[64];
    char name[32];
    char category[32];

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& s, Arguments& arg)
    {
        s.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return s;
    }

    // Function to print Structures data out to stream (for debugging)
    friend std::ostream& operator<<(std::ostream& o, const Arguments& arg)
    {
        return o << "{ 'transA': '" << arg.transA_option << "', 'transB': '" << arg.transB_option
                 << "', 'M': '" << arg.M << "', 'N': '" << arg.N << "', 'K': '" << arg.K
                 << "', 'lda': '" << arg.lda << "', 'ldb': '" << arg.ldb << "', 'ldc': '" << arg.ldc
                 << "', 'alpha': " << arg.alpha << ", 'beta': " << arg.beta << " }\n";
    }
};

static_assert(std::is_pod<Arguments>::value,
              "Arguments is not a POD type, and thus is incompatible with C.");

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
    using iterator = boost::filter_iterator<std::function<bool(const Arguments&)>,
                                            std::istream_iterator<Arguments>>;

    // Initialize class
    static void init(const std::string& file) { datafile = file; }

    // begin() iterator which accepts an optional filter.
    static iterator begin(std::function<bool(const Arguments&)> filter = [](const Arguments&) {
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
        return iterator(filter, std::istream_iterator<Arguments>(ifs));
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

    // Private constructor which opens file
    RocBLAS_Data()
    {
        ifs.open(datafile, std::ifstream::binary);
        if(ifs.fail())
        {
            std::cerr << "Cannot open " << datafile << ": " << strerror(errno) << std::endl;
            throw std::ifstream::failure("Cannot open " + datafile);
        }
    }

    static std::string datafile;
    std::ifstream ifs;
};

// The datafile must be initialized by calling RocBLAS_Data<>::init()
template <rocblas_data_class C>
std::string RocBLAS_Data<C>::datafile =
    "(Uninitialized data. RocBLAS_Data<...>::init needs to be called first.)";

// RocBLAS_Data is instantiated once per rocblas_data_class enum
// One is for the correctness tests; one is for the performance tests
using RocBLAS_TestData = RocBLAS_Data<rocblas_test_data>;
using RocBLAS_PerfData = RocBLAS_Data<rocblas_perf_data>;

#ifdef GOOGLE_TEST

// The tests are instantiated by filtering through the RocBLAS_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, categ0ry)                                           \
    INSTANTIATE_TEST_CASE_P(categ0ry,                                                            \
                            testclass,                                                           \
                            testing::ValuesIn(RocBLAS_TestData::begin([](const Arguments& arg) { \
                                                  return !strcmp(arg.category, #categ0ry) &&     \
                                                         testclass::type_filter(arg) &&          \
                                                         testclass::function_filter(arg);        \
                                              }),                                                \
                                              RocBLAS_TestData::end()),                          \
                            testclass::PrintToStringParamName());

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)        \
    INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
    INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
    INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
    INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

/* ============================================================================================ */
/*! \brief  Normalized test name to conform to Google Tests */
// Template parameter is used to generate multiple instantiations
template <typename>
class RocBLAS_TestName
{
    std::ostringstream strm;

    public:
    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This map is private to each instantation of RocBLAS_TestName
        static std::unordered_map<std::string, size_t> table;
        std::string name(strm.str());

        // Replace non-alphanumeric characters with letters
        std::replace(name.begin(), name.end(), '-', 'n'); // minus
        std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

        // Complex (A,B) is replaced with ArBi
        name.erase(std::remove(name.begin(), name.end(), '('), name.end());
        std::replace(name.begin(), name.end(), ',', 'r');
        std::replace(name.begin(), name.end(), ')', 'i');

        // If parameters are repeated, append an incrementing suffix
        auto p = table.find(name);
        if(p != table.end())
            name += "_t" + std::to_string(++p->second);
        else
            table[name] = 1;

        return name;
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend RocBLAS_TestName& operator<<(RocBLAS_TestName& name, U&& obj)
    {
        name.strm << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend RocBLAS_TestName&& operator<<(RocBLAS_TestName&& name, U&& obj)
    {
        name.strm << std::forward<U>(obj);
        return std::move(name);
    }
};

// ----------------------------------------------------------------------------
// RocBLAS_Test base class. All non-legacy rocBLAS Google tests derive from it.
// It defines a type_filter() function and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class RocBLAS_Test : public testing::TestWithParam<Arguments>
{
    protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments&) { return static_cast<bool>(FILTER<T...>()); }
    };

    public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            return TEST::name_suffix(info.param);
        }
    };
};

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct rocblas_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    explicit operator bool() { return false; }

    // If this specialization is actually called, print fatal error message
    void operator()(const Arguments&)
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types\n";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        fputs(msg, stderr);
        exit(1);
#endif
    }
};

#endif
