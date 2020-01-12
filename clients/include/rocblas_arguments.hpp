/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_ARGUMENTS_H_
#define ROCBLAS_ARGUMENTS_H_

#include "rocblas.h"
#include "rocblas_datatype2string.hpp"
#include "rocblas_math.hpp"
#include <istream>
#include <ostream>

// ROCBLAS_NAME_VALUE_PAIR(name) gives a string name and the named Arguments member
// ROCBLAS_NAME_VALUE_PAIR(name, value) gives a string name and the passed in value
#define ROCBLAS_SELECTOR(_1, _2, _3, ...) _3
#define ROCBLAS_SELECT_1ARG(NAME) #NAME, NAME
#define ROCBLAS_SELECT_2ARG(NAME, VALUE) #NAME, VALUE
#define ROCBLAS_NAME_VALUE_PAIR(...) \
    ROCBLAS_SELECTOR(__VA_ARGS__, ROCBLAS_SELECT_2ARG, ROCBLAS_SELECT_1ARG)(__VA_ARGS__)

/***************************************************************************
 *! \brief Class used to parse command arguments in both client & gtest    *
 * WARNING: If this data is changed, then rocblas_common.yaml must also be *
 * changed.                                                                *
 ***************************************************************************/
struct Arguments
{
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_int KL;
    rocblas_int KU;

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
    double alphai;
    double beta;
    double betai;

    char transA;
    char transB;
    char side;
    char uplo;
    char diag;

    rocblas_int batch_count;

    rocblas_int stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    rocblas_int stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    rocblas_int stride_c; //  stride_c > ldc * N
    rocblas_int stride_d; //  stride_d > ldd * N
    rocblas_int stride_x;
    rocblas_int stride_y;

    rocblas_int norm_check;
    rocblas_int unit_check;
    rocblas_int timing;
    rocblas_int iters;

    uint32_t algo;
    int32_t  solution_index;
    uint32_t flags;

    char function[64];
    char name[64];
    char category[64];

    rocblas_initialization initialization;
    char                   known_bug_platforms[64];

    /**********************
     * Tuple of arguments *
     **********************/
    auto as_tuple() const
    {
        return std::forward_as_tuple(ROCBLAS_NAME_VALUE_PAIR(M),
                                     ROCBLAS_NAME_VALUE_PAIR(N),
                                     ROCBLAS_NAME_VALUE_PAIR(K),
                                     ROCBLAS_NAME_VALUE_PAIR(KL),
                                     ROCBLAS_NAME_VALUE_PAIR(KU),
                                     ROCBLAS_NAME_VALUE_PAIR(lda),
                                     ROCBLAS_NAME_VALUE_PAIR(ldb),
                                     ROCBLAS_NAME_VALUE_PAIR(ldc),
                                     ROCBLAS_NAME_VALUE_PAIR(ldd),
                                     ROCBLAS_NAME_VALUE_PAIR(a_type),
                                     ROCBLAS_NAME_VALUE_PAIR(b_type),
                                     ROCBLAS_NAME_VALUE_PAIR(c_type),
                                     ROCBLAS_NAME_VALUE_PAIR(d_type),
                                     ROCBLAS_NAME_VALUE_PAIR(compute_type),
                                     ROCBLAS_NAME_VALUE_PAIR(incx),
                                     ROCBLAS_NAME_VALUE_PAIR(incy),
                                     ROCBLAS_NAME_VALUE_PAIR(incd),
                                     ROCBLAS_NAME_VALUE_PAIR(incb),
                                     ROCBLAS_NAME_VALUE_PAIR(alpha),
                                     ROCBLAS_NAME_VALUE_PAIR(alphai),
                                     ROCBLAS_NAME_VALUE_PAIR(beta),
                                     ROCBLAS_NAME_VALUE_PAIR(betai),
                                     ROCBLAS_NAME_VALUE_PAIR(transA),
                                     ROCBLAS_NAME_VALUE_PAIR(transB),
                                     ROCBLAS_NAME_VALUE_PAIR(side),
                                     ROCBLAS_NAME_VALUE_PAIR(uplo),
                                     ROCBLAS_NAME_VALUE_PAIR(diag),
                                     ROCBLAS_NAME_VALUE_PAIR(batch_count),
                                     ROCBLAS_NAME_VALUE_PAIR(stride_a),
                                     ROCBLAS_NAME_VALUE_PAIR(stride_b),
                                     ROCBLAS_NAME_VALUE_PAIR(stride_c),
                                     ROCBLAS_NAME_VALUE_PAIR(stride_d),
                                     ROCBLAS_NAME_VALUE_PAIR(stride_x),
                                     ROCBLAS_NAME_VALUE_PAIR(stride_y),
                                     ROCBLAS_NAME_VALUE_PAIR(norm_check),
                                     ROCBLAS_NAME_VALUE_PAIR(unit_check),
                                     ROCBLAS_NAME_VALUE_PAIR(timing),
                                     ROCBLAS_NAME_VALUE_PAIR(iters),
                                     ROCBLAS_NAME_VALUE_PAIR(algo),
                                     ROCBLAS_NAME_VALUE_PAIR(solution_index),
                                     ROCBLAS_NAME_VALUE_PAIR(flags),
                                     ROCBLAS_NAME_VALUE_PAIR(function),
                                     ROCBLAS_NAME_VALUE_PAIR(name),
                                     ROCBLAS_NAME_VALUE_PAIR(category),
                                     ROCBLAS_NAME_VALUE_PAIR(initialization),
                                     ROCBLAS_NAME_VALUE_PAIR(known_bug_platforms));
    }

    // Validate input format.
    static void validate(std::istream& ifs);

    // Convert (alpha, alphai) and (beta, betai) to a particular type
    // Return alpha, beta adjusted to 0 for when they are NaN
    template <typename T>
    T get_alpha() const
    {
        return rocblas_isnan(alpha) || rocblas_isnan(alphai) ? T(0)
                                                             : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return rocblas_isnan(beta) || rocblas_isnan(betai) ? T(0)
                                                           : convert_alpha_beta<T>(beta, betai);
    }

private:
    // Conversion from (real, imag) pair of values to T type
    template <typename T, typename U, typename std::enable_if<!is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r);
    }

    template <typename T, typename U, typename std::enable_if<+is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r, i);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg);
};

static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

#endif
