/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_ARGUMENTS_H_
#define ROCBLAS_ARGUMENTS_H_

#include "../../library/src/include/rocblas_ostream.hpp"
#include "rocblas.h"
#include "rocblas_datatype2string.hpp"
#include "rocblas_math.hpp"
#include <cstddef>
#include <istream>
#include <map>
#include <ostream>
#include <tuple>

/***************************************************************************
 *! \brief Class used to parse command arguments in both client & gtest    *
 * WARNING: If this data is changed, then rocblas_common.yaml must also be *
 * changed.                                                                *
 ***************************************************************************/
struct Arguments
{
    /*************************************************************************
     *                    Beginning Of Arguments                             *
     *************************************************************************/
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

    /*************************************************************************
     *                     End Of Arguments                                  *
     *************************************************************************/

    // Generic macro which operates over the list of arguments
    // (in order of declaration)
    // clang-format off

#define FOR_EACH_ARGUMENT(OPER) \
    OPER(M),                    \
    OPER(N),                    \
    OPER(K),                    \
    OPER(KL),                   \
    OPER(KU),                   \
    OPER(lda),                  \
    OPER(ldb),                  \
    OPER(ldc),                  \
    OPER(ldd),                  \
    OPER(a_type),               \
    OPER(b_type),               \
    OPER(c_type),               \
    OPER(d_type),               \
    OPER(compute_type),         \
    OPER(incx),                 \
    OPER(incy),                 \
    OPER(incd),                 \
    OPER(incb),                 \
    OPER(alpha),                \
    OPER(alphai),               \
    OPER(beta),                 \
    OPER(betai),                \
    OPER(transA),               \
    OPER(transB),               \
    OPER(side),                 \
    OPER(uplo),                 \
    OPER(diag),                 \
    OPER(batch_count),          \
    OPER(stride_a),             \
    OPER(stride_b),             \
    OPER(stride_c),             \
    OPER(stride_d),             \
    OPER(stride_x),             \
    OPER(stride_y),             \
    OPER(norm_check),           \
    OPER(unit_check),           \
    OPER(timing),               \
    OPER(iters),                \
    OPER(algo),                 \
    OPER(solution_index),       \
    OPER(flags),                \
    OPER(function),             \
    OPER(name),                 \
    OPER(category),             \
    OPER(initialization),       \
    OPER(known_bug_platforms)

  /***************************************
   * Tuple of argument name, value pairs *
   ***************************************/
    #define NAME_VALUE_PAIR(NAME) #NAME, NAME
    auto as_tuple() const
    {
        return std::make_tuple(FOR_EACH_ARGUMENT(NAME_VALUE_PAIR));
    }

  /***************************************
   * Map of argument names to offsets    *
   ***************************************/
    #define NAME_OFFSET_PAIR(NAME) { #NAME, offsetof(Arguments, NAME) }
    static const auto& as_map()
    {
        static std::map<const char*, size_t> map{FOR_EACH_ARGUMENT(NAME_OFFSET_PAIR)};
        return map;
    }

#if 0
    //TODO: Implement function to get Arguments by name
    auto&& get(const char* name) &&
    {
        auto& map = this->as_map();

    }
#endif

    // clang-format on

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
    template <typename T, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r);
    }

    template <typename T, typename U, std::enable_if_t<+is_complex<T>, int> = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r, i);
    }

    // Function to print Arguments out to stream in YAML format
    friend rocblas_ostream& operator<<(rocblas_ostream& str, const Arguments& arg);

    // Google Tests uses this with std:ostream automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg);
};

// We make sure that the Arguments struct is C-compatible
static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

#undef NAME_VALUE_PAIR
#undef NAME_OFFSET_PAIR
#endif
