/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_asum.hpp"
#include "testing_axpy.hpp"
#include "testing_copy.hpp"
#include "testing_dot.hpp"
#include "testing_iamax_iamin.hpp"
#include "testing_nrm2.hpp"
#include "testing_scal.hpp"
#include "testing_swap.hpp"
#include "type_dispatch.hpp"
#include "utility.hpp"

namespace
{
    enum class blas1
    {
        nrm2,
        asum,
        iamax,
        iamin,
        axpy,
        copy,
        dot,
        dotc,
        scal,
        swap,
    };

    // ----------------------------------------------------------------------------
    // BLAS1 testing template
    // ----------------------------------------------------------------------------
    template <template <typename...> class FILTER, blas1 BLAS1>
    struct blas1_test_template : public RocBLAS_Test<blas1_test_template<FILTER, BLAS1>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_blas1_dispatch<blas1_test_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg);

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<blas1_test_template> name;
            name << rocblas_datatype2string(arg.a_type);

            if(BLAS1 == blas1::scal && arg.a_type != arg.b_type)
                name << '_' << rocblas_datatype2string(arg.b_type);

            name << '_' << arg.N;

            if(BLAS1 == blas1::axpy || BLAS1 == blas1::scal)
                name << '_' << arg.alpha << "_" << arg.alphai;

            name << '_' << arg.incx;

            if(BLAS1 == blas1::axpy || BLAS1 == blas1::copy || BLAS1 == blas1::dot
               || BLAS1 == blas1::swap)
                name << '_' << arg.incy;

            return std::move(name);
        }
    };

    // This tells whether the BLAS1 tests are enabled
    template <blas1 BLAS1, typename Ti, typename To, typename Tc>
    using blas1_enabled = std::integral_constant<
        bool,
        (BLAS1 == blas1::asum && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
         && (std::is_same<Ti, rocblas_float_complex>{} || std::is_same<Ti, rocblas_double_complex>{}
             || std::is_same<Ti, float>{} || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::axpy && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_half>{} || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::dot && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::dotc && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{}))

            || (BLAS1 == blas1::nrm2 && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::scal && std::is_same<To, Tc>{}
                && ((std::is_same<Ti, rocblas_float_complex>{} && std::is_same<Ti, To>{})
                    || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<Ti, To>{})
                    || (std::is_same<Ti, float>{} && std::is_same<Ti, To>{})
                    || (std::is_same<Ti, double>{} && std::is_same<Ti, To>{})
                    || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{})
                    || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{})))

            || (BLAS1 == blas1::iamax && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::iamin && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::copy && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}
                    || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{}))

            || (BLAS1 == blas1::swap && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}
                    || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{}))>;

// Creates tests for one of the BLAS 1 functions
// ARG passes 1-3 template arguments to the testing_* function
// clang-format off
#define BLAS1_TESTING(NAME, ARG)                                               \
struct blas1_##NAME                                                            \
{                                                                              \
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>\
    struct testing : rocblas_test_invalid {};                                  \
                                                                               \
    template <typename Ti, typename To, typename Tc>                           \
    struct testing<Ti,                                                         \
                   To,                                                         \
                   Tc,                                                         \
                   typename std::enable_if<                                    \
                       blas1_enabled<blas1::NAME, Ti, To, Tc>{}>::type>        \
    {                                                                          \
        explicit operator bool() { return true; }                              \
        void operator()(const Arguments& arg)                                  \
        {                                                                      \
            if(!strcmp(arg.function, #NAME))                                   \
                testing_##NAME<ARG(Ti, To, Tc)>(arg);                          \
            else if(!strcmp(arg.function, #NAME "_bad_arg"))                   \
                testing_##NAME##_bad_arg<ARG(Ti, To, Tc)>(arg);                \
            else                                                               \
                FAIL() << "Internal error: Test called with unknown function: "\
                       << arg.function;                                        \
        }                                                                      \
    };                                                                         \
};                                                                             \
                                                                               \
using NAME = blas1_test_template<blas1_##NAME::template testing, blas1::NAME>; \
                                                                               \
template<>                                                                     \
inline bool NAME::function_filter(const Arguments& arg)                        \
{                                                                              \
    return !strcmp(arg.function, #NAME) ||                                     \
        !strcmp(arg.function, #NAME "_bad_arg");                               \
}                                                                              \
                                                                               \
TEST_P(NAME, blas1)                                                            \
{                                                                              \
    rocblas_blas1_dispatch<blas1_##NAME::template testing>(GetParam());        \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CATEGORIES(NAME)

#define ARG1(Ti, To, Tc) Ti
#define ARG2(Ti, To, Tc) Ti, To
#define ARG3(Ti, To, Tc) Ti, To, Tc

BLAS1_TESTING(asum,  ARG1)
BLAS1_TESTING(nrm2,  ARG1)
BLAS1_TESTING(iamax, ARG1)
BLAS1_TESTING(iamin, ARG1)
BLAS1_TESTING(axpy,  ARG1)
BLAS1_TESTING(copy,  ARG1)
BLAS1_TESTING(dot,   ARG1)
BLAS1_TESTING(dotc,  ARG1)
BLAS1_TESTING(scal,  ARG2)
BLAS1_TESTING(swap,  ARG1)

    // clang-format on

} // namespace
