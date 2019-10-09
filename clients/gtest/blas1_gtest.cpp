/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_asum.hpp"
#include "testing_asum_batched.hpp"
#include "testing_asum_strided_batched.hpp"
#include "testing_axpy.hpp"
#include "testing_copy.hpp"
#include "testing_copy_batched.hpp"
#include "testing_copy_strided_batched.hpp"
#include "testing_dot.hpp"
#include "testing_dot_batched.hpp"
#include "testing_dot_strided_batched.hpp"
#include "testing_iamax_iamin.hpp"
#include "testing_nrm2.hpp"
#include "testing_nrm2_batched.hpp"
#include "testing_nrm2_strided_batched.hpp"
#include "testing_rot.hpp"
#include "testing_rot_batched.hpp"
#include "testing_rot_strided_batched.hpp"
#include "testing_rotg.hpp"
#include "testing_rotg_batched.hpp"
#include "testing_rotg_strided_batched.hpp"
#include "testing_rotm.hpp"
#include "testing_rotm_batched.hpp"
#include "testing_rotm_strided_batched.hpp"
#include "testing_rotmg.hpp"
#include "testing_rotmg_batched.hpp"
#include "testing_rotmg_strided_batched.hpp"
#include "testing_scal.hpp"
#include "testing_scal_batched.hpp"
#include "testing_scal_strided_batched.hpp"
#include "testing_swap.hpp"
#include "testing_swap_batched.hpp"
#include "testing_swap_strided_batched.hpp"
#include "type_dispatch.hpp"
#include "utility.hpp"

namespace
{
    enum class blas1
    {
        nrm2,
        nrm2_batched,
        nrm2_strided_batched,
        asum,
        asum_batched,
        asum_strided_batched,
        iamax,
        iamin,
        axpy,
        copy,
        copy_batched,
        copy_strided_batched,
        dot,
        dotc,
        dot_batched,
        dotc_batched,
        dot_strided_batched,
        dotc_strided_batched,
        scal,
        scal_batched,
        scal_strided_batched,
        swap,
        swap_batched,
        swap_strided_batched,
        rot,
        rot_batched,
        rot_strided_batched,
        rotg,
        rotg_batched,
        rotg_strided_batched,
        rotm,
        rotm_batched,
        rotm_strided_batched,
        rotmg,
        rotmg_batched,
        rotmg_strided_batched,
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

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                bool is_scal    = (BLAS1 == blas1::scal || BLAS1 == blas1::scal_batched
                                || BLAS1 == blas1::scal_strided_batched);
                bool is_rot     = (BLAS1 == blas1::rot || BLAS1 == blas1::rot_batched
                               || BLAS1 == blas1::rot_strided_batched);
                bool is_rotg    = (BLAS1 == blas1::rotg || BLAS1 == blas1::rotg_batched
                                || BLAS1 == blas1::rotg_strided_batched);
                bool is_rotmg   = (BLAS1 == blas1::rotmg || BLAS1 == blas1::rotmg_batched
                                 || BLAS1 == blas1::rotmg_strided_batched);
                bool is_batched = (BLAS1 == blas1::nrm2_batched || BLAS1 == blas1::asum_batched
                                   || BLAS1 == blas1::scal_batched || BLAS1 == blas1::swap_batched
                                   || BLAS1 == blas1::copy_batched || BLAS1 == blas1::dot_batched
                                   || BLAS1 == blas1::dotc_batched || BLAS1 == blas1::rot_batched
                                   || BLAS1 == blas1::rotm_batched || BLAS1 == blas1::rotg_batched
                                   || BLAS1 == blas1::rotmg_batched);
                bool is_strided
                    = (BLAS1 == blas1::nrm2_strided_batched || BLAS1 == blas1::asum_strided_batched
                       || BLAS1 == blas1::scal_strided_batched
                       || BLAS1 == blas1::swap_strided_batched
                       || BLAS1 == blas1::copy_strided_batched
                       || BLAS1 == blas1::dot_strided_batched
                       || BLAS1 == blas1::dotc_strided_batched
                       || BLAS1 == blas1::rot_strided_batched
                       || BLAS1 == blas1::rotm_strided_batched
                       || BLAS1 == blas1::rotg_strided_batched
                       || BLAS1 == blas1::rotmg_strided_batched);

                if((is_scal || is_rotg || is_rot) && arg.a_type != arg.b_type)
                    name << '_' << rocblas_datatype2string(arg.b_type);
                if(is_rot && arg.compute_type != arg.a_type)
                    name << '_' << rocblas_datatype2string(arg.compute_type);

                if(!is_rotg && !is_rotmg)
                    name << '_' << arg.N;

                if(BLAS1 == blas1::axpy || is_scal)
                    name << '_' << arg.alpha << "_" << arg.alphai;

                if(!is_rotg && !is_rotmg)
                    name << '_' << arg.incx;

                if(is_strided && !is_rotg)
                {
                    name << '_' << arg.stride_x;
                }

                if(BLAS1 == blas1::axpy || BLAS1 == blas1::copy
                   || BLAS1 == blas1::copy_strided_batched || BLAS1 == blas1::copy_batched
                   || BLAS1 == blas1::dot || BLAS1 == blas1::dotc || BLAS1 == blas1::dot_batched
                   || BLAS1 == blas1::dotc_batched || BLAS1 == blas1::dot_strided_batched
                   || BLAS1 == blas1::dotc_strided_batched || BLAS1 == blas1::swap
                   || BLAS1 == blas1::swap_batched || BLAS1 == blas1::swap_strided_batched
                   || BLAS1 == blas1::rot || BLAS1 == blas1::rot_batched
                   || BLAS1 == blas1::rot_strided_batched || BLAS1 == blas1::rotm
                   || BLAS1 == blas1::rotm_batched || BLAS1 == blas1::rotm_strided_batched)
                {
                    name << '_' << arg.incy;
                }

                if(BLAS1 == blas1::swap_strided_batched || BLAS1 == blas1::copy_strided_batched
                   || BLAS1 == blas1::dot_strided_batched || BLAS1 == blas1::dotc_strided_batched
                   || BLAS1 == blas1::rot_strided_batched || BLAS1 == blas1::rotm_strided_batched)
                {
                    name << '_' << arg.stride_y;
                }

                if(BLAS1 == blas1::rotg_strided_batched)
                {
                    name << '_' << arg.stride_a << '_' << arg.stride_b << '_' << arg.stride_c << '_'
                         << arg.stride_d;
                }

                if(BLAS1 == blas1::rotm_strided_batched || BLAS1 == blas1::rotmg_strided_batched)
                {
                    name << '_' << arg.stride_c;
                }

                if(is_batched || is_strided)
                {
                    name << "_" << arg.batch_count;
                }
            }

            return std::move(name);
        }
    };

    // This tells whether the BLAS1 tests are enabled
    template <blas1 BLAS1, typename Ti, typename To, typename Tc>
    using blas1_enabled = std::integral_constant<
        bool,
        ((BLAS1 == blas1::asum || BLAS1 == blas1::asum_batched
          || BLAS1 == blas1::asum_strided_batched)
         && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
         && (std::is_same<Ti, rocblas_float_complex>{} || std::is_same<Ti, rocblas_double_complex>{}
             || std::is_same<Ti, float>{} || std::is_same<Ti, double>{}))

            || (BLAS1 == blas1::axpy && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_half>{} || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || ((BLAS1 == blas1::dot || BLAS1 == blas1::dot_batched
                 || BLAS1 == blas1::dot_strided_batched)
                && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_half>{} || std::is_same<Ti, rocblas_bfloat16>{}
                    || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || ((BLAS1 == blas1::dotc || BLAS1 == blas1::dotc_batched
                 || BLAS1 == blas1::dotc_strided_batched)
                && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{}))

            || ((BLAS1 == blas1::nrm2 || BLAS1 == blas1::nrm2_batched
                 || BLAS1 == blas1::nrm2_strided_batched)
                && std::is_same<Ti, To>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{} || std::is_same<Ti, float>{}
                    || std::is_same<Ti, double>{}))

            || ((BLAS1 == blas1::scal || BLAS1 == blas1::scal_batched
                 || BLAS1 == blas1::scal_strided_batched)
                && std::is_same<To, Tc>{}
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

            || ((BLAS1 == blas1::copy || BLAS1 == blas1::copy_batched
                 || BLAS1 == blas1::copy_strided_batched)
                && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}
                    || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{}))

            || ((BLAS1 == blas1::swap || BLAS1 == blas1::swap_batched
                 || BLAS1 == blas1::swap_strided_batched)
                && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}
                    || std::is_same<Ti, rocblas_float_complex>{}
                    || std::is_same<Ti, rocblas_double_complex>{}))

            || ((BLAS1 == blas1::rot || BLAS1 == blas1::rot_batched
                 || BLAS1 == blas1::rot_strided_batched)
                && ((std::is_same<Ti, float>{} && std::is_same<Ti, To>{} && std::is_same<To, Tc>{})
                    || (std::is_same<Ti, double>{} && std::is_same<Ti, To>{}
                        && std::is_same<To, Tc>{})
                    || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{}
                        && std::is_same<Tc, rocblas_float_complex>{})
                    || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{}
                        && std::is_same<Tc, float>{})
                    || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{}
                        && std::is_same<Tc, rocblas_double_complex>{})
                    || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{}
                        && std::is_same<Tc, double>{})))

            || ((BLAS1 == blas1::rotg || BLAS1 == blas1::rotg_batched
                 || BLAS1 == blas1::rotg_strided_batched)
                && std::is_same<To, Tc>{}
                && ((std::is_same<Ti, float>{} && std::is_same<Ti, To>{})
                    || (std::is_same<Ti, double>{} && std::is_same<Ti, To>{})
                    || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{})
                    || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{})))

            || ((BLAS1 == blas1::rotm || BLAS1 == blas1::rotm_batched
                 || BLAS1 == blas1::rotm_strided_batched)
                && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}))

            || ((BLAS1 == blas1::rotmg || BLAS1 == blas1::rotmg_batched
                 || BLAS1 == blas1::rotmg_strided_batched)
                && std::is_same<To, Ti>{} && std::is_same<To, Tc>{}
                && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}))>;

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
           !strcmp(arg.function, #NAME "_bad_arg");                            \
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
BLAS1_TESTING(asum_batched,  ARG1)
BLAS1_TESTING(asum_strided_batched,  ARG1)
BLAS1_TESTING(nrm2,  ARG1)
BLAS1_TESTING(nrm2_batched,  ARG1)
BLAS1_TESTING(nrm2_strided_batched,  ARG1)
BLAS1_TESTING(iamax, ARG1)
BLAS1_TESTING(iamin, ARG1)
BLAS1_TESTING(axpy,  ARG1)
BLAS1_TESTING(copy,  ARG1)
BLAS1_TESTING(copy_batched,  ARG1)
BLAS1_TESTING(copy_strided_batched,  ARG1)
BLAS1_TESTING(dot,   ARG1)
BLAS1_TESTING(dotc,  ARG1)
BLAS1_TESTING(dot_batched,   ARG1)
BLAS1_TESTING(dotc_batched,  ARG1)
BLAS1_TESTING(dot_strided_batched,   ARG1)
BLAS1_TESTING(dotc_strided_batched,  ARG1)
BLAS1_TESTING(scal,  ARG2)
BLAS1_TESTING(scal_batched, ARG2)
BLAS1_TESTING(scal_strided_batched, ARG2)
BLAS1_TESTING(swap,  ARG1)
BLAS1_TESTING(swap_batched, ARG1)
BLAS1_TESTING(swap_strided_batched, ARG1)
BLAS1_TESTING(rot,   ARG3)
BLAS1_TESTING(rot_batched, ARG3)
BLAS1_TESTING(rot_strided_batched, ARG3)
BLAS1_TESTING(rotg,  ARG2)
BLAS1_TESTING(rotg_batched, ARG2)
BLAS1_TESTING(rotg_strided_batched, ARG2)
BLAS1_TESTING(rotm,  ARG1)
BLAS1_TESTING(rotm_batched, ARG1)
BLAS1_TESTING(rotm_strided_batched, ARG1)
BLAS1_TESTING(rotmg, ARG1)
BLAS1_TESTING(rotmg_batched, ARG1)
BLAS1_TESTING(rotmg_strided_batched, ARG1)

    // clang-format on

} // namespace
