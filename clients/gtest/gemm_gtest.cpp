/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_gemm.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched.hpp"
#include "testing_gemm_strided_batched_ex.hpp"
#include "type_dispatch.hpp"

namespace {

// Types of GEMM tests
enum gemm_test_type
{
    GEMM,
    GEMM_EX,
    GEMM_STRIDED_BATCHED,
    GEMM_STRIDED_BATCHED_EX,
};

// ----------------------------------------------------------------------------
// GEMM testing template
// ----------------------------------------------------------------------------
// The first template parameter is a class template which determines which
// combination of types applies to this test, and for those that do, instantiates
// the test code based on the function named in the test Arguments. The second
// template parameter is an enum which allows the 4 different flavors of GEMM to
// be differentiated.
//
// The RocBLAS_Test base class takes this class (CRTP) and the first template
// parameter as arguments, and provides common types such as type_filter_functor,
// and derives from the Google Test parameterized base classes.
//
// This class defines functions for filtering the types and function names which
// apply to this test, and for generating the suffix of the Google Test name
// corresponding to each instance of this test.
template <template <typename...> class FILTER, gemm_test_type GEMM_TYPE>
struct gemm_test_template : RocBLAS_Test<gemm_test_template<FILTER, GEMM_TYPE>, FILTER>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_gemm_dispatch<gemm_test_template::template type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        switch(GEMM_TYPE)
        {
        case GEMM:
            return !strcmp(arg.function, "testing_gemm") ||
                   !strcmp(arg.function, "testing_gemm_NaN") ||
                   !strcmp(arg.function, "testing_gemm_bad_arg");

        case GEMM_EX:
            return !strcmp(arg.function, "testing_gemm_ex") ||
                   !strcmp(arg.function, "testing_gemm_ex_bad_arg");

        case GEMM_STRIDED_BATCHED: return !strcmp(arg.function, "testing_gemm_strided_batched");

        case GEMM_STRIDED_BATCHED_EX:
            return !strcmp(arg.function, "testing_gemm_strided_batched_ex") ||
                   !strcmp(arg.function, "testing_gemm_strided_batched_ex_bad_arg");
        }

        return false;
    }

    // Goggle Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        RocBLAS_TestName<gemm_test_template> name;
        name << rocblas_datatype2string(arg.a_type);

        if(GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
            name << rocblas_datatype2string(arg.b_type) << rocblas_datatype2string(arg.c_type)
                 << rocblas_datatype2string(arg.d_type)
                 << rocblas_datatype2string(arg.compute_type);

        name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB) << '_'
             << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_' << arg.lda << '_'
             << arg.ldb << '_' << arg.beta << '_' << arg.ldc;

        if(GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
            name << '_' << arg.ldd;

        if(GEMM_TYPE == GEMM_STRIDED_BATCHED || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
            name << '_' << arg.batch_count << '_' << arg.stride_a << '_' << arg.stride_b << '_'
                 << arg.stride_c;

        return std::move(name);
    }
};

// ----------------------------------------------------------------------------
// gemm
// gemm_strided_batched
// ----------------------------------------------------------------------------

// In the general case of <Ti, To, Tc>, these tests do not apply, and if this
// functor is called, an internal error message is generated. When converted
// to bool, this functor returns false.
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct gemm_testing : rocblas_test_invalid
{
};

// When Ti = To = Tc != void, this test applies.
// When converted to bool, this functor returns true.
// Complex is not supported yet.
template <typename T>
struct gemm_testing<T,
                    T,
                    T,
                    typename std::enable_if<!std::is_same<T, void>::value && !is_complex<T>>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "testing_gemm"))
            testing_gemm<T>(arg);
        else if(!strcmp(arg.function, "testing_gemm_NaN"))
            testing_gemm_NaN<T>(arg);
        else if(!strcmp(arg.function, "testing_gemm_bad_arg"))
            testing_gemm_bad_arg<T>(arg);
        else if(!strcmp(arg.function, "testing_gemm_strided_batched"))
            testing_gemm_strided_batched<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

using gemm = gemm_test_template<gemm_testing, GEMM>;
TEST_P(gemm, blas3) { rocblas_gemm_dispatch<gemm_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(gemm);

using gemm_strided_batched = gemm_test_template<gemm_testing, GEMM_STRIDED_BATCHED>;
TEST_P(gemm_strided_batched, blas3) { rocblas_gemm_dispatch<gemm_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched);

// ----------------------------------------------------------------------------
// gemm_ex
// gemm_strided_batched_ex
// ----------------------------------------------------------------------------

// In the general case of <Ti, To, Tc>, these tests do not apply, and if this
// functor is called, an internal error message is generated. When converted
// to bool, this functor returns false.
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct gemm_ex_testing : rocblas_test_invalid
{
};

// When Ti != void, this test applies.
// When converted to bool, this functor returns true.
// Complex is not supported yet.
template <typename Ti, typename To, typename Tc>
struct gemm_ex_testing<
    Ti,
    To,
    Tc,
    typename std::enable_if<!std::is_same<Ti, void>::value && !is_complex<Ti>>::type>
{
    explicit operator bool() { return true; }

    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "testing_gemm_ex"))
            testing_gemm_ex<Ti, To, Tc>(arg);
        else if(!strcmp(arg.function, "testing_gemm_ex_bad_arg"))
            testing_gemm_ex_bad_arg<Ti, To, Tc>(arg);
        else if(!strcmp(arg.function, "testing_gemm_strided_batched_ex"))
            testing_gemm_strided_batched_ex<Ti, To, Tc>(arg);
        else if(!strcmp(arg.function, "testing_gemm_strided_batched_ex_bad_arg"))
            testing_gemm_strided_batched_ex_bad_arg<Ti, To, Tc>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

using gemm_ex = gemm_test_template<gemm_ex_testing, GEMM_EX>;
TEST_P(gemm_ex, blas3) { rocblas_gemm_dispatch<gemm_ex_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(gemm_ex);

using gemm_strided_batched_ex = gemm_test_template<gemm_ex_testing, GEMM_STRIDED_BATCHED_EX>;
TEST_P(gemm_strided_batched_ex, blas3) { rocblas_gemm_dispatch<gemm_ex_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched_ex);

} // namespace
