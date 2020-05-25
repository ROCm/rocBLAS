/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_gemm.hpp"
#include "testing_gemm_batched.hpp"
#include "testing_gemm_batched_ex.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched.hpp"
#include "testing_gemm_strided_batched_ex.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // Types of GEMM tests
    enum gemm_test_type
    {
        GEMM,
        GEMM_EX,
        GEMM_BATCHED,
        GEMM_BATCHED_EX,
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
                return !strcmp(arg.function, "gemm") || !strcmp(arg.function, "gemm_bad_arg");

            case GEMM_EX:
                return !strcmp(arg.function, "gemm_ex") || !strcmp(arg.function, "gemm_ex_bad_arg");

            case GEMM_BATCHED:
                return !strcmp(arg.function, "gemm_batched")
                       || !strcmp(arg.function, "gemm_batched_bad_arg");

            case GEMM_BATCHED_EX:
                return !strcmp(arg.function, "gemm_batched_ex")
                       || !strcmp(arg.function, "gemm_batched_ex_bad_arg");

            case GEMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "gemm_strided_batched");

            case GEMM_STRIDED_BATCHED_EX:
                return !strcmp(arg.function, "gemm_strided_batched_ex")
                       || !strcmp(arg.function, "gemm_strided_batched_ex_bad_arg");
            }

            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemm_test_template> name;
            name << rocblas_datatype2string(arg.a_type);
            bool isEx = (GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_BATCHED_EX
                         || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX);
            bool isBatched
                = (GEMM_TYPE == GEMM_STRIDED_BATCHED || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX
                   || GEMM_TYPE == GEMM_BATCHED || GEMM_TYPE == GEMM_BATCHED_EX);

            if(isEx)
                name << rocblas_datatype2string(arg.b_type) << rocblas_datatype2string(arg.c_type)
                     << rocblas_datatype2string(arg.d_type)
                     << rocblas_datatype2string(arg.compute_type);

            name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB) << '_'
                 << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_' << arg.lda
                 << '_' << arg.ldb << '_' << arg.beta << '_' << arg.ldc;

            if(isEx)
                name << '_' << arg.ldd;

            if(isBatched)
                name << '_' << arg.batch_count;

            if(GEMM_TYPE == GEMM_STRIDED_BATCHED || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
                name << '_' << arg.stride_a << '_' << arg.stride_b << '_' << arg.stride_c;

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // ----------------------------------------------------------------------------
    // gemm
    // gemm_batched
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
    template <typename T>
    struct gemm_testing<
        T,
        T,
        T,
        std::enable_if_t<!std::is_same<T, void>{} && !std::is_same<T, rocblas_bfloat16>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemm"))
                testing_gemm<T>(arg);
            else if(!strcmp(arg.function, "gemm_bad_arg"))
                testing_gemm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemm_batched"))
                testing_gemm_batched<T>(arg);
            else if(!strcmp(arg.function, "gemm_batched_bad_arg"))
                testing_gemm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched"))
                testing_gemm_strided_batched<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemm = gemm_test_template<gemm_testing, GEMM>;
    TEST_P(gemm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm);

    using gemm_batched = gemm_test_template<gemm_testing, GEMM_BATCHED>;
    TEST_P(gemm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_batched);

    using gemm_strided_batched = gemm_test_template<gemm_testing, GEMM_STRIDED_BATCHED>;
    TEST_P(gemm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched);

    // ----------------------------------------------------------------------------
    // gemm_ex
    // gemm_batched_ex
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
    template <typename Ti, typename To, typename Tc>
    struct gemm_ex_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<!std::is_same<Ti, void>{}
                         && !(std::is_same<Ti, Tc>{} && std::is_same<Ti, rocblas_bfloat16>{})>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemm_ex"))
                testing_gemm_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_ex_bad_arg"))
                testing_gemm_ex_bad_arg<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_batched_ex"))
                testing_gemm_batched_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_batched_ex_bad_arg"))
                testing_gemm_batched_ex_bad_arg<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched_ex"))
                testing_gemm_strided_batched_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched_ex_bad_arg"))
                testing_gemm_strided_batched_ex_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemm_ex = gemm_test_template<gemm_ex_testing, GEMM_EX>;
    TEST_P(gemm_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemm_dispatch<gemm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_ex);

    using gemm_batched_ex = gemm_test_template<gemm_ex_testing, GEMM_BATCHED_EX>;
    TEST_P(gemm_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemm_dispatch<gemm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_batched_ex);

    using gemm_strided_batched_ex = gemm_test_template<gemm_ex_testing, GEMM_STRIDED_BATCHED_EX>;
    TEST_P(gemm_strided_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemm_dispatch<gemm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched_ex);

} // namespace
