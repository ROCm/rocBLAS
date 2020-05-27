/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_her2k.hpp"
#include "testing_her2k_batched.hpp"
#include "testing_her2k_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum her2k_test_type
    {
        HER2K,
        HER2K_BATCHED,
        HER2K_STRIDED_BATCHED,
    };

    // test template
    template <template <typename...> class FILTER, her2k_test_type HER2K_TYPE>
    struct her2k_template : RocBLAS_Test<her2k_template<FILTER, HER2K_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<her2k_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HER2K_TYPE)
            {
            case HER2K:
                return !strcmp(arg.function, "her2k") || !strcmp(arg.function, "her2k_bad_arg");
            case HER2K_BATCHED:
                return !strcmp(arg.function, "her2k_batched")
                       || !strcmp(arg.function, "her2k_batched_bad_arg");
            case HER2K_STRIDED_BATCHED:
                return !strcmp(arg.function, "her2k_strided_batched")
                       || !strcmp(arg.function, "her2k_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<her2k_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                // beta only real

                name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA) << '_'
                     << arg.N << '_' << arg.K;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(HER2K_TYPE == HER2K_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(HER2K_TYPE == HER2K_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldc;

                if(HER2K_TYPE == HER2K_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(HER2K_TYPE == HER2K_STRIDED_BATCHED || HER2K_TYPE == HER2K_BATCHED)
                    name << '_' << arg.batch_count;
            }

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct her2k_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct her2k_testing<T,
                         std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                          || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "her2k"))
                testing_her2k<T>(arg);
            else if(!strcmp(arg.function, "her2k_bad_arg"))
                testing_her2k_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "her2k_batched"))
                testing_her2k_batched<T>(arg);
            else if(!strcmp(arg.function, "her2k_batched_bad_arg"))
                testing_her2k_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "her2k_strided_batched"))
                testing_her2k_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "her2k_strided_batched_bad_arg"))
                testing_her2k_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using her2k = her2k_template<her2k_testing, HER2K>;
    TEST_P(her2k, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<her2k_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her2k);

    using her2k_batched = her2k_template<her2k_testing, HER2K_BATCHED>;
    TEST_P(her2k_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<her2k_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her2k_batched);

    using her2k_strided_batched = her2k_template<her2k_testing, HER2K_STRIDED_BATCHED>;
    TEST_P(her2k_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<her2k_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her2k_strided_batched);

} // namespace
