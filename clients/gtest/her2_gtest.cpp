/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_her2.hpp"
#include "testing_her2_batched.hpp"
#include "testing_her2_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum her2_test_type
    {
        HER2,
        HER2_BATCHED,
        HER2_STRIDED_BATCHED,
    };

    //her2 test template
    template <template <typename...> class FILTER, her2_test_type HER2_TYPE>
    struct her2_template : RocBLAS_Test<her2_template<FILTER, HER2_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<her2_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HER2_TYPE)
            {
            case HER2:
                return !strcmp(arg.function, "her2") || !strcmp(arg.function, "her2_bad_arg");
            case HER2_BATCHED:
                return !strcmp(arg.function, "her2_batched")
                       || !strcmp(arg.function, "her2_batched_bad_arg");
            case HER2_STRIDED_BATCHED:
                return !strcmp(arg.function, "her2_strided_batched")
                       || !strcmp(arg.function, "her2_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<her2_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                // T doesn't really matter here, just whether it's real or complex. In her2's case it's always complex
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_'
                     << arg.get_alpha<rocblas_double_complex>() << '_' << arg.incx << '_'
                     << arg.incy;

                if(HER2_TYPE == HER2_STRIDED_BATCHED)
                    name << '_' << arg.stride_x << '_' << arg.stride_y << '_' << arg.stride_a;

                if(HER2_TYPE == HER2_STRIDED_BATCHED || HER2_TYPE == HER2_BATCHED)
                    name << '_' << arg.batch_count;
            }
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct her2_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct her2_testing<T,
                        std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "her2"))
                testing_her2<T>(arg);
            else if(!strcmp(arg.function, "her2_bad_arg"))
                testing_her2_bad_arg<T>();
            else if(!strcmp(arg.function, "her2_batched"))
                testing_her2_batched<T>(arg);
            else if(!strcmp(arg.function, "her2_batched_bad_arg"))
                testing_her2_batched_bad_arg<T>();
            else if(!strcmp(arg.function, "her2_strided_batched"))
                testing_her2_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "her2_strided_batched_bad_arg"))
                testing_her2_strided_batched_bad_arg<T>();
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using her2 = her2_template<her2_testing, HER2>;
    TEST_P(her2, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<her2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her2);

    using her2_batched = her2_template<her2_testing, HER2_BATCHED>;
    TEST_P(her2_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<her2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her2_batched);

    using her2_strided_batched = her2_template<her2_testing, HER2_STRIDED_BATCHED>;
    TEST_P(her2_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<her2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her2_strided_batched);

} // namespace
