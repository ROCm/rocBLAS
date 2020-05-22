/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_hpr.hpp"
#include "testing_hpr_batched.hpp"
#include "testing_hpr_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum hpr_test_type
    {
        HPR,
        HPR_BATCHED,
        HPR_STRIDED_BATCHED,
    };

    //hpr test template
    template <template <typename...> class FILTER, hpr_test_type HPR_TYPE>
    struct hpr_template : RocBLAS_Test<hpr_template<FILTER, HPR_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<hpr_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HPR_TYPE)
            {
            case HPR:
                return !strcmp(arg.function, "hpr") || !strcmp(arg.function, "hpr_bad_arg");
            case HPR_BATCHED:
                return !strcmp(arg.function, "hpr_batched")
                       || !strcmp(arg.function, "hpr_batched_bad_arg");
            case HPR_STRIDED_BATCHED:
                return !strcmp(arg.function, "hpr_strided_batched")
                       || !strcmp(arg.function, "hpr_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<hpr_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_' << arg.alpha
                     << '_' << arg.incx;

                if(HPR_TYPE == HPR_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                if(HPR_TYPE == HPR_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(HPR_TYPE == HPR_STRIDED_BATCHED || HPR_TYPE == HPR_BATCHED)
                    name << '_' << arg.batch_count;
            }

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct hpr_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct hpr_testing<T,
                       std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                        || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "hpr"))
                testing_hpr<T>(arg);
            else if(!strcmp(arg.function, "hpr_bad_arg"))
                testing_hpr_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "hpr_batched"))
                testing_hpr_batched<T>(arg);
            else if(!strcmp(arg.function, "hpr_batched_bad_arg"))
                testing_hpr_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "hpr_strided_batched"))
                testing_hpr_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "hpr_strided_batched_bad_arg"))
                testing_hpr_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using hpr = hpr_template<hpr_testing, HPR>;
    TEST_P(hpr, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hpr_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hpr);

    using hpr_batched = hpr_template<hpr_testing, HPR_BATCHED>;
    TEST_P(hpr_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hpr_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hpr_batched);

    using hpr_strided_batched = hpr_template<hpr_testing, HPR_STRIDED_BATCHED>;
    TEST_P(hpr_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hpr_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hpr_strided_batched);

} // namespace
