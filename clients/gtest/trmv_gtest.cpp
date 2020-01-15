/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_trmv.hpp"
#include "testing_trmv_batched.hpp"
#include "testing_trmv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible trmv test cases
    enum trmv_test_type
    {
        TRMV,
        TRMV_BATCHED,
        TRMV_STRIDED_BATCHED,
    };

    //trmv test template
    template <template <typename...> class FILTER, trmv_test_type TRMV_TYPE>
    struct trmv_template : RocBLAS_Test<trmv_template<FILTER, TRMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRMV_TYPE)
            {
            case TRMV:
                return !strcmp(arg.function, "trmv") || !strcmp(arg.function, "trmv_bad_arg");
            case TRMV_BATCHED:
                return !strcmp(arg.function, "trmv_batched")
                       || !strcmp(arg.function, "trmv_batched_bad_arg");
            case TRMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "trmv_strided_batched")
                       || !strcmp(arg.function, "trmv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trmv_template> name;

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
                 << '_' << (char)std::toupper(arg.transA) << '_' << (char)std::toupper(arg.diag)
                 << '_' << arg.M << '_' << arg.lda;

            if(TRMV_TYPE == TRMV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(TRMV_TYPE == TRMV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            if(TRMV_TYPE == TRMV_STRIDED_BATCHED || TRMV_TYPE == TRMV_BATCHED)
                name << '_' << arg.batch_count;

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trmv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trmv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trmv"))
                testing_trmv<T>(arg);
            else if(!strcmp(arg.function, "trmv_bad_arg"))
                testing_trmv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trmv_batched"))
                testing_trmv_batched<T>(arg);
            else if(!strcmp(arg.function, "trmv_batched_bad_arg"))
                testing_trmv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trmv_strided_batched"))
                testing_trmv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "trmv_strided_batched_bad_arg"))
                testing_trmv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using trmv = trmv_template<trmv_testing, TRMV>;
    TEST_P(trmv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trmv);

    using trmv_batched = trmv_template<trmv_testing, TRMV_BATCHED>;
    TEST_P(trmv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trmv_batched);

    using trmv_strided_batched = trmv_template<trmv_testing, TRMV_STRIDED_BATCHED>;
    TEST_P(trmv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trmv_strided_batched);

} // namespace
