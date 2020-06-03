/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_tpsv.hpp"
#include "testing_tpsv_batched.hpp"
#include "testing_tpsv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible tpsv test cases
    enum tpsv_test_type
    {
        TPSV,
        TPSV_BATCHED,
        TPSV_STRIDED_BATCHED,
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct tpsv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct tpsv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "tpsv"))
                testing_tpsv<T>(arg);
            else if(!strcmp(arg.function, "tpsv_bad_arg"))
                testing_tpsv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "tpsv_batched"))
                testing_tpsv_batched<T>(arg);
            else if(!strcmp(arg.function, "tpsv_batched_bad_arg"))
                testing_tpsv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "tpsv_strided_batched"))
                testing_tpsv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "tpsv_strided_batched_bad_arg"))
                testing_tpsv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    template <template <typename...> class FILTER, tpsv_test_type TPSV_TYPE>
    struct tpsv_template : RocBLAS_Test<tpsv_template<FILTER, TPSV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<tpsv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TPSV_TYPE)
            {
            case TPSV:
                return !strcmp(arg.function, "tpsv") || !strcmp(arg.function, "tpsv_bad_arg");
            case TPSV_BATCHED:
                return !strcmp(arg.function, "tpsv_batched")
                       || !strcmp(arg.function, "tpsv_batched_bad_arg");
            case TPSV_STRIDED_BATCHED:
                return !strcmp(arg.function, "tpsv_strided_batched")
                       || !strcmp(arg.function, "tpsv_strided_batched_bad_arg");
            }

            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<tpsv_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
                name << "_bad_arg";

            name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA)
                 << (char)std::toupper(arg.diag) << '_' << arg.N;

            if(TPSV_TYPE == TPSV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(TPSV_TYPE == TPSV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            if(TPSV_TYPE != TPSV)
                name << '_' << arg.batch_count;

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    using tpsv = tpsv_template<tpsv_testing, TPSV>;
    TEST_P(tpsv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<tpsv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(tpsv);

    using tpsv_batched = tpsv_template<tpsv_testing, TPSV_BATCHED>;
    TEST_P(tpsv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<tpsv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(tpsv_batched);

    using tpsv_strided_batched = tpsv_template<tpsv_testing, TPSV_STRIDED_BATCHED>;
    TEST_P(tpsv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<tpsv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(tpsv_strided_batched);

} // namespace
