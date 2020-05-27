/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_gemv.hpp"
#include "testing_gemv_batched.hpp"
#include "testing_gemv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible gemv test cases
    enum gemv_test_type
    {
        GEMV,
        GEMV_BATCHED,
        GEMV_STRIDED_BATCHED,
    };

    //gemv test template
    template <template <typename...> class FILTER, gemv_test_type GEMV_TYPE>
    struct gemv_template : RocBLAS_Test<gemv_template<FILTER, GEMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<gemv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMV_TYPE)
            {
            case GEMV:
                return !strcmp(arg.function, "gemv") || !strcmp(arg.function, "gemv_bad_arg");
            case GEMV_BATCHED:
                return !strcmp(arg.function, "gemv_batched")
                       || !strcmp(arg.function, "gemv_batched_bad_arg");
            case GEMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "gemv_strided_batched")
                       || !strcmp(arg.function, "gemv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemv_template> name;

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.transA)
                 << '_' << arg.M << '_' << arg.N << '_' << arg.alpha << '_' << arg.lda;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            name << '_' << arg.beta << '_' << arg.incy;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                name << '_' << arg.stride_y;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED || GEMV_TYPE == GEMV_BATCHED)
                name << '_' << arg.batch_count;

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
    struct gemv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gemv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemv"))
                testing_gemv<T>(arg);
            else if(!strcmp(arg.function, "gemv_bad_arg"))
                testing_gemv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemv_batched"))
                testing_gemv_batched<T>(arg);
            else if(!strcmp(arg.function, "gemv_batched_bad_arg"))
                testing_gemv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemv_strided_batched"))
                testing_gemv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "gemv_strided_batched_bad_arg"))
                testing_gemv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemv = gemv_template<gemv_testing, GEMV>;
    TEST_P(gemv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv);

    using gemv_batched = gemv_template<gemv_testing, GEMV_BATCHED>;
    TEST_P(gemv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv_batched);

    using gemv_strided_batched = gemv_template<gemv_testing, GEMV_STRIDED_BATCHED>;
    TEST_P(gemv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv_strided_batched);

} // namespace
