/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_trsv.hpp"
#include "testing_trsv_batched.hpp"
#include "testing_trsv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible trsv test cases
    enum trsv_test_type
    {
        TRSV,
        TRSV_BATCHED,
        TRSV_STRIDED_BATCHED,
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trsv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trsv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trsv"))
                testing_trsv<T>(arg);
            else if(!strcmp(arg.function, "trsv_batched"))
                testing_trsv_batched<T>(arg);
            else if(!strcmp(arg.function, "trsv_strided_batched"))
                testing_trsv_strided_batched<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    template <template <typename...> class FILTER, trsv_test_type TRSV_TYPE>
    struct trsv_template : RocBLAS_Test<trsv_template<FILTER, TRSV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trsv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRSV_TYPE)
            {
            case TRSV:
                return !strcmp(arg.function, "trsv");
            case TRSV_BATCHED:
                return !strcmp(arg.function, "trsv_batched");
            case TRSV_STRIDED_BATCHED:
                return !strcmp(arg.function, "trsv_strided_batched");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trsv_template> name;
            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
                 << (char)std::toupper(arg.transA) << (char)std::toupper(arg.diag) << '_' << arg.M
                 << '_' << arg.lda;

            if(TRSV_TYPE == TRSV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(TRSV_TYPE == TRSV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            if(TRSV_TYPE != TRSV)
                name << '_' << arg.batch_count;

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    using trsv = trsv_template<trsv_testing, TRSV>;
    TEST_P(trsv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsv);

    using trsv_batched = trsv_template<trsv_testing, TRSV_BATCHED>;
    TEST_P(trsv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsv_batched);

    using trsv_strided_batched = trsv_template<trsv_testing, TRSV_STRIDED_BATCHED>;
    TEST_P(trsv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsv_strided_batched);

} // namespace
