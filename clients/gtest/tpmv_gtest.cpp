/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_tpmv.hpp"
#include "testing_tpmv_batched.hpp"
#include "testing_tpmv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible tpmv test cases
    enum tpmv_test_type
    {
        TPMV,
        TPMV_BATCHED,
        TPMV_STRIDED_BATCHED,
    };

    //tpmv test template
    template <template <typename...> class FILTER, tpmv_test_type TPMV_TYPE>
    struct tpmv_template : RocBLAS_Test<tpmv_template<FILTER, TPMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<tpmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TPMV_TYPE)
            {
            case TPMV:
                return !strcmp(arg.function, "tpmv") || !strcmp(arg.function, "tpmv_bad_arg");
            case TPMV_BATCHED:
                return !strcmp(arg.function, "tpmv_batched")
                       || !strcmp(arg.function, "tpmv_batched_bad_arg");
            case TPMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "tpmv_strided_batched")
                       || !strcmp(arg.function, "tpmv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<tpmv_template> name;

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
                 << '_' << (char)std::toupper(arg.transA) << '_' << (char)std::toupper(arg.diag)
                 << '_' << arg.M;

            if(TPMV_TYPE == TPMV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(TPMV_TYPE == TPMV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            if(TPMV_TYPE == TPMV_STRIDED_BATCHED || TPMV_TYPE == TPMV_BATCHED)
                name << '_' << arg.batch_count;

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct tpmv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct tpmv_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}
                                || std::is_same<T, rocblas_float_complex>{}
                                || std::is_same<T, rocblas_double_complex>{}>::type>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "tpmv"))
                testing_tpmv<T>(arg);
            else if(!strcmp(arg.function, "tpmv_bad_arg"))
                testing_tpmv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "tpmv_batched"))
                testing_tpmv_batched<T>(arg);
            else if(!strcmp(arg.function, "tpmv_batched_bad_arg"))
                testing_tpmv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "tpmv_strided_batched"))
                testing_tpmv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "tpmv_strided_batched_bad_arg"))
                testing_tpmv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using tpmv = tpmv_template<tpmv_testing, TPMV>;
    TEST_P(tpmv, blas2)
    {
        rocblas_simple_dispatch<tpmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(tpmv);

    using tpmv_batched = tpmv_template<tpmv_testing, TPMV_BATCHED>;
    TEST_P(tpmv_batched, blas2)
    {
        rocblas_simple_dispatch<tpmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(tpmv_batched);

    using tpmv_strided_batched = tpmv_template<tpmv_testing, TPMV_STRIDED_BATCHED>;
    TEST_P(tpmv_strided_batched, blas2)
    {
        rocblas_simple_dispatch<tpmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(tpmv_strided_batched);

} // namespace
