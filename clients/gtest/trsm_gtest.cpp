/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_trsm.hpp"
#include "testing_trsm_batched.hpp"
#include "testing_trsm_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible trsm test cases
    enum trsm_test_type
    {
        TRSM,
        TRSM_BATCHED,
        TRSM_STRIDED_BATCHED
    };

    // trsm test template
    template <template <typename...> class FILTER, trsm_test_type TRSM_TYPE>
    struct trsm_template : RocBLAS_Test<trsm_template<FILTER, TRSM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trsm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRSM_TYPE)
            {
            case TRSM:
                return !strcmp(arg.function, "trsm");
            case TRSM_BATCHED:
                return !strcmp(arg.function, "trsm_batched");
            case TRSM_STRIDED_BATCHED:
                return !strcmp(arg.function, "trsm_strided_batched");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trsm_template> name;

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.side)
                 << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA)
                 << (char)std::toupper(arg.diag) << '_' << arg.M << '_' << arg.N << '_' << arg.alpha
                 << '_' << arg.lda << '_';

            if(TRSM_TYPE == TRSM_STRIDED_BATCHED)
                name << arg.stride_a << '_';

            name << arg.ldb;

            if(TRSM_TYPE == TRSM_STRIDED_BATCHED)
                name << '_' << arg.stride_b;
            if(TRSM_TYPE == TRSM_STRIDED_BATCHED || TRSM_TYPE == TRSM_BATCHED)
                name << '_' << arg.batch_count;

            return std::move(name);
        }
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct trsm_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trsm_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
    {
        explicit operator bool()
        {
            return true;
        }
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trsm"))
                testing_trsm<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched"))
                testing_trsm_batched<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched"))
                testing_trsm_strided_batched<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using trsm = trsm_template<trsm_testing, TRSM>;
    TEST_P(trsm, blas3)
    {
        rocblas_simple_dispatch<trsm_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(trsm);

    using trsm_batched = trsm_template<trsm_testing, TRSM_BATCHED>;
    TEST_P(trsm_batched, blas3)
    {
        rocblas_simple_dispatch<trsm_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_batched);

    using trsm_strided_batched = trsm_template<trsm_testing, TRSM_STRIDED_BATCHED>;
    TEST_P(trsm_strided_batched, blas3)
    {
        rocblas_simple_dispatch<trsm_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_strided_batched);

} // namespace
