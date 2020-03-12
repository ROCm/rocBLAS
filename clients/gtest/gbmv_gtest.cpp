/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_gbmv.hpp"
#include "testing_gbmv_batched.hpp"
#include "testing_gbmv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible gbmv test cases
    enum gbmv_test_type
    {
        GBMV,
        GBMV_BATCHED,
        GBMV_STRIDED_BATCHED,
    };

    //gbmv test template
    template <template <typename...> class FILTER, gbmv_test_type GBMV_TYPE>
    struct gbmv_template : RocBLAS_Test<gbmv_template<FILTER, GBMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<gbmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GBMV_TYPE)
            {
            case GBMV:
                return !strcmp(arg.function, "gbmv") || !strcmp(arg.function, "gbmv_bad_arg");
            case GBMV_BATCHED:
                return !strcmp(arg.function, "gbmv_batched")
                       || !strcmp(arg.function, "gbmv_batched_bad_arg");
            case GBMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "gbmv_strided_batched")
                       || !strcmp(arg.function, "gbmv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gbmv_template> name;

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.transA)
                 << '_' << arg.M << '_' << arg.N << '_' << arg.KL << '_' << arg.KU << '_'
                 << arg.alpha << '_' << arg.lda;

            if(GBMV_TYPE == GBMV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(GBMV_TYPE == GBMV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            name << '_' << arg.beta << '_' << arg.incy;

            if(GBMV_TYPE == GBMV_STRIDED_BATCHED)
                name << '_' << arg.stride_y;

            if(GBMV_TYPE == GBMV_STRIDED_BATCHED || GBMV_TYPE == GBMV_BATCHED)
                name << '_' << arg.batch_count;

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct gbmv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gbmv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gbmv"))
                testing_gbmv<T>(arg);
            else if(!strcmp(arg.function, "gbmv_bad_arg"))
                testing_gbmv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gbmv_batched"))
                testing_gbmv_batched<T>(arg);
            else if(!strcmp(arg.function, "gbmv_batched_bad_arg"))
                testing_gbmv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gbmv_strided_batched"))
                testing_gbmv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "gbmv_strided_batched_bad_arg"))
                testing_gbmv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gbmv = gbmv_template<gbmv_testing, GBMV>;
    TEST_P(gbmv, blas2)
    {
        rocblas_simple_dispatch<gbmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(gbmv);

    using gbmv_batched = gbmv_template<gbmv_testing, GBMV_BATCHED>;
    TEST_P(gbmv_batched, blas2)
    {
        rocblas_simple_dispatch<gbmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(gbmv_batched);

    using gbmv_strided_batched = gbmv_template<gbmv_testing, GBMV_STRIDED_BATCHED>;
    TEST_P(gbmv_strided_batched, blas2)
    {
        rocblas_simple_dispatch<gbmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(gbmv_strided_batched);

} // namespace
