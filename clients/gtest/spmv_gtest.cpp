/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_spmv.hpp"
#include "testing_spmv_batched.hpp"
#include "testing_spmv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum spmv_test_type
    {
        SPMV,
        SPMV_BATCHED,
        SPMV_STRIDED_BATCHED,
    };

    //spmv test template
    template <template <typename...> class FILTER, spmv_test_type SPMV_TYPE>
    struct spmv_template : RocBLAS_Test<spmv_template<FILTER, SPMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<spmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SPMV_TYPE)
            {
            case SPMV:
                return !strcmp(arg.function, "spmv") || !strcmp(arg.function, "spmv_bad_arg");
            case SPMV_BATCHED:
                return !strcmp(arg.function, "spmv_batched")
                       || !strcmp(arg.function, "spmv_batched_bad_arg");
            case SPMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "spmv_strided_batched")
                       || !strcmp(arg.function, "spmv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<spmv_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_' << arg.alpha;

                if(SPMV_TYPE == SPMV_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.incx;

                if(SPMV_TYPE == SPMV_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.beta;

                name << '_' << arg.incy;

                if(SPMV_TYPE == SPMV_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                if(SPMV_TYPE == SPMV_STRIDED_BATCHED || SPMV_TYPE == SPMV_BATCHED)
                    name << '_' << arg.batch_count;
            }
            return std::move(name);
        }
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct spmv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct spmv_testing<T, std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "spmv"))
                testing_spmv<T>(arg);
            else if(!strcmp(arg.function, "spmv_bad_arg"))
                testing_spmv_bad_arg<T>();
            else if(!strcmp(arg.function, "spmv_batched"))
                testing_spmv_batched<T>(arg);
            else if(!strcmp(arg.function, "spmv_batched_bad_arg"))
                testing_spmv_batched_bad_arg<T>();
            else if(!strcmp(arg.function, "spmv_strided_batched"))
                testing_spmv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "spmv_strided_batched_bad_arg"))
                testing_spmv_strided_batched_bad_arg<T>();
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using spmv = spmv_template<spmv_testing, SPMV>;
    TEST_P(spmv, blas2)
    {
        rocblas_simple_dispatch<spmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(spmv);

    using spmv_batched = spmv_template<spmv_testing, SPMV_BATCHED>;
    TEST_P(spmv_batched, blas2)
    {
        rocblas_simple_dispatch<spmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(spmv_batched);

    using spmv_strided_batched = spmv_template<spmv_testing, SPMV_STRIDED_BATCHED>;
    TEST_P(spmv_strided_batched, blas2)
    {
        rocblas_simple_dispatch<spmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(spmv_strided_batched);

} // namespace
