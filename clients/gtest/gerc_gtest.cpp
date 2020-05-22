/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_ger.hpp"
#include "testing_ger_batched.hpp"
#include "testing_ger_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible gemv test cases
    enum gerc_test_type
    {
        GERC,
        GERC_BATCHED,
        GERC_STRIDED_BATCHED,
    };

    //ger test template
    template <template <typename...> class FILTER, gerc_test_type GERC_TYPE>
    struct gerc_template : RocBLAS_Test<gerc_template<FILTER, GERC_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<gerc_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GERC_TYPE)
            {
            case GERC:
                return !strcmp(arg.function, "gerc") || !strcmp(arg.function, "gerc_bad_arg");
            case GERC_BATCHED:
                return !strcmp(arg.function, "gerc_batched")
                       || !strcmp(arg.function, "gerc_batched_bad_arg");
            case GERC_STRIDED_BATCHED:
                return !strcmp(arg.function, "gerc_strided_batched")
                       || !strcmp(arg.function, "gerc_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gerc_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << arg.M << '_' << arg.N << '_' << arg.alpha << 'r' << arg.alphai << 'i'
                     << '_' << arg.incx;

                if(GERC_TYPE == GERC_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.incy;

                if(GERC_TYPE == GERC_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                name << '_' << arg.lda;

                if(GERC_TYPE == GERC_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(GERC_TYPE == GERC_STRIDED_BATCHED || GERC_TYPE == GERC_BATCHED)
                    name << '_' << arg.batch_count;
            }

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct gerc_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gerc_testing<T,
                        std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gerc"))
                testing_ger<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_bad_arg"))
                testing_ger_bad_arg<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_batched"))
                testing_ger_batched<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_batched_bad_arg"))
                testing_ger_batched_bad_arg<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_strided_batched"))
                testing_ger_strided_batched<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_strided_batched_bad_arg"))
                testing_ger_strided_batched_bad_arg<T, true>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gerc = gerc_template<gerc_testing, GERC>;
    TEST_P(gerc, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<gerc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gerc);

    using gerc_batched = gerc_template<gerc_testing, GERC_BATCHED>;
    TEST_P(gerc_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<gerc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gerc_batched);

    using gerc_strided_batched = gerc_template<gerc_testing, GERC_STRIDED_BATCHED>;
    TEST_P(gerc_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<gerc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gerc_strided_batched);

} // namespace
