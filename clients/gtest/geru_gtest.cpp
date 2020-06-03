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
    enum geru_test_type
    {
        GERU,
        GERU_BATCHED,
        GERU_STRIDED_BATCHED,
    };

    //ger test template
    template <template <typename...> class FILTER, geru_test_type GERU_TYPE>
    struct geru_template : RocBLAS_Test<geru_template<FILTER, GERU_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<geru_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GERU_TYPE)
            {
            case GERU:
                return !strcmp(arg.function, "geru") || !strcmp(arg.function, "geru_bad_arg");
            case GERU_BATCHED:
                return !strcmp(arg.function, "geru_batched")
                       || !strcmp(arg.function, "geru_batched_bad_arg");
            case GERU_STRIDED_BATCHED:
                return !strcmp(arg.function, "geru_strided_batched")
                       || !strcmp(arg.function, "geru_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<geru_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << arg.M << '_' << arg.N << '_' << arg.alpha << 'r' << arg.alphai << 'i'
                     << '_' << arg.incx;

                if(GERU_TYPE == GERU_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.incy;

                if(GERU_TYPE == GERU_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                name << '_' << arg.lda;

                if(GERU_TYPE == GERU_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(GERU_TYPE == GERU_STRIDED_BATCHED || GERU_TYPE == GERU_BATCHED)
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
    struct geru_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct geru_testing<T,
                        std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "geru"))
                testing_ger<T, false>(arg);
            else if(!strcmp(arg.function, "geru_bad_arg"))
                testing_ger_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "geru_batched"))
                testing_ger_batched<T, false>(arg);
            else if(!strcmp(arg.function, "geru_batched_bad_arg"))
                testing_ger_batched_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "geru_strided_batched"))
                testing_ger_strided_batched<T, false>(arg);
            else if(!strcmp(arg.function, "geru_strided_batched_bad_arg"))
                testing_ger_strided_batched_bad_arg<T, false>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using geru = geru_template<geru_testing, GERU>;
    TEST_P(geru, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<geru_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geru);

    using geru_batched = geru_template<geru_testing, GERU_BATCHED>;
    TEST_P(geru_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<geru_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geru_batched);

    using geru_strided_batched = geru_template<geru_testing, GERU_STRIDED_BATCHED>;
    TEST_P(geru_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<geru_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geru_strided_batched);

} // namespace
