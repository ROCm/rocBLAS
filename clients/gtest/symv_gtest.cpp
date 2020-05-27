/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_symv.hpp"
#include "testing_symv_batched.hpp"
#include "testing_symv_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum symv_test_type
    {
        SYMV,
        SYMV_BATCHED,
        SYMV_STRIDED_BATCHED,
    };

    //symv test template
    template <template <typename...> class FILTER, symv_test_type SYMV_TYPE>
    struct symv_template : RocBLAS_Test<symv_template<FILTER, SYMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<symv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYMV_TYPE)
            {
            case SYMV:
                return !strcmp(arg.function, "symv") || !strcmp(arg.function, "symv_bad_arg");
            case SYMV_BATCHED:
                return !strcmp(arg.function, "symv_batched")
                       || !strcmp(arg.function, "symv_batched_bad_arg");
            case SYMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "symv_strided_batched")
                       || !strcmp(arg.function, "symv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<symv_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_' << arg.alpha;

                name << '_' << arg.lda;

                if(SYMV_TYPE == SYMV_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.incx;

                if(SYMV_TYPE == SYMV_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.beta;

                name << '_' << arg.incy;

                if(SYMV_TYPE == SYMV_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                if(SYMV_TYPE == SYMV_STRIDED_BATCHED || SYMV_TYPE == SYMV_BATCHED)
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
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct symv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct symv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "symv"))
                testing_symv<T>(arg);
            else if(!strcmp(arg.function, "symv_bad_arg"))
                testing_symv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "symv_batched"))
                testing_symv_batched<T>(arg);
            else if(!strcmp(arg.function, "symv_batched_bad_arg"))
                testing_symv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "symv_strided_batched"))
                testing_symv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "symv_strided_batched_bad_arg"))
                testing_symv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using symv = symv_template<symv_testing, SYMV>;
    TEST_P(symv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<symv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symv);

    using symv_batched = symv_template<symv_testing, SYMV_BATCHED>;
    TEST_P(symv_batched, blas2)
    {
        rocblas_simple_dispatch<symv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(symv_batched);

    using symv_strided_batched = symv_template<symv_testing, SYMV_STRIDED_BATCHED>;
    TEST_P(symv_strided_batched, blas2)
    {
        rocblas_simple_dispatch<symv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(symv_strided_batched);

} // namespace
