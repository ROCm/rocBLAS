/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_syr2k.hpp"
#include "testing_syr2k_batched.hpp"
#include "testing_syr2k_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum syrkx_test_type
    {
        SYRKX,
        SYRKX_BATCHED,
        SYRKX_STRIDED_BATCHED,
    };

    // test template
    template <template <typename...> class FILTER, syrkx_test_type SYRKX_TYPE>
    struct syrkx_template : RocBLAS_Test<syrkx_template<FILTER, SYRKX_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<syrkx_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYRKX_TYPE)
            {
            case SYRKX:
                return !strcmp(arg.function, "syrkx") || !strcmp(arg.function, "syrkx_bad_arg");
            case SYRKX_BATCHED:
                return !strcmp(arg.function, "syrkx_batched")
                       || !strcmp(arg.function, "syrkx_batched_bad_arg");
            case SYRKX_STRIDED_BATCHED:
                return !strcmp(arg.function, "syrkx_strided_batched")
                       || !strcmp(arg.function, "syrkx_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<syrkx_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA) << '_'
                     << arg.N << '_' << arg.K;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(SYRKX_TYPE == SYRKX_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(SYRKX_TYPE == SYRKX_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                if(arg.b_type == rocblas_datatype_f32_c || arg.b_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_beta<rocblas_float_complex>();
                else
                    name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldc;

                if(SYRKX_TYPE == SYRKX_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(SYRKX_TYPE == SYRKX_STRIDED_BATCHED || SYRKX_TYPE == SYRKX_BATCHED)
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
    struct syrkx_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct syrkx_testing<T,
                         std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                          || std::is_same<T, rocblas_float_complex>{}
                                          || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            // testing_syr2k second template false for syrkx
            if(!strcmp(arg.function, "syrkx"))
                testing_syr2k<T, false>(arg);
            else if(!strcmp(arg.function, "syrkx_bad_arg"))
                testing_syr2k_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "syrkx_batched"))
                testing_syr2k_batched<T, false>(arg);
            else if(!strcmp(arg.function, "syrkx_batched_bad_arg"))
                testing_syr2k_batched_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "syrkx_strided_batched"))
                testing_syr2k_strided_batched<T, false>(arg);
            else if(!strcmp(arg.function, "syrkx_strided_batched_bad_arg"))
                testing_syr2k_strided_batched_bad_arg<T, false>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using syrkx = syrkx_template<syrkx_testing, SYRKX>;
    TEST_P(syrkx, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<syrkx_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syrkx);

    using syrkx_batched = syrkx_template<syrkx_testing, SYRKX_BATCHED>;
    TEST_P(syrkx_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<syrkx_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syrkx_batched);

    using syrkx_strided_batched = syrkx_template<syrkx_testing, SYRKX_STRIDED_BATCHED>;
    TEST_P(syrkx_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<syrkx_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syrkx_strided_batched);

} // namespace
