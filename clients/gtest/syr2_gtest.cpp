/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_syr2.hpp"
#include "testing_syr2_batched.hpp"
#include "testing_syr2_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum syr2_test_type
    {
        SYR2,
        SYR2_BATCHED,
        SYR2_STRIDED_BATCHED,
    };

    //syr2 test template
    template <template <typename...> class FILTER, syr2_test_type SYR2_TYPE>
    struct syr2_template : RocBLAS_Test<syr2_template<FILTER, SYR2_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<syr2_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYR2_TYPE)
            {
            case SYR2:
                return !strcmp(arg.function, "syr2") || !strcmp(arg.function, "syr2_bad_arg");
            case SYR2_BATCHED:
                return !strcmp(arg.function, "syr2_batched")
                       || !strcmp(arg.function, "syr2_batched_bad_arg");
            case SYR2_STRIDED_BATCHED:
                return !strcmp(arg.function, "syr2_strided_batched")
                       || !strcmp(arg.function, "syr2_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<syr2_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda << '_' << arg.incx;

                if(SYR2_TYPE == SYR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.incy;

                if(SYR2_TYPE == SYR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                if(SYR2_TYPE == SYR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(SYR2_TYPE == SYR2_STRIDED_BATCHED || SYR2_TYPE == SYR2_BATCHED)
                    name << '_' << arg.batch_count;
            }
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct syr2_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct syr2_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "syr2"))
                testing_syr2<T>(arg);
            else if(!strcmp(arg.function, "syr2_bad_arg"))
                testing_syr2_bad_arg<T>();
            else if(!strcmp(arg.function, "syr2_batched"))
                testing_syr2_batched<T>(arg);
            else if(!strcmp(arg.function, "syr2_batched_bad_arg"))
                testing_syr2_batched_bad_arg<T>();
            else if(!strcmp(arg.function, "syr2_strided_batched"))
                testing_syr2_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "syr2_strided_batched_bad_arg"))
                testing_syr2_strided_batched_bad_arg<T>();
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using syr2 = syr2_template<syr2_testing, SYR2>;
    TEST_P(syr2, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<syr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syr2);

    using syr2_batched = syr2_template<syr2_testing, SYR2_BATCHED>;
    TEST_P(syr2_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<syr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syr2_batched);

    using syr2_strided_batched = syr2_template<syr2_testing, SYR2_STRIDED_BATCHED>;
    TEST_P(syr2_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<syr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syr2_strided_batched);

} // namespace
