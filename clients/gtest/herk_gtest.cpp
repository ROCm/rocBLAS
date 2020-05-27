/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_herk.hpp"
#include "testing_herk_batched.hpp"
#include "testing_herk_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible gemv test cases
    enum herk_test_type
    {
        HERK,
        HERK_BATCHED,
        HERK_STRIDED_BATCHED,
    };

    //ger test template
    template <template <typename...> class FILTER, herk_test_type HERK_TYPE>
    struct herk_template : RocBLAS_Test<herk_template<FILTER, HERK_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<herk_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HERK_TYPE)
            {
            case HERK:
                return !strcmp(arg.function, "herk") || !strcmp(arg.function, "herk_bad_arg");
            case HERK_BATCHED:
                return !strcmp(arg.function, "herk_batched")
                       || !strcmp(arg.function, "herk_batched_bad_arg");
            case HERK_STRIDED_BATCHED:
                return !strcmp(arg.function, "herk_strided_batched")
                       || !strcmp(arg.function, "herk_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<herk_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                // alpha & beta only real

                name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA) << '_'
                     << arg.N << '_' << arg.K;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(HERK_TYPE == HERK_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(arg.b_type == rocblas_datatype_f32_c || arg.b_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_beta<rocblas_float_complex>();
                else
                    name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldc;

                if(HERK_TYPE == HERK_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(HERK_TYPE == HERK_STRIDED_BATCHED || HERK_TYPE == HERK_BATCHED)
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
    struct herk_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct herk_testing<T,
                        std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "herk"))
                testing_herk<T>(arg);
            else if(!strcmp(arg.function, "herk_bad_arg"))
                testing_herk_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "herk_batched"))
                testing_herk_batched<T>(arg);
            else if(!strcmp(arg.function, "herk_batched_bad_arg"))
                testing_herk_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "herk_strided_batched"))
                testing_herk_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "herk_strided_batched_bad_arg"))
                testing_herk_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using herk = herk_template<herk_testing, HERK>;
    TEST_P(herk, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<herk_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herk);

    using herk_batched = herk_template<herk_testing, HERK_BATCHED>;
    TEST_P(herk_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<herk_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herk_batched);

    using herk_strided_batched = herk_template<herk_testing, HERK_STRIDED_BATCHED>;
    TEST_P(herk_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<herk_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herk_strided_batched);

} // namespace
