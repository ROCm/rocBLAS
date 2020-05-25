/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_trmm.hpp"
#include "testing_trmm_batched.hpp"
#include "testing_trmm_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible trmm test cases
    enum trmm_test_type
    {
        TRMM,
        TRMM_BATCHED,
        TRMM_STRIDED_BATCHED,
    };

    //trmm test template
    template <template <typename...> class FILTER, trmm_test_type TRMM_TYPE>
    struct trmm_template : RocBLAS_Test<trmm_template<FILTER, TRMM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trmm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRMM_TYPE)
            {
            case TRMM:
                return !strcmp(arg.function, "trmm") || !strcmp(arg.function, "trmm_bad_arg");
            case TRMM_BATCHED:
                return !strcmp(arg.function, "trmm_batched")
                       || !strcmp(arg.function, "trmm_batched_bad_arg");
            case TRMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "trmm_strided_batched")
                       || !strcmp(arg.function, "trmm_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trmm_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {

                name << '_' << (char)std::toupper(arg.side) << (char)std::toupper(arg.uplo)
                     << (char)std::toupper(arg.transA) << (char)std::toupper(arg.diag) << '_'
                     << arg.M << '_' << arg.N;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(TRMM_TYPE == TRMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(TRMM_TYPE == TRMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                if(TRMM_TYPE == TRMM_STRIDED_BATCHED || TRMM_TYPE == TRMM_BATCHED)
                    name << '_' << arg.batch_count;
            }

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trmm_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trmm_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trmm"))
                testing_trmm<T>(arg);
            else if(!strcmp(arg.function, "trmm_bad_arg"))
                testing_trmm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trmm_batched"))
                testing_trmm_batched<T>(arg);
            else if(!strcmp(arg.function, "trmm_batched_bad_arg"))
                testing_trmm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trmm_strided_batched"))
                testing_trmm_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "trmm_strided_batched_bad_arg"))
                testing_trmm_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using trmm = trmm_template<trmm_testing, TRMM>;
    TEST_P(trmm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trmm);

    using trmm_batched = trmm_template<trmm_testing, TRMM_BATCHED>;
    TEST_P(trmm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trmm_batched);

    using trmm_strided_batched = trmm_template<trmm_testing, TRMM_STRIDED_BATCHED>;
    TEST_P(trmm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trmm_strided_batched);

} // namespace
