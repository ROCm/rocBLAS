/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_geam.hpp"
#include "testing_geam_batched.hpp"
#include "testing_geam_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible geam test cases
    enum geam_test_type
    {
        GEAM,
        GEAM_BATCHED,
        GEAM_STRIDED_BATCHED,
    };

    //geam test template
    template <template <typename...> class FILTER, geam_test_type GEAM_TYPE>
    struct geam_template : RocBLAS_Test<geam_template<FILTER, GEAM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<geam_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEAM_TYPE)
            {
            case GEAM:
                return !strcmp(arg.function, "geam") || !strcmp(arg.function, "geam_bad_arg");
            case GEAM_BATCHED:
                return !strcmp(arg.function, "geam_batched")
                       || !strcmp(arg.function, "geam_batched_bad_arg");
            case GEAM_STRIDED_BATCHED:
                return !strcmp(arg.function, "geam_strided_batched")
                       || !strcmp(arg.function, "geam_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<geam_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB)
                     << '_' << arg.M << '_' << arg.N;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(GEAM_TYPE == GEAM_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(arg.b_type == rocblas_datatype_f32_c || arg.b_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_beta<rocblas_float_complex>();
                else
                    name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldb;

                if(GEAM_TYPE == GEAM_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                name << '_' << arg.ldc;

                if(GEAM_TYPE == GEAM_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(GEAM_TYPE == GEAM_STRIDED_BATCHED || GEAM_TYPE == GEAM_BATCHED)
                    name << '_' << arg.batch_count;
            }

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct geam_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct geam_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "geam"))
                testing_geam<T>(arg);
            else if(!strcmp(arg.function, "geam_bad_arg"))
                testing_geam_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "geam_batched"))
                testing_geam_batched<T>(arg);
            else if(!strcmp(arg.function, "geam_batched_bad_arg"))
                testing_geam_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "geam_strided_batched"))
                testing_geam_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "geam_strided_batched_bad_arg"))
                testing_geam_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using geam = geam_template<geam_testing, GEAM>;
    TEST_P(geam, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<geam_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geam);

    using geam_batched = geam_template<geam_testing, GEAM_BATCHED>;
    TEST_P(geam_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<geam_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geam_batched);

    using geam_strided_batched = geam_template<geam_testing, GEAM_STRIDED_BATCHED>;
    TEST_P(geam_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<geam_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geam_strided_batched);

} // namespace
