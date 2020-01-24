/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_spr.hpp"
#include "testing_spr_batched.hpp"
#include "testing_spr_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum spr_test_type
    {
        SPR,
        SPR_BATCHED,
        SPR_STRIDED_BATCHED,
    };

    //spr test template
    template <template <typename...> class FILTER, spr_test_type SPR_TYPE>
    struct spr_template : RocBLAS_Test<spr_template<FILTER, SPR_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<spr_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SPR_TYPE)
            {
            case SPR:
                return !strcmp(arg.function, "spr") || !strcmp(arg.function, "spr_bad_arg");
            case SPR_BATCHED:
                return !strcmp(arg.function, "spr_batched")
                       || !strcmp(arg.function, "spr_batched_bad_arg");
            case SPR_STRIDED_BATCHED:
                return !strcmp(arg.function, "spr_strided_batched")
                       || !strcmp(arg.function, "spr_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<spr_template> name;

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_';

                // T in get_alpha doesn't matter besides being real/complex.
                if(arg.a_type == rocblas_datatype_f32_r || arg.a_type == rocblas_datatype_f64_r)
                    name << arg.get_alpha<double>();
                else
                    name << arg.get_alpha<rocblas_double_complex>();

                name << '_' << arg.incx;

                if(SPR_TYPE == SPR_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                if(SPR_TYPE == SPR_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(SPR_TYPE == SPR_STRIDED_BATCHED || SPR_TYPE == SPR_BATCHED)
                    name << '_' << arg.batch_count;
            }
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct spr_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct spr_testing<T,
                       std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                        || std::is_same<T, rocblas_float_complex>{}
                                        || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "spr"))
                testing_spr<T>(arg);
            else if(!strcmp(arg.function, "spr_bad_arg"))
                testing_spr_bad_arg<T>();
            else if(!strcmp(arg.function, "spr_batched"))
                testing_spr_batched<T>(arg);
            else if(!strcmp(arg.function, "spr_batched_bad_arg"))
                testing_spr_batched_bad_arg<T>();
            else if(!strcmp(arg.function, "spr_strided_batched"))
                testing_spr_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "spr_strided_batched_bad_arg"))
                testing_spr_strided_batched_bad_arg<T>();
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using spr = spr_template<spr_testing, SPR>;
    TEST_P(spr, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<spr_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spr);

    using spr_batched = spr_template<spr_testing, SPR_BATCHED>;
    TEST_P(spr_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<spr_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spr_batched);

    using spr_strided_batched = spr_template<spr_testing, SPR_STRIDED_BATCHED>;
    TEST_P(spr_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<spr_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spr_strided_batched);

} // namespace
