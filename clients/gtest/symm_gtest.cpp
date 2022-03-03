/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_symm_hemm.hpp"
#include "testing_symm_hemm_batched.hpp"
#include "testing_symm_hemm_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum symm_test_type
    {
        SYMM,
        SYMM_BATCHED,
        SYMM_STRIDED_BATCHED,
    };

    // test template
    template <template <typename...> class FILTER, symm_test_type SYMM_TYPE>
    struct symm_template : RocBLAS_Test<symm_template<FILTER, SYMM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<symm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYMM_TYPE)
            {
            case SYMM:
                return !strcmp(arg.function, "symm") || !strcmp(arg.function, "symm_bad_arg");
            case SYMM_BATCHED:
                return !strcmp(arg.function, "symm_batched")
                       || !strcmp(arg.function, "symm_batched_bad_arg");
            case SYMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "symm_strided_batched")
                       || !strcmp(arg.function, "symm_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<symm_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.side) << (char)std::toupper(arg.uplo) << '_'
                     << arg.M << '_' << arg.N;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(SYMM_TYPE == SYMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(SYMM_TYPE == SYMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                if(arg.b_type == rocblas_datatype_f32_c || arg.b_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_beta<rocblas_float_complex>();
                else
                    name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldc;

                if(SYMM_TYPE == SYMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(SYMM_TYPE == SYMM_STRIDED_BATCHED || SYMM_TYPE == SYMM_BATCHED)
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
    struct symm_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct symm_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            // testing_symm second argument is Hermitian to perform hemm function
            if(!strcmp(arg.function, "symm"))
                testing_symm_hemm<T, false>(arg);
            else if(!strcmp(arg.function, "symm_bad_arg"))
                testing_symm_hemm_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "symm_batched"))
                testing_symm_hemm_batched<T, false>(arg);
            else if(!strcmp(arg.function, "symm_batched_bad_arg"))
                testing_symm_hemm_batched_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "symm_strided_batched"))
                testing_symm_hemm_strided_batched<T, false>(arg);
            else if(!strcmp(arg.function, "symm_strided_batched_bad_arg"))
                testing_symm_hemm_strided_batched_bad_arg<T, false>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using symm = symm_template<symm_testing, SYMM>;
    TEST_P(symm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<symm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symm);

    using symm_batched = symm_template<symm_testing, SYMM_BATCHED>;
    TEST_P(symm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<symm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symm_batched);

    using symm_strided_batched = symm_template<symm_testing, SYMM_STRIDED_BATCHED>;
    TEST_P(symm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<symm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symm_strided_batched);

} // namespace
