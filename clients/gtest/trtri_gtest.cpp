/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#include "testing_trtri_strided_batched.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trtri_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trtri_testing<T,
                         std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                          || std::is_same<T, rocblas_float_complex>{}
                                          || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trtri"))
                testing_trtri<T>(arg);
            else if(!strcmp(arg.function, "trtri_batched"))
                testing_trtri_batched<T>(arg);
            else if(!strcmp(arg.function, "trtri_strided_batched"))
                testing_trtri_strided_batched<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    enum trtri_kind
    {
        trtri_k,
        trtri_batched_k,
        trtri_strided_batched_k,
    };

    template <trtri_kind K>
    struct trtri_template : RocBLAS_Test<trtri_template<K>, trtri_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trtri_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            if(K == trtri_k)
                return !strcmp(arg.function, "trtri");
            else if(K == trtri_batched_k)
                return !strcmp(arg.function, "trtri_batched");
            else
                return !strcmp(arg.function, "trtri_strided_batched");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trtri_template> name;
            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
                 << (char)std::toupper(arg.diag) << '_' << arg.N << '_' << arg.lda;
            if(K != trtri_k)
                name << '_' << arg.batch_count;

            if(arg.fortran)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    using trtri = trtri_template<trtri_k>;
    TEST_P(trtri, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri);

    using trtri_batched = trtri_template<trtri_batched_k>;
    TEST_P(trtri_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri_batched);

    using trtri_strided_batched = trtri_template<trtri_strided_batched_k>;
    TEST_P(trtri_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri_strided_batched);

} // namespace
