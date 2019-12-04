/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_tbmv.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible tbmv test cases
    enum tbmv_test_type
    {
        TBMV,
    };

    //tbmv test template
    template <template <typename...> class FILTER, tbmv_test_type TBMV_TYPE>
    struct tbmv_template : RocBLAS_Test<tbmv_template<FILTER, TBMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<tbmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TBMV_TYPE)
            {
            case TBMV:
                return !strcmp(arg.function, "tbmv") || !strcmp(arg.function, "tbmv_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<tbmv_template> name;

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
                 << '_' << (char)std::toupper(arg.transA) << '_' << (char)std::toupper(arg.diag)
                 << '_' << arg.M << '_' << arg.K << '_' << arg.lda;

            name << '_' << arg.incx;

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct tbmv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct tbmv_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}
                                || std::is_same<T, rocblas_float_complex>{}
                                || std::is_same<T, rocblas_double_complex>{}>::type>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "tbmv"))
                testing_tbmv<T>(arg);
            else if(!strcmp(arg.function, "tbmv_bad_arg"))
                testing_tbmv_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using tbmv = tbmv_template<tbmv_testing, TBMV>;
    TEST_P(tbmv, blas2)
    {
        rocblas_simple_dispatch<tbmv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(tbmv);

} // namespace
