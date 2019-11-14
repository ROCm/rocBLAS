/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_gemv.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct gemv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gemv_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}
                                || std::is_same<T, rocblas_float_complex>{}
                                || std::is_same<T, rocblas_double_complex>{}>::type>
    {
        explicit operator bool()
        {
            return true;
        }
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemv"))
                testing_gemv<T>(arg);
            else if(!strcmp(arg.function, "gemv_bad_arg"))
                testing_gemv_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct gemv : RocBLAS_Test<gemv, gemv_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "gemv") || !strcmp(arg.function, "gemv_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<gemv>{}
                   << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.transA)
                   << '_' << arg.M << '_' << arg.N << '_' << arg.alpha << '_' << arg.alphai << '_'
                   << arg.lda << '_' << arg.incx << '_' << arg.beta << '_' << arg.betai << '_'
                   << arg.incy;
        }
    };

    TEST_P(gemv, blas2)
    {
        rocblas_simple_dispatch<gemv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(gemv);

} // namespace
