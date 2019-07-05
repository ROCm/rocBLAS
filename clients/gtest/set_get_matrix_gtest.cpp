/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_set_get_matrix.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct set_get_matrix_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct set_get_matrix_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
    {
        explicit operator bool()
        {
            return true;
        }
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_matrix"))
                testing_set_get_matrix<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct set_get_matrix : RocBLAS_Test<set_get_matrix, set_get_matrix_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_matrix");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_matrix>{} << rocblas_datatype2string(arg.a_type) << '_'
                                                      << arg.M << '_' << arg.N << '_' << arg.lda
                                                      << '_' << arg.ldb;
        }
    };

    TEST_P(set_get_matrix, auxilliary)
    {
        rocblas_simple_dispatch<set_get_matrix_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix);

} // namespace
