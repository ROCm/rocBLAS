/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_set_get_matrix.hpp"
#include "testing_set_get_matrix_async.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    enum sync_type
    {
        SET_GET_MATRIX_SYNC,
        SET_GET_MATRIX_ASYNC,
    };

    template <template <typename...> class FILTER, sync_type TRANSFER_TYPE>
    struct matrix_set_get_template
        : RocBLAS_Test<matrix_set_get_template<FILTER, TRANSFER_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<matrix_set_get_template::template type_filter_functor>(
                arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRANSFER_TYPE)
            {
            case SET_GET_MATRIX_SYNC:
                return !strcmp(arg.function, "set_get_matrix_sync");
            case SET_GET_MATRIX_ASYNC:
                return !strcmp(arg.function, "set_get_matrix_async");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<matrix_set_get_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << arg.M << '_' << arg.N << '_' << arg.lda << '_' << arg.ldb;
            }
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct set_get_matrix_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct set_get_matrix_testing<
        T,
        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_matrix_sync"))
                testing_set_get_matrix<T>(arg);
            else if(!strcmp(arg.function, "set_get_matrix_async"))
                testing_set_get_matrix_async<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using set_get_matrix_sync
        = matrix_set_get_template<set_get_matrix_testing, SET_GET_MATRIX_SYNC>;
    TEST_P(set_get_matrix_sync, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_matrix_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix_sync);

    using set_get_matrix_async
        = matrix_set_get_template<set_get_matrix_testing, SET_GET_MATRIX_ASYNC>;
    TEST_P(set_get_matrix_async, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_matrix_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix_async);

} // namespace
