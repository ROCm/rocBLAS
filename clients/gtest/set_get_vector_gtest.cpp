/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_set_get_vector.hpp"
#include "testing_set_get_vector_async.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    enum sync_type
    {
        SET_GET_VECTOR_SYNC,
        SET_GET_VECTOR_ASYNC,
    };

    template <template <typename...> class FILTER, sync_type TRANSFER_TYPE>
    struct vec_set_get_template : RocBLAS_Test<vec_set_get_template<FILTER, TRANSFER_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<vec_set_get_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRANSFER_TYPE)
            {
            case SET_GET_VECTOR_SYNC:
                return !strcmp(arg.function, "set_get_vector_sync");
            case SET_GET_VECTOR_ASYNC:
                return !strcmp(arg.function, "set_get_vector_async");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<vec_set_get_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << arg.M << '_' << arg.incx << '_' << arg.incy << '_' << arg.incb;
            }
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct set_get_vector_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct set_get_vector_testing<
        T,
        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_vector_sync"))
                testing_set_get_vector<T>(arg);
            else if(!strcmp(arg.function, "set_get_vector_async"))
                testing_set_get_vector_async<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using set_get_vector_sync = vec_set_get_template<set_get_vector_testing, SET_GET_VECTOR_SYNC>;
    TEST_P(set_get_vector_sync, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_vector_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_vector_sync);

    using set_get_vector_async = vec_set_get_template<set_get_vector_testing, SET_GET_VECTOR_ASYNC>;
    TEST_P(set_get_vector_async, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_vector_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_vector_async);

} // namespace
