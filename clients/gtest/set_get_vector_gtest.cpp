/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include "testing_set_get_vector.hpp"
#include "type_dispatch.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct set_get_vector_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct set_get_vector_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "set_get_vector"))
            testing_set_get_vector<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct set_get_vector : RocBLAS_Test<set_get_vector, set_get_vector_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "set_get_vector");
    }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<set_get_vector>{} << rocblas_datatype2string(arg.a_type) << '_'
                                                  << arg.M << '_' << arg.incx << '_' << arg.incy
                                                  << '_' << arg.incb;
    }
};

TEST_P(set_get_vector, auxilliary) { rocblas_simple_dispatch<set_get_vector_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(set_get_vector);

} // namespace
