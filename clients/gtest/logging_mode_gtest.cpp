/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_logging.hpp"
#include "type_dispatch.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct logging_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct logging_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "logging"))
            testing_logging<T>();
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct logging : RocBLAS_Test<logging, logging_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg) { return !strcmp(arg.function, "logging"); }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<logging>{} << rocblas_datatype2string(arg.a_type);
    }
};

TEST_P(logging, auxilliary) { rocblas_simple_dispatch<logging_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(logging);

} // namespace
