/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_ger.hpp"
#include "type_dispatch.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct ger_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct ger_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "ger"))
            testing_ger<T>(arg);
        else if(!strcmp(arg.function, "ger_bad_arg"))
            testing_ger_bad_arg<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct ger : RocBLAS_Test<ger, ger_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "ger") || !strcmp(arg.function, "ger_bad_arg");
    }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<ger>{} << rocblas_datatype2string(arg.a_type) << '_' << arg.M << '_'
                                       << arg.N << '_' << arg.alpha << '_' << arg.incx << '_'
                                       << arg.incy << '_' << arg.lda;
    }
};

TEST_P(ger, blas2) { rocblas_simple_dispatch<ger_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(ger);

} // namespace
