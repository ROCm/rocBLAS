/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_geam.hpp"
#include "type_dispatch.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct geam_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct geam_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "geam"))
            testing_geam<T>(arg);
        else if(!strcmp(arg.function, "geam_bad_arg"))
            testing_geam_bad_arg<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct geam : RocBLAS_Test<geam, geam_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "geam") || !strcmp(arg.function, "geam_bad_arg");
    }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<geam>{}
               << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.transA)
               << (char)std::toupper(arg.transB) << '_' << arg.M << '_' << arg.N << '_' << arg.alpha
               << '_' << arg.lda << '_' << arg.beta << '_' << arg.ldb << '_' << arg.ldc;
    }
};

TEST_P(geam, blas3) { rocblas_simple_dispatch<geam_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(geam);

} // namespace
