/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_trsv.hpp"
#include "type_dispatch.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct trsv_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct trsv_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "trsv"))
            testing_trsv<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct trsv : RocBLAS_Test<trsv, trsv_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg) { return !strcmp(arg.function, "trsv"); }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<trsv>{}
               << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
               << (char)std::toupper(arg.transA) << (char)std::toupper(arg.diag) << '_' << arg.M
               << '_' << arg.lda << '_' << arg.incx;
    }
};

TEST_P(trsv, blas2) { rocblas_simple_dispatch<trsv_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(trsv);

} // namespace
