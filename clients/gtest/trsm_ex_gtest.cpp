/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_trsm_ex.hpp"
#include "type_dispatch.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct trsm_ex_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct trsm_ex_testing<
    T,
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "trsm_ex"))
            testing_trsm_ex<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct trsm_ex : RocBLAS_Test<trsm_ex, trsm_ex_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg) { return !strcmp(arg.function, "trsm_ex"); }

    // Goggle Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<trsm_ex>{}
               << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.side)
               << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA)
               << (char)std::toupper(arg.diag) << '_' << arg.M << '_' << arg.N << '_' << arg.alpha
               << '_' << arg.lda << '_' << arg.ldb;
    }
};

TEST_P(trsm_ex, blas3) { rocblas_simple_dispatch<trsm_ex_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(trsm_ex);

} // namespace
