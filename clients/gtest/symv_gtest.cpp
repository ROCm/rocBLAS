/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_symv.hpp"
#include "type_dispatch.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename, typename = void>
struct symv_testing : rocblas_test_invalid
{
};

#if 0 // TODO: Right now rocblas_symv is unimplemented

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct symv_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "symv"))
            testing_symv<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

#endif

struct symv : RocBLAS_Test<symv, symv_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg) { return !strcmp(arg.function, "symv"); }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<symv>{}
               << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo) << '_'
               << arg.N << '_' << arg.alpha << '_' << arg.lda << '_' << arg.incx << '_' << arg.beta
               << '_' << arg.incy;
    }
};

TEST_P(symv, blas2) { rocblas_simple_dispatch<symv_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(symv);

} // namespace
