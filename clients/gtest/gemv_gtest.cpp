/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_gemv.hpp"
#include "type_dispatch.hpp"

namespace {

// By default, arbitrary type combinations are invalid.
// The unnamed second parameter is used for enable_if below.
template <typename T, typename = void>
struct gemv_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct gemv_testing<
    T,
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "testing_gemv"))
            testing_gemv<T>(arg);
        else if(!strcmp(arg.function, "testing_gemv_bad_arg"))
            testing_gemv_bad_arg<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct gemv : RocBLAS_Test<gemv, gemv_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "testing_gemv") ||
               !strcmp(arg.function, "testing_gemv_bad_arg");
    }

    // Goggle Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<gemv>() << rocblas_datatype2string(arg.a_type) << '_'
                                        << (char)std::toupper(arg.transA) << '_' << arg.M << '_'
                                        << arg.N << '_' << arg.alpha << '_' << arg.lda << '_'
                                        << arg.incx << '_' << arg.beta << '_' << arg.incy;
    }
};

TEST_P(gemv, blas2) { rocblas_simple_dispatch<gemv_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(gemv);

} // namespace
