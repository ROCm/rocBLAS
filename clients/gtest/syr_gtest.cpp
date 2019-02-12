/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_syr.hpp"
#include "type_dispatch.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename T, typename = void>
struct syr_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct syr_testing<
    T,
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "testing_syr"))
            testing_syr<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

struct syr : RocBLAS_Test<syr, syr_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "testing_syr");
    }

    // Goggle Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<syr>() << rocblas_datatype2string(arg.a_type) << '_'
                                       << (char)std::toupper(arg.uplo) << '_' << arg.N << '_'
                                       << arg.alpha << '_' << arg.incx << '_' << arg.lda;
    }
};

TEST_P(syr, blas2) { rocblas_simple_dispatch<syr_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(syr);

} // namespace
