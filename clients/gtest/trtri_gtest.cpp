/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <type_traits>
#include <cstring>
#include <cctype>
#include "rocblas_test.hpp"
#include "rocblas_data.hpp"
#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#include "type_dispatch.hpp"
#include "rocblas_datatype2string.hpp"

namespace {

// By default, this test does not apply to any types.
// The unnamed second parameter is used for enable_if below.
template <typename T, typename = void>
struct trtri_testing : rocblas_test_invalid
{
};

// When the condition in the second argument is satisfied, the type combination
// is valid. When the condition is false, this specialization does not apply.
template <typename T>
struct trtri_testing<
    T,
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value>::type>
{
    explicit operator bool() { return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "testing_trtri"))
            testing_trtri<T>(arg);
        else if(!strcmp(arg.function, "testing_trtri_batched"))
            testing_trtri_batched<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};

enum trtri_kind
{
    trtri_k,
    trtri_batched_k
};

template <trtri_kind K>
struct trtri_template : RocBLAS_Test<trtri_template<K>, trtri_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<trtri_template::template type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return K == trtri_k ? !strcmp(arg.function, "testing_trtri")
                            : !strcmp(arg.function, "testing_trtri_batched");
    }

    // Goggle Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        RocBLAS_TestName<trtri_template> name;
        name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
             << (char)std::toupper(arg.diag) << '_' << arg.N << '_' << arg.lda;
        if(K == trtri_batched_k)
            name << '_' << arg.batch_count;
        return std::move(name);
    }
};

using trtri = trtri_template<trtri_k>;
TEST_P(trtri, blas3) { rocblas_simple_dispatch<trtri_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(trtri);

using trtri_batched = trtri_template<trtri_batched_k>;
TEST_P(trtri_batched, blas3) { rocblas_simple_dispatch<trtri_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(trtri_batched);

} // namespace
