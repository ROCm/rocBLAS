/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_TEST_H_
#define ROCBLAS_TEST_H_

#include <cstdio>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include "rocblas.h"
#include "rocblas_arguments.hpp"
#include "test_cleanup.hpp"

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define EXPECT_ROCBLAS_STATUS ASSERT_EQ

#else // GOOGLE_TEST

inline const char* rocblas_status_to_string(rocblas_status status)
{
    switch(status)
    {
    case rocblas_status_success: return "rocblas_status_success";
    case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
    case rocblas_status_memory_error: return "rocblas_status_memory_error";
    case rocblas_status_internal_error: return "rocblas_status_internal_error";
    default: return "<undefined rocblas_status value>";
    }
}

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        std::cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                  << ", received " << rocblas_status_to_string(status) << std::endl;
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                    \
    do                                            \
    {                                             \
        auto error = ERROR;                       \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#endif // GOOGLE_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

#ifdef GOOGLE_TEST

// The tests are instantiated by filtering through the RocBLAS_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, categ0ry)                                           \
    INSTANTIATE_TEST_CASE_P(categ0ry,                                                            \
                            testclass,                                                           \
                            testing::ValuesIn(RocBLAS_TestData::begin([](const Arguments& arg) { \
                                                  return !strcmp(arg.category, #categ0ry) &&     \
                                                         testclass::type_filter(arg) &&          \
                                                         testclass::function_filter(arg);        \
                                              }),                                                \
                                              RocBLAS_TestData::end()),                          \
                            testclass::PrintToStringParamName());

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)        \
    INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
    INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
    INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
    INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

/* ============================================================================================ */
/*! \brief  Normalized test name to conform to Google Tests */
// Template parameter is used to generate multiple instantiations
template <typename>
class RocBLAS_TestName
{
    using table_t = std::unordered_map<std::string, size_t>;
    std::ostringstream str;

    table_t& get_table()
    {
        // Placed inside function to avoid dependency on initialization order
        static table_t* table = test_cleanup::allocate<table_t>(table);
        return *table;
    }

    public:
    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This table is private to each instantation of RocBLAS_TestName
        table_t& table = get_table();
        std::string name(str.str());

        if(name == "")
            name = "1";

        // Warn about unset letter parameters
        if(name.find('*') != name.npos)
            fputs("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                  "Warning: Character * found in name."
                  " This means a required letter parameter\n"
                  "(e.g., transA, diag, etc.) has not been set in the YAML file."
                  " Check the YAML file.\n"
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
                  stderr);

        // Replace non-alphanumeric characters with letters
        std::replace(name.begin(), name.end(), '-', 'n'); // minus
        std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

        // Complex (A,B) is replaced with ArBi
        name.erase(std::remove(name.begin(), name.end(), '('), name.end());
        std::replace(name.begin(), name.end(), ',', 'r');
        std::replace(name.begin(), name.end(), ')', 'i');

        // If parameters are repeated, append an incrementing suffix
        auto p = table.find(name);
        if(p != table.end())
            name += "_t" + std::to_string(++p->second);
        else
            table[name] = 1;

        return name;
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend RocBLAS_TestName& operator<<(RocBLAS_TestName& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend RocBLAS_TestName&& operator<<(RocBLAS_TestName&& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return std::move(name);
    }

    RocBLAS_TestName()                        = default;
    RocBLAS_TestName(const RocBLAS_TestName&) = delete;
    RocBLAS_TestName& operator=(const RocBLAS_TestName&) = delete;
};

// ----------------------------------------------------------------------------
// RocBLAS_Test base class. All non-legacy rocBLAS Google tests derive from it.
// It defines a type_filter_functor() and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class RocBLAS_Test : public testing::TestWithParam<Arguments>
{
    protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments&) { return static_cast<bool>(FILTER<T...>{}); }
    };

    public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            return TEST::name_suffix(info.param);
        }
    };
};

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct rocblas_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    explicit operator bool() { return false; }

    // If this specialization is actually called, print fatal error message
    void operator()(const Arguments&)
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types\n";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        fputs(msg, stderr);
        exit(EXIT_FAILURE);
#endif
    }
};

#endif
