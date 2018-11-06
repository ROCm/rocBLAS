/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include "testing_gemm.hpp"
#include <unordered_map>

#include <algorithm>
#include <list>
#include <future>
#include <thread>
//#include <omp.h>

using namespace std;

/* =====================================================================
     BLAS-3 GEMM:
   =================================================================== */

namespace {

// GEMM testing class
struct gemm : ::testing::TestWithParam<Arguments>
{
    // Filter for which tests get into gemm right now
    static function<bool(const Arguments&)> filter()
    {
        return [](const Arguments& arg) {
            return !strcmp(arg.function, "testing_gemm") ||
                   !strcmp(arg.function, "testing_gemm_NaN") ||
                   !strcmp(arg.function, "testing_gemm_bad_arg");
        };
    }

    struct PrintToStringParamName
    {
        template <class ParamType>
        string operator()(const ParamType& info) const
        {
            auto arg = info.param;
            static unordered_map<string, size_t> hit;
            ostringstream strm;
            strm << rocblas_datatype2char(arg.a_type) << '_' << arg.transA_option << '_'
                 << arg.transB_option << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_'
                 << arg.lda << '_' << arg.ldb << '_' << arg.ldc << '_' << arg.alpha << '_'
                 << arg.beta;
            return normalized_test_name(strm.str(), hit);
        }
    };
};

template <class... T>
void testit(const Arguments& arg)
{
    if(!strcmp(arg.function, "testing_gemm"))
    {
        rocblas_status status = testing_gemm<T...>(arg);

        if(arg.M < 0 || arg.N < 0 || arg.K < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else
        {
            EXPECT_EQ(rocblas_status_success, status);
        }
    }
    else if(!strcmp(arg.function, "testing_gemm_NaN"))
    {
        testing_gemm_NaN<T...>(arg);
    }
    else if(!strcmp(arg.function, "testing_gemm_bad_arg"))
    {
        testing_gemm_bad_arg<T...>();
    }
    else
    {
        FAIL() << "Unknown test type.";
    }
}

TEST_P(gemm, test)
{
    const Arguments& arg = GetParam();
    switch(arg.a_type)
    {
    default: FAIL() << "Unknown data type"; break;
    case rocblas_datatype_f64_r: return testit<double>(arg);
    case rocblas_datatype_f32_r: return testit<float>(arg);
    case rocblas_datatype_f16_r:
        return testit<half>(arg);
        // case rocblas_datatype_f64_c: return testit<rocblas_double_complex>(arg);
        // case rocblas_datatype_f32_c: return testit<rocblas_float_complex>(arg);
        // case rocblas_datatype_f16_c: return testit<rocblas_half_complex>(arg);
        //
        // TODO:^   NOTE: testit<...> is ready for multiple data types, if
        // a_type, c_type, compute_type, etc. need to be passed separately.
    }
}

// The tests are instantiated by filtering through the RocBLAS_Data stream
#define INSTANTIATE_CATEGORY(cat)                                                                  \
    INSTANTIATE_TEST_CASE_P(cat,                                                                   \
                            gemm,                                                                  \
                            ::testing::ValuesIn(RocBLAS_TestData::begin([](const Arguments& arg) { \
                                                    return !strcmp(arg.category, #cat) &&          \
                                                           gemm::filter()(arg);                    \
                                                }),                                                \
                                                RocBLAS_TestData::end()),                          \
                            gemm::PrintToStringParamName());

INSTANTIATE_CATEGORY(quick)
INSTANTIATE_CATEGORY(pre_checkin)
INSTANTIATE_CATEGORY(nightly)
INSTANTIATE_CATEGORY(known_bug)

// Parallel GEMM testing class
struct parallel_gemm : ::testing::TestWithParam<std::vector<Arguments>>
{
    // Filter for which tests get into gemm right now
    static function<bool(const Arguments&)> filter()
    {
        return [](const Arguments& arg) { return !strcmp(arg.function, "testing_gemm"); };
    }
};

#if 1
TEST_P(parallel_gemm, test)
{
    std::vector<Arguments> args = GetParam();
    std::random_shuffle(args.begin(), args.end());

    // should up this to 64 once the 2-thread case passes.
    const int max_threads = 2;
    std::list<std::future<void>> futures;

    int i = 0;
    for(auto const& arg : args)
    {
        while(futures.size() >= max_threads)
        {
            futures.front().wait();
            futures.pop_front();
        }

        futures.emplace_back(async(launch::async, [&, this, i]() {
            rocblas_status status = testing_gemm<float>(arg);

            if(arg.M < 0 || arg.N < 0 || arg.K < 0)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else if(arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else if(arg.ldc < arg.M)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else
            {
                EXPECT_EQ(rocblas_status_success, status);
            }
        }));
        i++;
    }

    for(auto& future : futures)
        future.wait();
}
#else
TEST_P(parallel_gemm, test)
{
    std::vector<Arguments> args = GetParam();
    std::random_shuffle(args.begin(), args.end());

#pragma omp parallel for
    for(int i = 0; i < args.size(); i++)
    {
        auto arg = args[i];
        EXPECT_GT(omp_get_num_threads(), 1);
        rocblas_status status = testing_gemm<float>(arg);

        if(arg.M < 0 || arg.N < 0 || arg.K < 0)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else if(arg.ldc < arg.M)
        {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
        else
        {
            EXPECT_EQ(rocblas_status_success, status);
        }
    }
}
#endif

INSTANTIATE_TEST_CASE_P(
    parallel,
    parallel_gemm,
    ::testing::Values(vector<Arguments>(RocBLAS_TestData::begin([](const Arguments& arg) {
                                            return arg.a_type == rocblas_datatype_f32_r &&
                                                   !strcmp(arg.category, "quick") &&
                                                   parallel_gemm::filter()(arg);
                                        }),
                                        RocBLAS_TestData::end())));

} // namespace
