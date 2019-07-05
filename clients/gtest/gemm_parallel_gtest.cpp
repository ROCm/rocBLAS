/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "testing_gemm.hpp"
#include "testing_gemm_parallel.hpp"
#include <algorithm>
#include <future>
#include <list>
#include <thread>
#include <unordered_map>
//#include <omp.h>

/* =====================================================================
     BLAS-3 GEMM:
   =================================================================== */

namespace
{
#if 0 // TODO: Need to refactor into new framework; currently disabled because of HIP bug

// Parallel GEMM testing class
struct parallel_gemm : testing::TestWithParam<std::vector<Arguments>>
{
    // Filter for which tests get into gemm right now
    static bool filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "gemm") && (arg.unit_check || arg.norm_check) &&
            !arg.timing;
    }
};

using parallel_gemm_synchronized = parallel_gemm;

TEST_P(parallel_gemm_synchronized, DISABLED_test)
{
    std::vector<Arguments> args = GetParam();
    std::random_shuffle(args.begin(), args.end());

    const int max_threads = 1;
    std::list<std::future<void>> futures;

    std::condition_variable cv;
    int waiting_threads = 0;
    int num_threads     = max_threads;

    std::cout << "Running " << args.size() << " tests in " << max_threads << " threads."
              << std::endl;

    for(auto it = args.begin(); it != args.end(); it++)
    {
        Arguments const& arg = *it;

        if(futures.empty())
        {
            num_threads = std::max<int>(max_threads, args.end() - it);
        }

        futures.emplace_back(async(std::launch::async, [&, this, num_threads]() {
            testing_gemm_parallel<float>(arg, cv, waiting_threads, num_threads);
        }));

        if(futures.size() == num_threads)
        {
            for(auto& future : futures)
                future.wait();
            futures.clear();
            waiting_threads = 0;

            CHECK_HIP_ERROR(hipDeviceReset());
        }
    }

    for(auto& future : futures)
        future.wait();
}

TEST_P(parallel_gemm, DISABLED_test)
{
    std::vector<Arguments> args = GetParam();
    std::random_shuffle(args.begin(), args.end());

    const int max_threads = 64;
    std::list<std::future<void>> futures;

    std::cout << "Running " << args.size() << " tests in " << max_threads << " threads."
              << std::endl;

    int i = 0;
    for(auto const& arg : args)
    {
        while(futures.size() >= max_threads)
        {
            futures.front().wait();
            futures.pop_front();
        }

        futures.emplace_back(
            async(std::launch::async, [&, this, i]() { testing_gemm<float>(arg); }));
        i++;
    }

    for(auto& future : futures)
        future.wait();
}

INSTANTIATE_TEST_CASE_P(
    parallel,
    parallel_gemm_synchronized,
    testing::Values(std::vector<Arguments>(RocBLAS_TestData::begin([](const Arguments& arg) {
                                               return arg.a_type == rocblas_datatype_f32_r &&
                                                      !strcmp(arg.category, "quick") &&
                                                      parallel_gemm::filter(arg);
                                           }),
                                           RocBLAS_TestData::end())));

INSTANTIATE_TEST_CASE_P(
    parallel,
    parallel_gemm,
    testing::Values(std::vector<Arguments>(RocBLAS_TestData::begin([](const Arguments& arg) {
                                               return arg.a_type == rocblas_datatype_f32_r &&
                                                      !strcmp(arg.category, "quick") &&
                                                      parallel_gemm::filter(arg);
                                           }),
                                           RocBLAS_TestData::end())));

#endif

} // namespace
