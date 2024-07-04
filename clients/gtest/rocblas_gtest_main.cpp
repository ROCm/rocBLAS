/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <string>

#include "rocblas_data.hpp"
#include "rocblas_parse_data.hpp"
#include "rocblas_test.hpp"
#include "test_cleanup.hpp"

#include "client_utility.hpp"

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* const eventListener;
    std::atomic_size_t       skipped_tests{0}; // Number of skipped tests.

public:
    bool showTestCases           = true; // Show the names of each test case.
    bool showTestNames           = true; // Show the names of each test.
    bool showSuccesses           = true; // Show each success.
    bool showInlineFailures      = true; // Show each failure as it occurs.
    bool showEnvironment         = true; // Show the setup of the global environment.
    bool showInlineSkips         = true; // Show when we skip a test.
    bool showInlineSkipTooFewGPU = false; // Only show in summary

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    void OnTestIterationStart(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpStart(unit_test);
    }

    void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
    }

    void OnTestCaseStart(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseStart(test_case);
    }

    void OnTestStart(const TestInfo& test_info) override
    {
        if(showTestNames)
            eventListener->OnTestStart(test_info);
    }

    void OnTestPartResult(const TestPartResult& result) override
    {
        if(result.type() == TestPartResult::kSkip)
        {
            ++skipped_tests;

            if(strstr(result.message(), LIMITED_RAM_STRING))
            {
                if(showInlineSkips)
                    rocblas_cout << "Skipped test due to limited RAM environment." << std::endl;
            }
            else if(strstr(result.message(), LIMITED_VRAM_STRING))
            {
                if(showInlineSkips)
                    rocblas_cout << "Skipped test due to limited GPU memory environment."
                                 << std::endl;
            }
            else if(strstr(result.message(), HMM_NOT_SUPPORTED_STRING))
            {
                if(showInlineSkips)
                    rocblas_cout << "Skipped test due to HMM not supported." << std::endl;
            }
            else if(strstr(result.message(), TOO_FEW_DEVICES_PRESENT_STRING))
            {
                if(showInlineSkipTooFewGPU) // specific default for gpu
                    rocblas_cout << "Skipped test due to too few GPUs." << std::endl;
            }
            else if(showInlineSkips)
            {
                // this is more output than the simple sentences above
                eventListener->OnTestPartResult(result);
            }
        }
        else
        {
            eventListener->OnTestPartResult(result);
        }
    }

    void OnTestEnd(const TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
            eventListener->OnTestEnd(test_info);
    }

    void OnTestCaseEnd(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseEnd(test_case);
    }

    void OnEnvironmentsTearDownStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownStart(unit_test);
    }

    void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
    }

    void OnTestIterationEnd(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const UnitTest& unit_test) override
    {
        if(skipped_tests)
            rocblas_cout << "[ SKIPPED  ] " << skipped_tests << " tests." << std::endl;
        eventListener->OnTestProgramEnd(unit_test);
    }
};

// Set the listener for Google Tests
static void rocblas_set_listener()
{
    // remove the default listener
    auto& listeners       = testing::UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // add our listener, by default everything is on (the same as using the default listener)
    // here I am turning everything off so I only see the 3 lines for the result
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto* listener       = new ConfigurableEventListener(default_printer);
    auto* gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        rocblas_cout << "environment GTEST_LISTENER=NO_PASS_LINE_IN_LOG is now the default.\n"
                        "To see pass lines use: GTEST_LISTENER=PASS_LINE_IN_LOG"
                     << std::endl;
    }

    if(gtest_listener && !strcmp(gtest_listener, "PASS_LINE_IN_LOG"))
    {
    }
    else
    {
        // default is now the same as GTEST_LISTENER=NO_PASS_LINE_IN_LOG
        listener->showTestNames      = false;
        listener->showSuccesses      = false;
        listener->showInlineFailures = true; // easier reading
        listener->showInlineSkips    = false;
    }

    listeners.Append(listener);
}

static std::string rocblas_version_string()
{
    size_t size;
    rocblas_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocblas_get_version_string(str.data(), size);
    return str;
}

// Print Version
static void rocblas_print_version()
{
    static std::string blas_version = rocblas_version_string();

    rocblas_cout << "rocBLAS version: " << blas_version << "\n" << std::endl;
}

// Print rocBLAS and Tensile commit hashes
static void rocblas_print_commit_hashes()
{
    const char* rocblas_tensile_commit_hash[] = {ROCBLAS_TENSILE_COMMIT_ID};

#if BUILD_WITH_TENSILE
    rocblas_cout << "rocBLAS-commit-hash: " << rocblas_tensile_commit_hash[0] << std::endl
                 << std::endl;
    rocblas_cout << "Tensile-commit-hash: " << rocblas_tensile_commit_hash[1] << std::endl
                 << std::endl;
#else
    rocblas_cout << "rocBLAS-commit-hash: " << rocblas_tensile_commit_hash[0] << std::endl
                 << std::endl;
    rocblas_cout << "Tensile-commit-hash: N/A, as rocBLAS was built without Tensile" << std::endl
                 << std::endl;
#endif
}

static void rocblas_print_usage_warning()
{
    std::string warning(
        "parsing of test data may take a couple minutes before any test output appears...");

    rocblas_cout << "info: " << warning << "\n" << std::endl;
}

static std::string rocblas_capture_args(int argc, char** argv)
{
    std::ostringstream cmdLine;
    cmdLine << "command line: ";
    for(int i = 0; i < argc; i++)
    {
        if(argv[i])
            cmdLine << std::string(argv[i]) << " ";
    }
    return cmdLine.str();
}

static void rocblas_print_args(const std::string& args)
{
    rocblas_cout << args << std::endl;
    rocblas_cout.flush();
}

// Device Query
static void rocblas_set_test_device()
{
    int device_id    = 0;
    int device_count = query_device_property();
    if(device_count <= device_id)
    {
        rocblas_cerr << "Error: invalid device ID. There may not be such device ID." << std::endl;
        exit(-1);
    }
    set_device(device_id);
}

/*****************
 * Main function *
 *****************/
int main(int argc, char** argv)
{
    std::string args = rocblas_capture_args(argc, argv);

    auto* no_signal_handling = getenv("ROCBLAS_TEST_NO_SIGACTION");
    if(no_signal_handling)
    {
        rocblas_cout << "rocblas-test INFO: sigactions disabled." << std::endl;
    }
    else
    {
        // Set signal handler
        rocblas_test_sigaction();
    }

    rocblas_print_version();

    // Print rocBLAS and Tensile commit hashes
    rocblas_print_commit_hashes();

    // Warn users if using older reference library
    print_reference_lib_warning();

    // Set test device
    rocblas_set_test_device();

    rocblas_print_usage_warning();

    // Set data file path
    rocblas_parse_data(argc, argv, rocblas_exepath() + "rocblas_gtest.data");

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // Set Google Test listener
    rocblas_set_listener();

    // Run the tests
    int status = RUN_ALL_TESTS();

    // Failures printed at end for reporting so repeat version info
    rocblas_print_version();

    // end test results with command line
    rocblas_print_args(args);

    //rocblas_shutdown();

    return status;
}
