/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <string>

#include "rocblas_data.hpp"
#include "rocblas_parse_data.hpp"
#include "rocblas_test.hpp"
#include "test_cleanup.hpp"
#include "utility.hpp"

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* const eventListener;
    std::atomic_size_t       skipped_tests{0}; // Number of skipped tests.

public:
    bool showTestCases      = true; // Show the names of each test case.
    bool showTestNames      = true; // Show the names of each test.
    bool showSuccesses      = true; // Show each success.
    bool showInlineFailures = true; // Show each failure as it occurs.
    bool showEnvironment    = true; // Show the setup of the global environment.
    bool showInlineSkips    = true; // Show when we skip a test.

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
        if(!strcmp(result.message(), LIMITED_MEMORY_STRING_GTEST))
        {
            if(showInlineSkips)
                rocblas_cout << "Skipped test due to limited memory environment." << std::endl;
            ++skipped_tests;
        }
        else if(!strcmp(result.message(), TOO_MANY_DEVICES_STRING_GTEST))
        {
            if(showInlineSkips)
                rocblas_cout << "Skipped test due to too few GPUs." << std::endl;
            ++skipped_tests;
        }
        eventListener->OnTestPartResult(result);
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
        listener->showTestNames      = false;
        listener->showSuccesses      = false;
        listener->showInlineFailures = false;
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

    // Set signal handler
    rocblas_test_sigaction();

    rocblas_print_version();

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
