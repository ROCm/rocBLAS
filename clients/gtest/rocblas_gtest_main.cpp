/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_parse_data.hpp"
#include "rocblas_test.hpp"
#include "test_cleanup.hpp"
#include "utility.hpp"

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* eventListener;

public:
    bool showTestCases; // Show the names of each test case.
    bool showTestNames; // Show the names of each test.
    bool showSuccesses; // Show each success.
    bool showInlineFailures; // Show each failure as it occurs.
    bool showEnvironment; // Show the setup of the global environment.
    bool showInlineSkips; // Show when we skip a test.
    int  skipped_tests; // Number of skipped tests.

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
        , showTestCases(true)
        , showTestNames(true)
        , showSuccesses(true)
        , showInlineFailures(true)
        , showEnvironment(true)
        , showInlineSkips(true)
        , skipped_tests(0)
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
        if(strcmp(result.message(), LIMITED_MEMORY_STRING_GTEST) == 0)
        {
            skipped_tests++;
            if(showInlineSkips)
                rocblas_cout << "Skipped test due to limited memory environment." << std::endl;
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

// Print Version
static void rocblas_print_version()
{
    static char blas_version[100];
    static int  once = (rocblas_get_version_string(blas_version, sizeof(blas_version)), 0);

#ifdef USE_TENSILE_HOST
    rocblas_cout << "rocBLAS version: " << blas_version << " (new Tensile client)\n" << std::endl;
#else
    rocblas_cout << "rocBLAS version: " << blas_version << "\n" << std::endl;
#endif
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
    // Set signal handler
    rocblas_test_sigaction();

    // Print rocBLAS version
    rocblas_print_version();

    // Set test device
    rocblas_set_test_device();

    // Set data file path
    rocblas_parse_data(argc, argv, rocblas_exepath() + "rocblas_gtest.data");

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // Set Google Test listener
    rocblas_set_listener();

    // Initialize rocBLAS (not explicitly needed; just included for testing)
    rocblas_initialize();

    // Run the tests
    int status = RUN_ALL_TESTS();

    // Failures printed at end for reporting so repeat version info
    rocblas_print_version();

    return status;
}
