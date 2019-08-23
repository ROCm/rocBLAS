/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_parse_data.hpp"
#include "test_cleanup.hpp"
#include "utility.hpp"
#include <cstdlib>
#include <gtest/gtest.h>

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

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
        , showTestCases(true)
        , showTestNames(true)
        , showSuccesses(true)
        , showInlineFailures(true)
        , showEnvironment(true)
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
        eventListener->OnTestProgramEnd(unit_test);
    }
};

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Print Version
    char blas_version[100];
    rocblas_get_version_string(blas_version, sizeof(blas_version));
    printf("rocBLAS version: %s\n\n", blas_version);

    // Device Query
    int device_id    = 0;
    int device_count = query_device_property();
    if(device_count <= device_id)
    {
        std::cerr << "Error: invalid device ID. There may not be such device ID.\n";
        return -1;
    }
    set_device(device_id);

    // Set data file path
    rocblas_parse_data(argc, argv, rocblas_exepath() + "rocblas_gtest.data");

    // initialize
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

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
    auto listener       = new ConfigurableEventListener(default_printer);
    auto gtest_listener = getenv("GTEST_LISTENER");
    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
        listener->showTestNames = listener->showSuccesses = listener->showInlineFailures = false;
    listeners.Append(listener);

    return RUN_ALL_TESTS();
}
