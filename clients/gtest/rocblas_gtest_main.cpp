/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include "utility.hpp"
#include "rocblas_data.hpp"
#include "test_cleanup.hpp"
#include "rocblas_parse_data.hpp"

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{

    protected:
    TestEventListener* eventListener;

    public:
    /**
    * Show the names of each test case.
    */
    bool showTestCases;

    /**
    * Show the names of each test.
    */
    bool showTestNames;

    /**
    * Show each success.
    */
    bool showSuccesses;

    /**
    * Show each failure as it occurs. You will also see it at the bottom after the full suite
    * is run.
    */
    bool showInlineFailures;

    /**
    * Show the setup of the global environment.
    */
    bool showEnvironment;

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
    {
        showTestCases      = true;
        showTestNames      = true;
        showSuccesses      = true;
        showInlineFailures = true;
        showEnvironment    = true;
    }

    virtual ~ConfigurableEventListener() { delete eventListener; }

    virtual void OnTestProgramStart(const UnitTest& unit_test)
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    virtual void OnTestIterationStart(const UnitTest& unit_test, int iteration)
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }
    virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test)
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpStart(unit_test);
        }
    }

    virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test)
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
        }
    }

    virtual void OnTestCaseStart(const TestCase& test_case)
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseStart(test_case);
        }
    }

    virtual void OnTestStart(const TestInfo& test_info)
    {
        if(showTestNames)
        {
            eventListener->OnTestStart(test_info);
        }
    }

    virtual void OnTestPartResult(const TestPartResult& result)
    {
        eventListener->OnTestPartResult(result);
    }

    virtual void OnTestEnd(const TestInfo& test_info)
    {
        if((showInlineFailures && test_info.result()->Failed()) ||
           (showSuccesses && !test_info.result()->Failed()))
        {
            eventListener->OnTestEnd(test_info);
        }
    }

    virtual void OnTestCaseEnd(const TestCase& test_case)
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseEnd(test_case);
        }
    }

    virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test)
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownStart(unit_test);
        }
    }

    virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test)
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
        }
    }

    virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration)
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    virtual void OnTestProgramEnd(const UnitTest& unit_test)
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

    int device_id = 0;

    int device_count = query_device_property();

    if(device_count <= device_id)
    {
        printf("Error: invalid device ID. There may not be such device ID. Will exit \n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    // Set data file path
    static constexpr char GTEST_DATA[] = "rocblas_gtest.data";
    rocblas_parse_data(argc, argv, rocblas_exepath() + GTEST_DATA);

    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // initialize
    ::testing::InitGoogleTest(&argc, argv);

    // remove the default listener
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    auto default_printer                   = listeners.Release(listeners.default_result_printer());

    // add our listener, by default everything is on (the same as using the default listener)
    // here I am turning everything off so I only see the 3 lines for the result
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    ConfigurableEventListener* listener = new ConfigurableEventListener(default_printer);
    listener->showEnvironment           = false;
    listener->showTestCases             = false;
    listener->showTestNames             = false;
    listener->showSuccesses             = false;
    listener->showInlineFailures        = false;
    listeners.Append(listener);

    return RUN_ALL_TESTS();
}
