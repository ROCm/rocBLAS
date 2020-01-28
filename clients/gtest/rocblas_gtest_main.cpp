/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_parse_data.hpp"
#include "test_cleanup.hpp"
#include "utility.hpp"
#include <atomic>
#include <cerrno>
#include <csetjmp>
#include <csignal>
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

/*********************************************
 * Signal-handling for detecting test faults *
 *********************************************/
// Id of the thread which is catching signals
static volatile pthread_t rocblas_test_sighandler_tid;

// sigjmp_buf describing stack frame to go back to
static sigjmp_buf rocblas_sigjmp_buf;

// Whether rocblas_sigjmp_buf is set and catching signals is enabled
static volatile sig_atomic_t rocblas_sighandler_enabled = false;

// Signal handler
extern "C" void rocblas_test_signal_handler(int sig)
{
    int e = errno; // Save errno

    // If the signal handler is disabled, restore this signal's disposition
    // to default, and reraise the signal
    if(!rocblas_sighandler_enabled)
    {
        signal(sig, SIG_DFL);
        raise(sig);
        errno = e; // Restore errno
        return;
    }

    // If the thread receiving the signal is different from the thread
    // catching signals, send the signal to the thread catching signals.
    if(!pthread_equal(pthread_self(), rocblas_test_sighandler_tid))
    {
        pthread_kill(rocblas_test_sighandler_tid, sig);
        sleep(1);
        errno = e; // Restore errno
        return;
    }

    // Jump back to the handler code
    // Note: This bypasses stack unwinding, and may lead to memory leaks, but
    // it is better than crashing. Throwing exceptions from signal handlers is
    // poorly supported, and may result in recursive calls to std::terminate.
    errno = e; // Restore errno
    siglongjmp(rocblas_sigjmp_buf, sig);
}

// Set up signal handlers
static void rocblas_test_sigaction()
{
    struct sigaction act;
    sigfillset(&act.sa_mask);
    act.sa_flags   = 0;
    act.sa_handler = rocblas_test_signal_handler;

    for(int sig : {SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGPIPE, SIGSEGV, SIGSYS, SIGUSR1, SIGUSR2})
    {
        sigaction(sig, &act, nullptr);
    }
}

// Lambda wrapper which detects signals and exceptions in an invokable function
void catch_signals_and_exceptions_as_failures(const std::function<void()>& test)
{
    // Save this thread's id, to detect signals in different threads
    rocblas_test_sighandler_tid = pthread_self();

    // Set up the return point
    int sig = sigsetjmp(rocblas_sigjmp_buf, true);

    // If this is a return, indicate the signal received
    if(sig)
    {
        rocblas_sighandler_enabled = false;
        FAIL() << "Received " << strsignal(sig) << " signal";
    }
    else
    {
        // Run the test function, catching signals and exceptions
        // Disable the signal handler after running the test
        rocblas_sighandler_enabled = true;
        try
        {
            test();
            rocblas_sighandler_enabled = false;
        }
        catch(...)
        {
            rocblas_sighandler_enabled = false;
            FAIL() << "Received unhandled exception";
        }
    }
}

/*****************
 * Main function *
 *****************/
int main(int argc, char** argv)
{
    // Set signal handler
    rocblas_test_sigaction();

    // Print Version
    char blas_version[100];
    rocblas_get_version_string(blas_version, sizeof(blas_version));

#ifdef USE_TENSILE_HOST
    printf("rocBLAS version: %s (new Tensile client)\n\n", blas_version);
#else
    printf("rocBLAS version: %s\n\n", blas_version);
#endif

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

    int status = RUN_ALL_TESTS();

    // failures at end copied for reporting so repeat this info
    printf("rocBLAS version: %s\n\n", blas_version);

    return status;
}
