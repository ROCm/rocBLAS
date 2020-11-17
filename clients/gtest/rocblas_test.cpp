#include "rocblas_test.hpp"
#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <pthread.h>
#include <unistd.h>

/*********************************************
 * Signal-handling for detecting test faults *
 *********************************************/

// State of the signal handler
static thread_local struct
{
    // Whether sigjmp_buf is set and catching signals is enabled
    volatile sig_atomic_t enabled = false;

    // sigjmp_buf describing stack frame to go back to
    sigjmp_buf sigjmp_buf;

    // The signal which was received
    volatile sig_atomic_t signal;
} handler;

// Signal handler (must have external "C" linkage)
extern "C" void rocblas_test_signal_handler(int sig)
{
    int saved_errno = errno; // Save errno

    // If the signal handler is disabled, because we're not in the middle of
    // running a rocBLAS test, restore this signal's disposition to default,
    // and reraise the signal
    if(!handler.enabled)
    {
        signal(sig, SIG_DFL);
        errno = saved_errno;
        raise(sig);
        return;
    }

    // If this is an alarm timeout, we abort
    if(sig == SIGALRM)
    {
        static constexpr char msg[]
            = "\nAborting tests due to an alarm timeout.\n\n"
              "This could be due to a deadlock caused by mutexes being left locked\n"
              "after a previous test's signal was caught and partially recovered from.\n";
        // We must use write() because it's async-signal-safe and other IO might be blocked
        write(STDERR_FILENO, msg, sizeof(msg) - 1);
        rocblas_abort();
    }

    // Jump back to the handler code after setting handler.signal
    // Note: This bypasses stack unwinding, and may lead to memory leaks, but
    // it is better than crashing.
    handler.signal = sig;
    errno          = saved_errno;
    siglongjmp(handler.sigjmp_buf, true);
}

// Set up signal handlers
void rocblas_test_sigaction()
{
    struct sigaction act;
    act.sa_flags = 0;
    sigfillset(&act.sa_mask);
    act.sa_handler = rocblas_test_signal_handler;

    // Catch SIGALRM and synchronous signals
    for(int sig : {SIGALRM, SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV})
        sigaction(sig, &act, nullptr);
}

static const unsigned test_timeout = [] {
    // Number of seconds each test is allowed to take before all testing is killed.
    constexpr unsigned TEST_TIMEOUT = 600;
    unsigned           timeout;
    const char*        env = getenv("ROCBLAS_TEST_TIMEOUT");
    return env && sscanf(env, "%u", &timeout) == 1 ? timeout : TEST_TIMEOUT;
}();

// Lambda wrapper which detects signals and exceptions in an invokable function
void catch_signals_and_exceptions_as_failures(std::function<void()> test, bool set_alarm)
{
    // Save the current handler (to allow nested calls to this function)
    auto old_handler = handler;

    // Set up the return point, and handle siglongjmp returning back to here
    if(sigsetjmp(handler.sigjmp_buf, true))
    {
        FAIL() << "Received " << sys_siglist[handler.signal] << " signal";
    }
    else
    {
        // Alarm to detect deadlocks or hangs
        if(set_alarm)
            alarm(test_timeout);

        // Enable the signal handler
        handler.enabled = true;

        // Run the test function, catching signals and exceptions
        try
        {
            test();
        }
        catch(const std::exception& e)
        {
            FAIL() << "Received uncaught exception: " << e.what();
        }
        catch(...)
        {
            FAIL() << "Received uncaught exception";
        }
    }

    // Cancel the alarm if it was set
    if(set_alarm)
        alarm(0);

    // Restore the previous handler
    handler = old_handler;
}

void launch_test_on_streams(std::function<void()> test, size_t numStreams, size_t numDevices)
{
    size_t devices = numDevices > 1 ? numDevices : 1;
    size_t streams = numStreams > 1 ? numStreams : 1;
    for(size_t i = 0; i < devices; ++i)
    {
        if(numDevices)
            hipSetDevice(i);
        for(size_t j = 0; j < streams; ++j)
        {
            if(numStreams)
                rocblas_set_stream_callback.reset(
                    new std::function<void(rocblas_handle&)>([=](rocblas_handle& handle) {
                        rocblas_set_stream(handle, g_stream_pool.get_stream_pointer(i)[j]);
                    }));
            catch_signals_and_exceptions_as_failures(test, true);
        }
    }
}

void launch_test_on_threads(std::function<void()> test,
                            size_t                numThreads,
                            size_t                numStreams,
                            size_t                numDevices)
{
    auto promise = std::make_unique<std::promise<void>[]>(numThreads);
    auto future  = std::make_unique<std::future<void>[]>(numThreads);

    for(size_t i = 0; i < numThreads; ++i)
        future[i] = promise[i].get_future();

    for(size_t i = 0; i < numThreads; ++i)
        g_thread_pool.submit([&] { launch_test_on_streams(test, numStreams, numDevices); },
                             std::move(promise[i]));
    for(size_t i = 0; i < numThreads; ++i)
        future[i].get(); //Wait for tasks to complete
}
