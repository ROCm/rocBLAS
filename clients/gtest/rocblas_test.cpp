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
static struct
{
    // Whether sigjmp_buf is set and catching signals is enabled
    volatile sig_atomic_t enabled;

    // Id of the thread which is catching signals
    volatile pthread_t tid;

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

    // If the thread receiving the signal is different from the thread
    // catching signals, send the signal to the thread catching signals.
    if(!pthread_equal(pthread_self(), handler.tid))
    {
        pthread_kill(handler.tid, sig);
        errno = saved_errno;
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
    // Set up a separate stack in case stack overflows occur
    alignas(64) static char stack_memory[SIGSTKSZ];
    stack_t                 stack = {stack_memory, 0, sizeof(stack_memory)};

    struct sigaction act;
    sigfillset(&act.sa_mask);

    // Signal handler above
    act.sa_handler = rocblas_test_signal_handler;

    // We use the stack if sigaltstack() returns success
    act.sa_flags = !sigaltstack(&stack, nullptr) ? SA_ONSTACK : 0;

    for(int sig : {SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGALRM, SIGSEGV, SIGSYS, SIGUSR1, SIGUSR2})
        sigaction(sig, &act, nullptr);
}

// Number of seconds each test is allowed to take before all testing is killed.
constexpr unsigned TEST_TIMEOUT = 600;

// Lambda wrapper which detects signals and exceptions in an invokable function
void catch_signals_and_exceptions_as_failures(std::function<void()> test, bool set_alarm)
{
    // Save the current handler (to allow nested calls to this function)
    auto old_handler = handler;

    // Set up the return point, and handle siglongjmp returning back to here
    if(sigsetjmp(handler.sigjmp_buf, true))
    {
        FAIL() << "Received " << strsignal(handler.signal) << " signal";
    }
    else
    {
        // Test timeout can be overriden by the ROCBLAS_TEST_TIMEOUT environment variable
        static const unsigned test_timeout = []() {
            unsigned    timeout;
            const char* env = getenv("ROCBLAS_TEST_TIMEOUT");
            return env && sscanf(env, "%u", &timeout) == 1 ? timeout : TEST_TIMEOUT;
        }();

        // Alarm to detect deadlocks or hangs
        if(set_alarm)
            alarm(test_timeout);

        // Save this thread's id, to detect signals in different threads
        handler.tid = pthread_self();

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
