/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_TEST_H_
#define ROCBLAS_TEST_H_

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#include "../../library/src/include/handle.hpp"
#include "argument_model.hpp"
#include "rocblas.h"
#include "rocblas_arguments.hpp"
#include "test_cleanup.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef GOOGLE_TEST

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define CHECK_DEVICE_ALLOCATION(ERROR)                   \
    do                                                   \
    {                                                    \
        /* Use error__ in case ERROR contains "error" */ \
        hipError_t error__ = (ERROR);                    \
        if(error__ != hipSuccess)                        \
        {                                                \
            if(error__ == hipErrorOutOfMemory)           \
                SUCCEED() << LIMITED_MEMORY_STRING;      \
            else                                         \
                FAIL() << hipGetErrorString(error__);    \
            return;                                      \
        }                                                \
    } while(0)

// This wraps the rocBLAS call with catch_signals_and_exceptions_as_failures().
// By placing it at the rocBLAS call site, memory resources are less likely to
// be leaked in the event of a caught signal.
#define EXPECT_ROCBLAS_STATUS(STATUS, EXPECT)                \
    do                                                       \
    {                                                        \
        volatile bool signal_or_exception = true;            \
        /* Use status__ in case STATUS contains "status" */  \
        rocblas_status status__;                             \
        catch_signals_and_exceptions_as_failures([&] {       \
            status__            = (STATUS);                  \
            signal_or_exception = false;                     \
        });                                                  \
        if(signal_or_exception)                              \
            return;                                          \
        { /* localize status for ASSERT_EQ message */        \
            rocblas_status status = status__;                \
            ASSERT_EQ(status, EXPECT); /* prints "status" */ \
        }                                                    \
    } while(0)

#define CHECK_ALLOC_QUERY(STATUS)                                  \
    do                                                             \
    {                                                              \
        auto status__ = (STATUS);                                  \
        ASSERT_TRUE(status__ == rocblas_status_size_increased      \
                    || status__ == rocblas_status_size_unchanged); \
    } while(0)

#else // GOOGLE_TEST

#define CHECK_ALLOC_QUERY(STATUS)                                                              \
    do                                                                                         \
    {                                                                                          \
        auto status__ = (STATUS);                                                              \
        if(!(status__ == rocblas_status_size_increased                                         \
             || status__ == rocblas_status_size_unchanged))                                    \
        {                                                                                      \
            rocblas_cerr << "rocBLAS status error: Expected rocblas_status_size_unchanged or " \
                            "rocblas_status_size_increase,\nreceived "                         \
                         << rocblas_status_to_string(status__) << std::endl;                   \
            rocblas_abort();                                                                   \
        }                                                                                      \
    } while(0)

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        rocblas_cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                     << ", received " << rocblas_status_to_string(status) << std::endl;
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                                                         \
    do                                                                                 \
    {                                                                                  \
        /* Use error__ in case ERROR contains "error" */                               \
        hipError_t error__ = (ERROR);                                                  \
        if(error__ != hipSuccess)                                                      \
        {                                                                              \
            rocblas_cerr << "error: " << hipGetErrorString(error__) << " (" << error__ \
                         << ") at " __FILE__ ":" << __LINE__ << std::endl;             \
            rocblas_abort();                                                           \
        }                                                                              \
    } while(0)

#define CHECK_DEVICE_ALLOCATION(ERROR)

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#endif // GOOGLE_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

#ifdef GOOGLE_TEST

/* ============================================================================================ */
// Function which matches Arguments with a category, accounting for arg.known_bug_platforms
bool match_test_category(const Arguments& arg, const char* category);

// The tests are instantiated by filtering through the RocBLAS_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, category)                                            \
    INSTANTIATE_TEST_SUITE_P(category,                                                            \
                             testclass,                                                           \
                             testing::ValuesIn(RocBLAS_TestData::begin([](const Arguments& arg) { \
                                                   return testclass::type_filter(arg)             \
                                                          && testclass::function_filter(arg)      \
                                                          && match_test_category(arg, #category); \
                                               }),                                                \
                                               RocBLAS_TestData::end()),                          \
                             testclass::PrintToStringParamName());

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)        \
    INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
    INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
    INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
    INSTANTIATE_TEST_CATEGORY(testclass, multi_gpu)   \
    INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

// Function to catch signals and exceptions as failures
void catch_signals_and_exceptions_as_failures(std::function<void()> test, bool set_alarm = false);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(test) \
    catch_signals_and_exceptions_as_failures([&] { test; }, true)

// Function to catch signals and exceptions as failures
void launch_test_on_threads(std::function<void()> test,
                            size_t                numThreads,
                            size_t                numStreams,
                            size_t                numDevices);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define LAUNCH_TEST_ON_THREADS(test, threads, streams, devices) \
    launch_test_on_threads([&] { test; }, threads, streams, devices)

// Function to catch signals and exceptions as failures
void launch_test_on_streams(std::function<void()> test, size_t numStreams, size_t numDevices);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define LAUNCH_TEST_ON_STREAMS(test, streams, devices) \
    launch_test_on_streams([&] { test; }, streams, devices)

// Macro to run test across threads
#define RUN_TEST_ON_THREADS_STREAMS(test)                            \
    do                                                               \
    {                                                                \
        const auto& arg          = GetParam();                       \
        auto        threads      = arg.threads;                      \
        auto        streams      = arg.streams;                      \
        auto        devices      = arg.devices;                      \
        int         availDevices = 0;                                \
        hipGetDeviceCount(&availDevices);                            \
        if(devices > availDevices)                                   \
            FAIL() << "Too many devices requested";                  \
        g_stream_pool.reset(devices, streams);                       \
        if(threads)                                                  \
            LAUNCH_TEST_ON_THREADS(test, threads, streams, devices); \
        else                                                         \
            LAUNCH_TEST_ON_STREAMS(test, streams, devices);          \
    } while(0)

// Thread worker class
class thread_pool
{
    std::atomic_bool                                                 m_done{false};
    std::queue<std::pair<std::function<void()>, std::promise<void>>> m_work_queue;
    std::vector<std::thread>                                         m_threads;
    std::mutex                                                       m_mutex;
    std::condition_variable                                          m_cond;

    void worker_thread()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while(!m_done)
        {
            m_cond.wait(lock, [&] { return m_done || !m_work_queue.empty(); });
            if(m_done)
                break;
            auto task    = std::move(m_work_queue.front().first);
            auto promise = std::move(m_work_queue.front().second);
            m_work_queue.pop();
            lock.unlock();
            task();
            promise.set_value();
            lock.lock();
        }
    }

public:
    thread_pool()
    {
        auto thread_count = std::thread::hardware_concurrency();
        do
            m_threads.push_back(std::thread([&] { worker_thread(); }));
        while(thread_count-- > 1);
    }

    ~thread_pool()
    {
        m_done = true;
        m_cond.notify_all();
        for(auto& thread : m_threads)
            thread.join();
    }

    template <typename FUNC>
    void submit(FUNC&& func, std::promise<void> promise)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_work_queue.push(std::make_pair(std::forward<FUNC>(func), std::move(promise)));
        }
        m_cond.notify_one();
    }
};

class stream_pool
{
    std::vector<std::vector<hipStream_t>> m_streams;

public:
    stream_pool() = default;

    void reset(size_t numDevices = 0, size_t numStreams = 0)
    {
        for(size_t i = 0; i < m_streams.size(); ++i)
            for(size_t j = 0; j < m_streams[i].size(); ++j)
                hipStreamDestroy(m_streams[i][j]);

        m_streams.clear();

        for(size_t i = 0; i < (numDevices > 1 ? numDevices : 1); ++i)
        {
            if(numDevices)
                hipSetDevice(i);
            std::vector<hipStream_t> temp;
            for(size_t j = 0; j < numStreams; ++j)
            {
                hipStream_t stream;
                if(hipStreamCreate(&stream) == hipSuccess)
                    temp.push_back(stream);
            }
            m_streams.push_back(temp);
        }
    }

    ~stream_pool()
    {
        reset();
    }

    hipStream_t* get_stream_pointer(size_t deviceId)
    {
        return m_streams[deviceId].data();
    }
};

extern thread_local std::shared_ptr<std::function<void(rocblas_handle&)>>
    rocblas_set_stream_callback;

inline void rocblas_test_set_stream(rocblas_handle& handle)
{
    if(rocblas_set_stream_callback)
    {
        (*rocblas_set_stream_callback)(handle);
        rocblas_set_stream_callback.reset();
    }
}

extern stream_pool g_stream_pool;
extern thread_pool g_thread_pool;

/* ============================================================================================ */
/*! \brief  Normalized test name to conform to Google Tests */
// Template parameter is used to generate multiple instantiations
template <typename>
class RocBLAS_TestName
{
    std::ostringstream str;

    static auto& get_table()
    {
        // Placed inside function to avoid dependency on initialization order
        static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
        return *table;
    }

public:
    explicit RocBLAS_TestName(const char* name)
    {
        str << name << '_';
    }

    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This table is private to each instantation of RocBLAS_TestName
        auto&       table = get_table();
        std::string name(str.str());

        // Remove trailing underscore
        if(!name.empty() && name.back() == '_')
            name.pop_back();

        // If name is empty, make it 1
        if(name.empty())
            name = "1";

        // Warn about unset letter parameters
        if(name.find('*') != name.npos)
            rocblas_cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            "Warning: Character * found in name."
                            " This means a required letter parameter\n"
                            "(e.g., transA, diag, etc.) has not been set in the YAML file."
                            " Check the YAML file.\n"
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                         << std::endl;

        // Replace non-alphanumeric characters with letters
        std::replace(name.begin(), name.end(), '-', 'n'); // minus
        std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

        // Complex (A,B) is replaced with ArBi
        name.erase(std::remove(name.begin(), name.end(), '('), name.end());
        std::replace(name.begin(), name.end(), ',', 'r');
        std::replace(name.begin(), name.end(), ')', 'i');

        // If parameters are repeated, append an incrementing suffix
        auto p = table.find(name);
        if(p != table.end())
            name += "_t" + std::to_string(++p->second);
        else
            table[name] = 1;

        return name;
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend RocBLAS_TestName& operator<<(RocBLAS_TestName& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend RocBLAS_TestName&& operator<<(RocBLAS_TestName&& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return std::move(name);
    }

    RocBLAS_TestName()                        = default;
    RocBLAS_TestName(const RocBLAS_TestName&) = delete;
    RocBLAS_TestName& operator=(const RocBLAS_TestName&) = delete;
};

// ----------------------------------------------------------------------------
// RocBLAS_Test base class. All non-legacy rocBLAS Google tests derive from it.
// It defines a type_filter_functor() and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class RocBLAS_Test : public testing::TestWithParam<Arguments>
{
protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments&)
        {
            return static_cast<bool>(FILTER<T...>{});
        }
    };

public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            return TEST::name_suffix(info.param);
        }
    };
};

// Function to set up signal handlers
void rocblas_test_sigaction();

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Normal tests which return true when converted to bool
// ----------------------------------------------------------------------------
struct rocblas_test_valid
{
    // Return true to indicate the type combination is valid, for filtering
    virtual explicit operator bool() final
    {
        return true;
    }

    // Require derived class to define functor which takes (const Arguments &)
    virtual void operator()(const Arguments&) = 0;

    virtual ~rocblas_test_valid() = default;
};

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct rocblas_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    virtual explicit operator bool() final
    {
        return false;
    }

    // If this specialization is actually called, print fatal error message
    virtual void operator()(const Arguments&) final
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        rocblas_cerr << msg << std::endl;
        rocblas_abort();
#endif
    }

    virtual ~rocblas_test_invalid() = default;
};

#endif
