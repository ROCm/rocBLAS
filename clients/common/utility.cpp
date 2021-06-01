/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifdef WIN32
#include <windows.h>
#endif
#include "../../library/src/include/handle.hpp"
#include "rocblas_random.hpp"
#include "utility.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#ifdef WIN32
#define strcasecmp(A, B) _stricmp(A, B)
#endif

//
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
//
#ifdef __cpp_lib_filesystem
#include <filesystem>
#else
#include <experimental/filesystem>

namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
const rocblas_rng_t rocblas_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
const std::thread::id main_thread_id = std::this_thread::get_id();

// For the main thread, we use rocblas_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
thread_local rocblas_rng_t t_rocblas_rng = get_seed();

/* ============================================================================================ */
// Return path of this executable
std::string rocblas_exepath()
{
#ifdef WIN32
    //wchar_t wpath[MAX_PATH + 1] = {0};
    //GetModuleFileNameW(NULL, wpath, MAX_PATH + 1);

    std::vector<wchar_t> result(MAX_PATH + 1);
    // Ensure result is large enough to accomodate the path
    for(;;)
    {
        auto length = GetModuleFileNameW(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length);
            result.shrink_to_fit();
            break;
        }
        result.resize(result.size() * 2);
    }

    std::wstring          wspath(result.data());
    std::filesystem::path exepath(wspath.begin(), wspath.end());
    exepath = exepath.remove_filename();
    // Add trailing "/" to exepath if required
    exepath += exepath.empty() ? "" : "/";
    return exepath.string();
#else
    std::string pathstr;
    char*       path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
#endif
}

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device(void)
{
    hipDeviceSynchronize();

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): no GPU synchronization */
double get_time_us_no_sync(void)
{
    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
rocblas_int query_device_property()
{
    int            device_count;
    rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
    if(status != rocblas_status_success)
    {
        rocblas_cerr << "Query device error: cannot get device count" << std::endl;
        return -1;
    }
    else
    {
        rocblas_cout << "Query device success: there are " << device_count << " devices"
                     << std::endl;
    }

    for(rocblas_int i = 0;; i++)
    {
        rocblas_cout
            << "-------------------------------------------------------------------------------"
            << std::endl;

        if(i >= device_count)
            break;

        hipDeviceProp_t props;
        rocblas_status  status = (rocblas_status)hipGetDeviceProperties(&props, i);
        if(status != rocblas_status_success)
        {
            rocblas_cerr << "Query device error: cannot get device ID " << i << "'s property"
                         << std::endl;
        }
        else
        {
            char buf[320];
            snprintf(
                buf,
                sizeof(buf),
                "Device ID %d : %s %s\n"
                "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, compute capability "
                "%d.%d\n"
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                i,
                props.name,
                props.gcnArchName,
                props.totalGlobalMem / 1e9,
                (int)(props.clockRate / 1000),
                (int)(props.memoryClockRate / 1000),
                props.major,
                props.minor,
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);
            rocblas_cout << buf;
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    rocblas_status status = (rocblas_status)hipSetDevice(device_id);
    if(status != rocblas_status_success)
    {
        rocblas_cerr << "Set device error: cannot set device ID " << device_id
                     << ", there may not be such device ID" << std::endl;
    }
}

/*****************
 * local handles *
 *****************/

rocblas_local_handle::rocblas_local_handle()
{
    auto status = rocblas_create_handle(&m_handle);
    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));

#ifdef GOOGLE_TEST
    if(t_set_stream_callback)
    {
        (*t_set_stream_callback)(m_handle);
        t_set_stream_callback.reset();
    }
#endif
}

rocblas_local_handle::rocblas_local_handle(const Arguments& arg)
    : rocblas_local_handle()
{
    // Set the atomics mode
    auto status = rocblas_set_atomics_mode(m_handle, arg.atomics_mode);

    if(status == rocblas_status_success)
    {
        // If the test specifies user allocated workspace, allocate and use it
        if(arg.user_allocated_workspace)
        {
            if((hipMalloc)(&m_memory, arg.user_allocated_workspace) != hipSuccess)
                throw std::bad_alloc();
            status = rocblas_set_workspace(m_handle, m_memory, arg.user_allocated_workspace);
        }
    }

    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));
}

rocblas_local_handle::~rocblas_local_handle()
{
    if(m_memory)
        (hipFree)(m_memory);
    rocblas_destroy_handle(m_handle);
}
