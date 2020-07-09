/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "utility.hpp"
#include "../../library/src/include/handle.h"
#include "rocblas_random.hpp"
#include <cstdlib>
#include <cstring>
#include <regex>
#include <sys/time.h>

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
const rocblas_rng_t rocblas_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
const std::thread::id main_thread_id = std::this_thread::get_id();

// For the main thread, we use rocblas_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
thread_local rocblas_rng_t rocblas_rng = get_seed();

/* ============================================================================================ */
// Return path of this executable
std::string rocblas_exepath()
{
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
}

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device(void)
{
    hipDeviceSynchronize();
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
};

/*! \brief  CPU Timer(in microsecond): no GPU synchronization */
double get_time_us_no_sync(void)
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
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
                "Device ID %d : %s\n"
                "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, compute capability "
                "%d.%d\n"
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                i,
                props.name,
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

/********************************************************************************************
 * Function which matches Arguments with a category, accounting for arg.known_bug_platforms *
 ********************************************************************************************/
bool match_test_category(const Arguments& arg, const char* category)
{
    if(*arg.known_bug_platforms)
    {
        // Regular expression for token delimiters
        static const std::regex regex{"[:, \\f\\n\\r\\t\\v]+", std::regex_constants::optimize};

        // The name of the current GPU platform
        static const std::string platform = rocblas_get_arch_name();

        // Token iterator
        std::cregex_token_iterator iter{arg.known_bug_platforms,
                                        arg.known_bug_platforms + strlen(arg.known_bug_platforms),
                                        regex,
                                        -1};

        // Iterate across tokens in known_bug_platforms, looking for matches with platform
        for(; iter != std::cregex_token_iterator(); ++iter)
        {
            // If a platform matches, set category to "known_bug"
            if(!strcasecmp(iter->str().c_str(), platform.c_str()))
            {
                // We know that underlying arg object is non-const, so we can use const_cast
                strcpy(const_cast<char*>(arg.category), "known_bug");
                break;
            }
        }
    }

    // Return whether arg.category matches the requested category
    return !strcmp(arg.category, category);
}
