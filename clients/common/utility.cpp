/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifdef WIN32
#include <windows.h>
//
#include <random>
#endif
#include "../../library/src/include/handle.hpp"
#include "d_vector.hpp"
#include "utility.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <stdlib.h>

#ifdef WIN32
#define strcasecmp(A, B) _stricmp(A, B)

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// Not WIN32
#else
#include <fcntl.h>
#endif

/* ============================================================================================ */
// Return path of this executable
std::string rocblas_exepath()
{
#ifdef WIN32
    // for now not building with wide chars
    // wchar_t wpath[MAX_PATH + 1] = {0};
    // GetModuleFileNameW(NULL, wpath, MAX_PATH + 1);
    // std::vector<wchar_t> result(MAX_PATH + 1);

    std::vector<TCHAR> result(MAX_PATH + 1);
    // Ensure result is large enough to accomodate the path
    DWORD length = 0;
    for(;;)
    {
        length = GetModuleFileNameA(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length + 1);
            break;
        }
        result.resize(result.size() * 2);
    }

    // std::wstring          wspath(result.data());
    // fs::path exepath(wspath.begin(), wspath.end());

    fs::path exepath(result.begin(), result.end());
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
// Temp directory rooted random path
std::string rocblas_tempname()
{
#ifdef WIN32
    // Generate "/tmp/rocblas-XXXXXX" like file name
    const std::string alphanum     = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv";
    int               stringlength = alphanum.length() - 1;
    std::string       uniquestr    = "rocblas-";

    for(auto n : {0, 1, 2, 3, 4, 5})
        uniquestr += alphanum.at(rand() % stringlength);

    fs::path tmpname = fs::temp_directory_path() / uniquestr;

    return tmpname.string();
#else
    char tmp[] = "/tmp/rocblas-XXXXXX";
    int  fd    = mkostemp(tmp, O_CLOEXEC);
    if(fd == -1)
    {
        dprintf(STDERR_FILENO, "Cannot open temporary file: %m\n");
        exit(EXIT_FAILURE);
    }

    return std::string(tmp);
#endif
}

/* ============================================================================================ */
/*  memory allocation requirements :*/

/*! \brief Compute strided batched matrix allocation size allowing for strides smaller than full matrix */
size_t
    strided_batched_matrix_size(int rows, int cols, int lda, rocblas_stride stride, int batch_count)
{
    size_t size = size_t(lda) * cols;
    if(batch_count > 1)
    {
        // for cases where batch_count strides may not exceed full matrix size use full matrix size
        // e.g. row walking a larger matrix we just use full matrix size
        size_t size_strides = (batch_count - 1) * stride;
        size += size < size_strides + (cols - 1) * size_t(lda) + rows ? size_strides : 0;
    }
    return size;
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

    // memory guard control, with multi-threading should not change values across threads
    d_vector_set_pad_length(arg.pad);

    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));
}

rocblas_local_handle::~rocblas_local_handle()
{
    if(m_memory)
        (hipFree)(m_memory);
    rocblas_destroy_handle(m_handle);
}

/*!
 * Initialize rocBLAS for the current HIP device and report
 * the time taken to complete the initialization. This is to
 * avoid costly startup time at the first call on that device.
 * Internal use for benchmark & testing.
 */
void rocblas_client_initialize()
{
    // when executed on a CPU under normal load( Disk I/O, memory etc.),
    // this routine completes execution under max limit of 12 seconds.
    // The minimum time it takes to complete varies based on
    // the architecture & build options used while building the library.
    // Setting a max duration of 5 seconds for rocblas library initialization to complete.
    constexpr static int max_duration = 5;

    // Store the start timepoint of rocblas initialize
    auto start_time = std::chrono::steady_clock::now();

    rocblas_initialize();

    // Store the end timepoint of rocblas initialize
    auto end_time = std::chrono::steady_clock::now();

    // Compute the time taken to load the Tensile kernels (in seconds).
    auto total_library_initialize_time
        = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    // Compute the time taken to load the Tensile kernels (in milliseconds).
    auto init_time_in_ms
        = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    rocblas_cout << "\nrocBLAS info: Time taken to complete rocBLAS library initialization is "
                 << init_time_in_ms << " milliseconds." << std::endl;

    // If initialization time exceeds the max duration, display the following info message.
    if(total_library_initialize_time > max_duration)
        rocblas_cerr << "\nrocBLAS info: rocBLAS initialization exceeded the max duration of "
                     << max_duration << " seconds. Check CPU's load metrics." << std::endl;
}
