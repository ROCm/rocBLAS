/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_ostream.hpp"
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <type_traits>

/***********************************************************************
 * rocblas_ostream functions                                           *
 ***********************************************************************/

// abort() function which safely flushes all IO
ROCBLAS_EXPORT void rocblas_abort()
{
    // Make sure the alarm action is default
    signal(SIGALRM, SIG_DFL);

    // Timeout
    alarm(2);

    // Obtain the map lock
    rocblas_ostream::map_mutex().lock();

    // Clear the map
    rocblas_ostream::map().clear();

    // TODO: Use synchronization with other threads instead of arbitrary time
    sleep(1);

    // Flush any remaining files
    fflush(NULL);

    // Abort
    abort();
}

// Get worker for file descriptor
std::shared_ptr<rocblas_ostream::worker> rocblas_ostream::get_worker(int fd)
{
    // For a fd indicating an error
    if(fd == -1)
        return nullptr;

    // C++ allows type punning of common initial sequences
    union
    {
        struct stat statbuf;
        file_id_t   file_id;
    };

    // Assert common initial sequence
    static_assert(std::is_standard_layout<file_id_t>{} && std::is_standard_layout<struct stat>{}
                      && offsetof(file_id_t, st_dev) == 0 && offsetof(struct stat, st_dev) == 0
                      && offsetof(file_id_t, st_ino) == offsetof(struct stat, st_ino)
                      && std::is_same<decltype(file_id_t::st_dev), decltype(stat::st_dev)>{}
                      && std::is_same<decltype(file_id_t::st_ino), decltype(stat::st_ino)>{},
                  "struct stat and file_id_t are not layout-compatible");

    // Get the device ID and inode, to detect files already open
    if(fstat(fd, &statbuf))
    {
        perror("Error executing fstat()");
        return {};
    }

    // Lock the map
    std::lock_guard<std::recursive_mutex> lock(map_mutex());

    // Insert a nullptr element if it doesn't already exist
    auto& worker_ptr = map().emplace(file_id, nullptr).first->second;

    // If a new entry was inserted, or an old entry is empty, create worker
    if(!worker_ptr)
        worker_ptr = std::make_shared<worker>(fd);

    // Return the existing or new worker matching the file
    return worker_ptr;
}

// Construct rocblas_ostream from a file descriptor, which is duped
ROCBLAS_EXPORT rocblas_ostream::rocblas_ostream(int fd)
    : worker_ptr(get_worker(fcntl(fd, F_DUPFD_CLOEXEC, 0)))
{
    if(!worker_ptr)
    {
        dprintf(STDERR_FILENO, "Cannot dup file descriptor %d: %m\n", fd);
        rocblas_abort();
    }
}

// Construct from a filename
ROCBLAS_EXPORT rocblas_ostream::rocblas_ostream(const char* filename)
    : worker_ptr(get_worker(open(filename, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644)))
{
    if(!worker_ptr)
    {
        dprintf(STDERR_FILENO, "Cannot open %s: %m\n", filename);
        rocblas_abort();
    }
}

// Flush the output atomically
ROCBLAS_EXPORT void rocblas_ostream::flush()
{
    if(worker_ptr)
    {
        worker_ptr->enqueue(std::make_shared<std::string>(os.str()));
        clear();
    }
}

/***********************************************************************
 * Formatted Output                                                    *
 ***********************************************************************/

// Floating-point output
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, double x)
{
    char        s[32];
    const char* out;

    if(std::isnan(x))
        out = os.yaml ? ".nan" : "nan";
    else if(std::isinf(x))
        out = os.yaml ? (x < 0 ? "-.inf" : ".inf") : (x < 0 ? "-inf" : "inf");
    else
    {
        out = s;
        snprintf(s, sizeof(s) - 2, "%.17g", x);

        // If no decimal point or exponent, append .0
        for(char* end = s; *end != '.' && *end != 'e' && *end != 'E'; ++end)
        {
            if(!*end)
            {
                end[0] = '.';
                end[1] = '0';
                end[2] = '\0';
                break;
            }
        }
    }
    os.os << out;
    return os;
}

// bool output
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, bool b)
{
    if(os.yaml)
        os.os << (b ? "true" : "false");
    else
        os.os << (b ? 1 : 0);
    return os;
}

// Character output
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, char c)
{
    if(os.yaml)
    {
        char s[]{c, 0};
        os.os << std::quoted(s, '\'');
    }
    else
        os.os << c;
    return os;
}

// String output
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, const char* s)
{
    if(os.yaml)
        os.os << std::quoted(s);
    else
        os.os << s;
    return os;
}

ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, const std::string& s)
{
    if(os.yaml)
        os << std::quoted(s.c_str());
    else
        os << s;
    return os;
}

// Transfer rocblas_ostream to std::ostream
ROCBLAS_EXPORT std::ostream& operator<<(std::ostream& os, const rocblas_ostream& str)
{
    return os << str.str();
}

// IO Manipulators
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, std::ostream& (*pf)(std::ostream&))
{
    if(pf == rocblas_ostream::yaml_on)
        os.yaml = true;
    else if(pf == rocblas_ostream::yaml_off)
        os.yaml = false;
    else
    {
        // Output the manipulator to the buffer
        if(pf)
            os.os << pf;

        // If the manipulator is std::endl or std::flush, flush the output
        if(pf == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)
           || pf == static_cast<std::ostream& (*)(std::ostream&)>(std::flush))
        {
            os.flush();
        }
    }
    return os;
}

// YAML Manipulators (only used for their addresses and signatures now)
ROCBLAS_EXPORT std::ostream& rocblas_ostream::yaml_on(std::ostream& os)
{
    return os;
}
ROCBLAS_EXPORT std::ostream& rocblas_ostream::yaml_off(std::ostream& os)
{
    return os;
}

/***********************************************************************
 * rocblas_ostream::worker functions handle logging in a single thread *
 ***********************************************************************/

// enqueue a string to be written and freed by the worker
void rocblas_ostream::worker::enqueue(std::shared_ptr<std::string> ptr)
{
    // Only nullptr or nonzero sized strings are sent
    if(!ptr || ptr->size())
    {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(ptr);
        cond.notify_one();
    }
}

// Worker thread which waits for strings to be logged
void rocblas_ostream::worker::thread_function()
{
    // Clear any errors in the file
    clearerr(file);

    // Lock the mutex in preparation for cond.wait
    std::unique_lock<std::mutex> lock(mutex);

    while(true)
    {
        // Wait for any data
        cond.wait(lock, [&] { return !queue.empty(); });

        // With the mutex locked, pop data from the front of queue
        std::shared_ptr<std::string> log = queue.front();
        queue.pop();

        // A nullptr indicates closing of the file
        if(!log)
            break;

        // Temporarily unlock mutex, unblocking other threads
        lock.unlock();

        // Write the data
        fwrite(log->data(), 1, log->size(), file);

        // Unreference the data
        log.reset();

        // Detect error
        if(ferror(file))
        {
            perror("Error writing log file");
            break;
        }

        // Re-lock the mutex in preparation for cond.wait
        lock.lock();
    }

    // Flush all files
    fflush(NULL);
}

// Constructor creates a worker thread
rocblas_ostream::worker::worker(int fd)
    : file(fdopen(fd, "a"))
    , thread([=] { thread_function(); })
{
    if(!file)
    {
        perror("fdopen() error");
        rocblas_abort();
    }
}

// Destroy a worker when all references to it are gone
rocblas_ostream::worker::~worker()
{
    // Tell the worker thread to exit
    enqueue(nullptr);

    // Wait for the thread to exit
    thread.join();

    // Flush all files
    fflush(NULL);

    // Close the file
    fclose(file);
}
