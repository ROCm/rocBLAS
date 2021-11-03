/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// Predeclare rocblas_abort_once() for friend declaration in rocblas_ostream.hpp
static void rocblas_abort_once [[noreturn]] ();

#include "rocblas_ostream.hpp"
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <type_traits>
#ifdef WIN32
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <windows.h>

#define FDOPEN(A, B) _fdopen(A, B)
#define OPEN(A) _open(A, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_APPEND, _S_IREAD | _S_IWRITE);
#define CLOSE(A) _close(A)
#else
#define FDOPEN(A, B) fdopen(A, B)
#define OPEN(A) open(A, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
#define CLOSE(A) close(A)
#endif

/***********************************************************************
 * rocblas_internal_ostream functions                                           *
 ***********************************************************************/

// Abort function which is called only once by rocblas_abort
static void rocblas_abort_once()
{
#ifndef WIN32
    // Make sure the alarm and abort actions are default
    signal(SIGALRM, SIG_DFL);
    signal(SIGABRT, SIG_DFL);

    // Unblock the alarm and abort signals
    sigset_t set[1];
    sigemptyset(set);
    sigaddset(set, SIGALRM);
    sigaddset(set, SIGABRT);
    sigprocmask(SIG_UNBLOCK, set, nullptr);

    // Timeout in case of deadlock
    alarm(5);
#endif

    // Clear the map, stopping all workers
    rocblas_internal_ostream::clear_workers();

    // Flush all
    fflush(NULL);

    // Abort
    std::abort();
}

// Abort function which safely flushes all IO
extern "C" void rocblas_abort()
{
    // If multiple threads call rocblas_abort(), the first one wins
    static int once = (rocblas_abort_once(), 0);
}

// Get worker for writing to a file descriptor
std::shared_ptr<rocblas_internal_ostream::worker> rocblas_internal_ostream::get_worker(int fd)
{
    // For a file descriptor indicating an error, return a nullptr
    if(fd == -1)
        return nullptr;

    // C++ allows type punning of common initial sequences
    union
    {
        struct stat statbuf;
        file_id_t   file_id;
    };

    // Verify common initial sequence
    static_assert(std::is_standard_layout<file_id_t>{} && std::is_standard_layout<struct stat>{}
                      && offsetof(file_id_t, st_dev) == 0 && offsetof(struct stat, st_dev) == 0
                      && offsetof(file_id_t, st_ino) == offsetof(struct stat, st_ino)
                      && std::is_same<decltype(file_id_t::st_dev), decltype(stat::st_dev)>{}
                      && std::is_same<decltype(file_id_t::st_ino), decltype(stat::st_ino)>{},
                  "struct stat and file_id_t are not layout-compatible");

#ifndef WIN32
    // Get the device ID and inode, to detect common files
    if(fstat(fd, &statbuf))
    {
        perror("Error executing fstat()");
        return nullptr;
    }
#else
    HANDLE                     fh = (HANDLE)_get_osfhandle(fd);
    BY_HANDLE_FILE_INFORMATION bhfi;

    if(GetFileInformationByHandle(fh, &bhfi))
    {
        // Index info should be unique
        file_id.st_dev = bhfi.nFileIndexLow;
        file_id.st_ino = bhfi.nFileIndexHigh;
    }
    else
    {
        // assign what should be unique
        file_id.st_dev = fd;
        file_id.st_ino = 0;
    }
#endif

    // Lock the map from file_id -> std::shared_ptr<rocblas_internal_ostream::worker>
    std::lock_guard<std::recursive_mutex> lock(worker_map_mutex());

    // Insert a nullptr map element if file_id doesn't exist in map already
    // worker_ptr is a reference to the std::shared_ptr<rocblas_internal_ostream::worker>
    auto& worker_ptr = worker_map().emplace(file_id, nullptr).first->second;

    // If a new entry was inserted, or an old entry is empty, create new worker
    if(!worker_ptr)
        worker_ptr = std::make_shared<worker>(fd);

    // Return the existing or new worker matching the file
    return worker_ptr;
}

// Construct rocblas_internal_ostream from a file descriptor
rocblas_internal_ostream::rocblas_internal_ostream(int fd)
    : worker_ptr(get_worker(fd))
{
    if(!worker_ptr)
    {
        std::cerr << "Error: Bad file descriptor " << fd << std::endl;
        rocblas_abort();
    }
}

// Construct rocblas_internal_ostream from a filename opened for writing with truncation
rocblas_internal_ostream::rocblas_internal_ostream(const char* filename)
{
    int fd     = OPEN(filename);
    worker_ptr = get_worker(fd);
    if(!worker_ptr)
    {
        std::cerr << "Cannot open " << filename << std::endl;
        rocblas_abort();
    }
    CLOSE(fd);
}

rocblas_internal_ostream::~rocblas_internal_ostream()
{
    flush(); // Flush any pending IO
}

// Flush the output
void rocblas_internal_ostream::flush()
{
    // Flush only if this stream contains a worker (i.e., is not a string)
    if(worker_ptr)
    {
        // The contents of the string buffer
        auto str = os.str();

        // Empty string buffers kill the worker thread, so they are not flushed here
        if(str.size())
            worker_ptr->send(std::move(str));

        // Clear the string buffer
        clear();
    }
}

void rocblas_internal_ostream::clear_workers()
{
    std::lock_guard<std::recursive_mutex> lock(worker_map_mutex());
    worker_map().clear();
}

// YAML Manipulators (only used for their addresses now)
std::ostream& rocblas_internal_ostream::yaml_on(std::ostream& os)
{
    return os;
}

std::ostream& rocblas_internal_ostream::yaml_off(std::ostream& os)
{
    return os;
}

/***********************************************************************
 * rocblas_internal_ostream::worker functions handle logging in a single thread *
 ***********************************************************************/

// Send a string to the worker thread for this stream's device/inode
// Empty strings tell the worker thread to exit
void rocblas_internal_ostream::worker::send(std::string str)
{
    // Create a promise to wait for the operation to complete
    std::promise<void> promise;

    // The future indicating when the operation has completed
    auto future = promise.get_future();

#ifdef WIN32
    // Passing an empty string will make the worker thread exit.
    // The below flag will be used to handle worker thread exit condition for Windows
    bool empty_string = str.empty();
#endif

    // task_t consists of string and promise
    // std::move transfers ownership of str and promise to task
    task_t worker_task(std::move(str), std::move(promise));

    // Submit the task to the worker assigned to this device/inode
    // Hold mutex for as short as possible, to reduce contention
    {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(worker_task));

        // no lock needed for notification but keeping here
        cond.notify_one();
    }

// Wait for the task to be completed, to ensure flushed IO
#ifdef WIN32
    if(empty_string)
        // Occassionaly this thread is not getting the promise set by the 'worker' thread during exit condition.
        // Added a timed wait to exit after one second, if we do not get the promise from worker thread.
        future.wait_for(std::chrono::seconds(1));
    else
        future.get();
#else
    future.get();
#endif
}

// Worker thread which serializes data to be written to a device/inode
void rocblas_internal_ostream::worker::thread_function()
{
    // Clear any errors in the FILE
    clearerr(file);

    // Lock the mutex in preparation for cond.wait
    std::unique_lock<std::mutex> lock(mutex);

    while(true)
    {
        // Wait for any data, ignoring spurious wakeups, locks lock on continue
        cond.wait(lock, [&] { return !queue.empty(); });

        // With the mutex locked, get and pop data from the front of queue
        task_t task = std::move(queue.front());
        queue.pop();

        // Temporarily unlock queue mutex, unblocking other threads
        lock.unlock();

        // An empty message indicates the closing of the stream
        if(!task.size())
        {
            // Tell future to wake up
            task.set_value();
            break;
        }

        // Write the data
        fwrite(task.data(), 1, task.size(), file);

        // Detect any error and flush the C FILE stream
        if(ferror(file) || fflush(file))
        {
            perror("Error writing log file");

            // Tell future to wake up
            task.set_value();
            break;
        }

        // Promise that the data has been written
        task.set_value();

        // Re-lock the mutex in preparation for cond.wait
        lock.lock();
    }
}

// Constructor creates a worker thread from a file descriptor
rocblas_internal_ostream::worker::worker(int fd)
{
    // The worker duplicates the file descriptor (RAII)
#ifdef WIN32
    fd = _dup(fd);
#else
    fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
#endif

    // If the dup fails or fdopen fails, print error and abort
    if(fd == -1 || !(file = FDOPEN(fd, "a")))
    {
        perror("fdopen() error");
        rocblas_abort();
    }

    // Create a worker thread, capturing *this
    thread = std::thread([=] { thread_function(); });

    // Detatch from the worker thread
    thread.detach();
}

rocblas_internal_ostream::worker::~worker()
{
    // Tell worker thread to exit, by sending it an empty string
    send({});

    // Close the FILE
    if(file)
        fclose(file);
}
