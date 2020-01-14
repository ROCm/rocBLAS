#ifndef _ROCBLAS_OSTREAM_HPP_
#define _ROCBLAS_OSTREAM_HPP_

#include "rocblas.h"
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>

/******************************************************************************
 * The log_worker class sets up a worker thread for writing to log files.     *
 * Two files are considered the same if they have the same device ID / inode. *
 ******************************************************************************/
class log_worker
{
    // Log file
    FILE* file;

    // This worker's thread
    std::thread worker_thread;

    // Condition variable for worker notification
    std::condition_variable cond;

    // Mutex for this thread's queue
    std::mutex mutex;

    // Queue of strings to be logged
    std::deque<std::shared_ptr<std::string>> queue;

    // Worker thread which waits for strings to log
    void worker();

public:
    // Enqueue a string to be written
    void enqueue(std::shared_ptr<std::string>);

    // Constructor creates a worker thread
    explicit log_worker(int fd);

    // Get worker for file descriptor
    static std::shared_ptr<log_worker> get_worker(int fd);

    // Get worker for filename
    static std::shared_ptr<log_worker> get_worker(const char* filename);

    // Destroy a worker when all references to it are gone
    ~log_worker();

    // Disallow default construction, copying, assignment
    log_worker()                  = delete;
    log_worker(const log_worker&) = delete;
    log_worker& operator=(const log_worker&) = delete;
};

/***************************************************************************
 * The rocblas_ostream class performs atomic IO on log files, and provides *
 * consistent formatting                                                   *
 ***************************************************************************/
class rocblas_ostream
{
    // Output buffer for formatted IO
    std::ostringstream& os;

    // Worker thread for accepting logs
    std::shared_ptr<log_worker> worker;

    // Construct from various streams. We do not support std::ostream, because
    // we cannot guarantee atomic logging using std::ofstream without using
    // coarse-grained locking. With POSIX file descriptors, we can restrict
    // all writers to the same device/inode to one thread, and do fine-grained
    // locking. With std::ostringstream, we assume strings are thread-local.

protected:
    explicit rocblas_ostream(std::ostringstream& str)
        : os(str)
        , worker(nullptr) // no worker for strings, assumed thread-local
    {
    }

public:
    // Construct from a filehandle
    explicit rocblas_ostream(int fd)
        : os(*new std::ostringstream)
        , worker(log_worker::get_worker(fcntl(fd, F_DUPFD_CLOEXEC)))
    {
    }

    // Construct from a C filename
    explicit rocblas_ostream(const char* filename)
        : os(*new std::ostringstream)
        , worker(log_worker::get_worker(filename))
    {
    }

    // Construct from a std::string filename
    explicit rocblas_ostream(const std::string& filename)
        : rocblas_ostream(filename.c_str())
    {
    }

    // Destroy the rocblas_ostream
    ~rocblas_ostream();

    // Flush the output atomically
    void flush();

    // Default output
    template <typename T>
    friend rocblas_ostream& operator<<(rocblas_ostream& os, T&& x)
    {
        os.os << std::forward<T>(x);
        return os;
    }

    // Floating-point output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, double x);

    friend rocblas_ostream& operator<<(rocblas_ostream& os, rocblas_half half)
    {
        os.os << double(half);
        return os;
    }

    friend rocblas_ostream& operator<<(rocblas_ostream& os, const rocblas_bfloat16& bf16)
    {
        return os << double(bf16);
    }

    // Complex output
    template <typename T>
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const rocblas_complex_num<T>& x)
    {
        os.os << "'(" << std::real(x) << "," << std::imag(x) << ")'";
        return os;
    }

    // Character output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, char c)
    {
        char s[]{c, 0};
        os.os << std::quoted(s, '\'');
        return os;
    }

    // bool output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, bool b)
    {
        os.os << (b ? "true" : "false");
        return os;
    }

    // string output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const char* s)
    {
        os.os << std::quoted(s);
        return os;
    }

    friend rocblas_ostream& operator<<(rocblas_ostream& os, const std::string& s)
    {
        os.os << s.c_str();
        return os;
    }

    // Disallow default and copy construction and assignment
    rocblas_ostream()                       = delete;
    rocblas_ostream(const rocblas_ostream&) = delete;
    rocblas_ostream& operator=(const rocblas_ostream&) = delete;
    rocblas_ostream& operator=(rocblas_ostream&&) = delete;

    // Allow move constructor
    rocblas_ostream(rocblas_ostream&& other)
        : os(other.os)
        , worker(std::move(other.worker))
    {
    }
};

// rocblas_ostringstream acts like std::ostringstream
struct rocblas_ostringstream : rocblas_ostream, std::ostringstream
{
    rocblas_ostringstream()
        : rocblas_ostream(static_cast<std::ostringstream&>(*this))
        , std::ostringstream()
    {
    }

    rocblas_ostringstream(rocblas_ostringstream&& other)
        : rocblas_ostream(std::move(other))
        , std::ostringstream(std::move(other))
    {
    }

    explicit rocblas_ostringstream(const std::string& s)
        : rocblas_ostream(static_cast<std::ostringstream&>(*this))
        , std::ostringstream(s)
    {
    }
};

#endif
