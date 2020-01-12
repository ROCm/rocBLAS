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
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>

/******************************************************************************
 * The log_worker class sets up a worker thread for writing to log files.     *
 * Two files are considered the same if they have the same device ID / inode. *
 ******************************************************************************/
class log_worker
{
    // Log file
    int filehandle;

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

    // Compares device IDs and inodes for containers
    struct file_id_less
    {
        bool operator()(const struct stat& lhs, const struct stat& rhs) const
        {
            return lhs.st_dev < rhs.st_dev || (lhs.st_dev == rhs.st_dev && lhs.st_ino < rhs.st_ino);
        }
    };

    // Map from file_id to a log_worker shared_ptr
    static std::map<struct stat, std::shared_ptr<log_worker>, file_id_less> map;

    // Mutex for accessing the map
    static std::mutex map_mutex;

public:
    // Enqueue a string to be written and freed by the worker
    void enqueue(std::shared_ptr<std::string> str);

    // Constructor creates a worker thread
    explicit log_worker(int fh)
        : filehandle(dup(fh))
        , worker_thread([=] { worker(); })
    {
    }

    // Get worker for filehandle
    static std::shared_ptr<log_worker> get_worker(int fh);

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
    // Output stream for formatted IO
    std::ostream& os;

    // Worker thread for accepting logs
    std::shared_ptr<log_worker> worker;

public:
    // Construct from std::ostream (does not guarantee atomicity)
    explicit rocblas_ostream(std::ostream& stream)
        : os(stream)
    {
    }

    // Construct from filehandle
    explicit rocblas_ostream(int filehandle)
        : os(*new std::ostringstream)
        , worker(log_worker::get_worker(filehandle))
    {
    }

    // Construct from C filename
    explicit rocblas_ostream(const char* filename)
        : os(*new std::ostringstream)
        , worker(log_worker::get_worker(filename))
    {
    }

    // Construct from std::string filename
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

    friend rocblas_ostream& operator<<(rocblas_ostream& os, rocblas_half x)
    {
        os.os << double(x);
        return os;
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
};

#endif
