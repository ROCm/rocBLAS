/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_OSTREAM_HPP_
#define _ROCBLAS_OSTREAM_HPP_

#include "rocblas.h"
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

/*****************************************************************************
 * rocBLAS output streams                                                    *
 *****************************************************************************/

#define rocblas_cout (rocblas_ostream::cout())
#define rocblas_cerr (rocblas_ostream::cerr())

/***************************************************************************
 * The rocblas_ostream class performs atomic IO on log files, and provides *
 * consistent formatting                                                   *
 ***************************************************************************/
class rocblas_ostream
{
    /**************************************************************************
     * The worker class sets up a worker thread for writing to log files. Two *
     * files are considered the same if they have the same device ID / inode. *
     **************************************************************************/
    class worker
    {
        // Log file
        FILE* file;

        // This worker's thread
        std::thread thread;

        // Condition variable for worker notification
        std::condition_variable cond;

        // Mutex for this thread's queue
        std::mutex mutex;

        // Queue of strings to be logged
        std::queue<std::shared_ptr<std::string>> queue;

        // Worker thread which waits for strings to log
        void thread_function();

    public:
        // Constructor creates a worker thread
        explicit worker(int fd);

        // Enqueue a string to be written
        void enqueue(std::shared_ptr<std::string>);

        // Destroy a worker when all references to it are gone
        ~worker();
    };

    // Initial slice of struct stat which contains device ID and inode
    struct file_id_t
    {
        dev_t st_dev; // ID of device containing file
        ino_t st_ino; // Inode number
    };

    // Compares device IDs and inodes for containers
    struct file_id_less
    {
        bool operator()(const file_id_t& lhs, const file_id_t& rhs) const
        {
            return lhs.st_ino < rhs.st_ino || (lhs.st_ino == rhs.st_ino && lhs.st_dev < rhs.st_dev);
        }
    };

    // Map from file_id to a worker shared_ptr
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& map()
    {
        static std::map<file_id_t, std::shared_ptr<worker>, file_id_less> map;
        return map;
    }

    // Mutex for accessing the map
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& map_mutex()
    {
        static std::recursive_mutex map_mutex;
        return map_mutex;
    }

    // Get worker for file descriptor
    static std::shared_ptr<worker> get_worker(int fd);

    // Output buffer for formatted IO
    std::ostringstream os;

    // Worker thread for accepting logs
    std::shared_ptr<worker> worker_ptr;

    // Flag indicating whether YAML mode is turned on
    bool yaml = false;

public:
    // Default constructor is a std::ostringstream with no worker
    rocblas_ostream() = default;

    // Construct from a file descriptor, which is duped
    explicit rocblas_ostream(int fd);

    // Construct from a C filename
    explicit rocblas_ostream(const char* filename);

    // Construct from a std::string filename
    explicit rocblas_ostream(const std::string& filename)
        : rocblas_ostream(filename.c_str())
    {
    }

    // Convert stream output to string
    std::string str() const
    {
        return os.str();
    }

    // Clear the buffer
    void clear()
    {
        os.clear();
        os.str({});
    }

    // Flush the output atomically
    void flush();

    // Destroy the rocblas_ostream
    virtual ~rocblas_ostream()
    {
        flush(); // Flush any pending IO
    }

    // stdout
    // Implemented as singleton to avoid the static initialization order fiasco
    static rocblas_ostream& cout()
    {
        static rocblas_ostream cout(STDOUT_FILENO);
        return cout;
    }

    // stderr
    // Implemented as singleton to avoid the static initialization order fiasco
    static rocblas_ostream& cerr()
    {
        static rocblas_ostream cerr(STDERR_FILENO);
        return cerr;
    }

    // abort() function which safely flushes all IO
    friend void rocblas_abort();

    /*************************************************************************
     * Non-member friend functions for formatted output                      *
     *************************************************************************/

    // Default output
    template <typename T>
    friend rocblas_ostream& operator<<(rocblas_ostream& os, T&& x)
    {
        os.os << std::forward<T>(x);
        return os;
    }

    // Complex output
    template <typename T>
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const rocblas_complex_num<T>& x)
    {
        if(os.yaml)
            os.os << "'(" << std::real(x) << "," << std::imag(x) << ")'";
        else
            os.os << x;
        return os;
    }

    // Floating-point output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, double x);
    friend rocblas_ostream& operator<<(rocblas_ostream& os, rocblas_half half)
    {
        return os << double(half);
    }
    friend rocblas_ostream& operator<<(rocblas_ostream& os, rocblas_bfloat16 bf16)
    {
        return os << double(bf16);
    }

    // Integer output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, int32_t x)
    {
        os.os << x;
        return os;
    }
    friend rocblas_ostream& operator<<(rocblas_ostream& os, uint32_t x)
    {
        os.os << x;
        return os;
    }
    friend rocblas_ostream& operator<<(rocblas_ostream& os, int64_t x)
    {
        os.os << x;
        return os;
    }
    friend rocblas_ostream& operator<<(rocblas_ostream& os, uint64_t x)
    {
        os.os << x;
        return os;
    }

    // bool output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, bool b);

    // Character output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, char c);

    // String output
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const char* s);
    friend rocblas_ostream& operator<<(rocblas_ostream& os, const std::string& s);

    // Transfer rocblas_ostream to std::ostream
    friend std::ostream& operator<<(std::ostream& os, const rocblas_ostream& str);

    // IO Manipulators
    friend rocblas_ostream& operator<<(rocblas_ostream& os, std::ostream& (*pf)(std::ostream&));

    // YAML Manipulators
    static std::ostream& yaml_on(std::ostream&);
    static std::ostream& yaml_off(std::ostream&);
};

// abort() function which safely flushes all IO
void rocblas_abort();

#endif
