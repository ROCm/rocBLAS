#include "rocblas_ostream.hpp"
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <map>
#include <sys/stat.h>
#include <type_traits>

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

        // Flush the data and detect error
        if(ferror(file) || fflush(file))
        {
            perror("Error writing log file");
            break;
        }

        // Re-lock the mutex in preparation for cond.wait
        lock.lock();
    }
}

// Constructor creates a worker thread
rocblas_ostream::worker::worker(int fd)
    : file(fdopen(fd, "a"))
    , thread([=] { thread_function(); })
{
    if(!file)
    {
        perror("fdopen() error");
        abort();
    }
}

// Destroy a worker when all references to it are gone
rocblas_ostream::worker::~worker()
{
    enqueue(nullptr); // Tell the worker thread to exit
    thread.join(); // Wait for the worker thread to exit
    fclose(file); // Close the file
}

/***********************************************************************
 * rocblas_ostream functions                                           *
 ***********************************************************************/

// Get worker for file descriptor
std::shared_ptr<rocblas_ostream::worker> rocblas_ostream::get_worker(int fd)
{
    // For a fd indicating an error
    if(fd == -1)
        return nullptr;

    // Initial slice of struct stat which contains device ID and inode
    struct file_id_t
    {
        dev_t st_dev; // ID of device containing file
        ino_t st_ino; // Inode number
    };

    // Assert common initial sequence
    static_assert(std::is_standard_layout<file_id_t>{} && std::is_standard_layout<struct stat>{}
                      && offsetof(file_id_t, st_dev) == 0 && offsetof(struct stat, st_dev) == 0
                      && offsetof(file_id_t, st_ino) == offsetof(struct stat, st_ino)
                      && std::is_same<decltype(file_id_t::st_dev), decltype(stat::st_dev)>{}
                      && std::is_same<decltype(file_id_t::st_ino), decltype(stat::st_ino)>{},
                  "struct stat and file_id_t are not layout-compatible");

    // Compares device IDs and inodes for containers
    struct file_id_less
    {
        bool operator()(const file_id_t& lhs, const file_id_t& rhs) const
        {
            return lhs.st_ino < rhs.st_ino || (lhs.st_ino == rhs.st_ino && lhs.st_dev < rhs.st_dev);
        }
    };

    // Map from file_id to a worker shared_ptr
    static std::map<file_id_t, std::shared_ptr<worker>, file_id_less> map;

    // Mutex for accessing the map
    static std::mutex map_mutex;

    // C++ allows type punning of common initial sequences
    union
    {
        struct stat statbuf;
        file_id_t   file_id;
    };

    // Get the device ID and inode, to detect files already open
    if(fstat(fd, &statbuf))
    {
        perror("Error executing fstat()");
        return {};
    }

    // Lock the map
    std::lock_guard<std::mutex> lock(map_mutex);

    // Insert an element if it doesn't already exist
    auto worker_iter = map.emplace(file_id, nullptr).first;

    // If a new entry was inserted, or an old entry is empty, create worker
    if(!worker_iter->second)
        worker_iter->second = std::make_shared<worker>(fd);

    // Return the existing or new worker matching the file
    return worker_iter->second;
}

// Construct from a file descriptor, which is duped
rocblas_ostream::rocblas_ostream(int fd)
    : worker_ptr(get_worker(fcntl(fd, F_DUPFD_CLOEXEC)))
{
    if(!worker_ptr)
    {
        dprintf(STDERR_FILENO, "Cannot clone file descriptor %d: %m\n", fd);
        abort();
    }
}

// Open a file and return a worker
std::shared_ptr<rocblas_ostream::worker> rocblas_ostream::get_worker(const char* filename)
{
    int fd = open(filename, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    return fd == -1 ? nullptr : get_worker(fd);
}

// Construct from a C filename
rocblas_ostream::rocblas_ostream(const char* filename)
    : worker_ptr(get_worker(filename))
{
    if(!worker_ptr)
    {
        dprintf(STDERR_FILENO, "Cannot open %s: %m\n", filename);
        abort();
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

// Floating-point output
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, double x)
{
    char        s[32];
    const char* out;

    if(std::isnan(x))
        out = ".nan";
    else if(std::isinf(x))
        out = x < 0 ? "-.inf" : ".inf";
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

// IO Manipulators
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, std::ostream& (*pf)(std::ostream&))
{
    // Output the manipulator to the buffer
    os.os << pf;

    // If the manipulator is std::endl or std::flush, flush the output
    if(pf == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)
       || pf == static_cast<std::ostream& (*)(std::ostream&)>(std::flush))
        os.flush();

    return os;
}

// Output streams
ROCBLAS_EXPORT rocblas_ostream rocblas_cout(STDOUT_FILENO);
ROCBLAS_EXPORT rocblas_ostream rocblas_cerr(STDERR_FILENO);
