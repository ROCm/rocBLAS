#include "rocblas_ostream.hpp"
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <map>
#include <sys/stat.h>
#include <type_traits>

/***********************************************************************
 * log_worker functions handle logging in a single thread              *
 ***********************************************************************/

// enqueue a string to be written and freed by the worker
void log_worker::enqueue(std::shared_ptr<std::string> ptr)
{
    // Only nullptr or nonzero sized strings are sent
    if(!ptr || ptr->size())
    {
        { // Keep lock for as short as possible
            std::lock_guard<std::mutex> lock(mutex);
            queue.push_back(ptr);
        }
        cond.notify_all();
    }
}

// Worker thread which waits for strings to be logged
void log_worker::worker()
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
        queue.pop_front();

        // A nullptr indicates closing of the file
        if(!log)
            break;

        // Temporarily unlock mutex, allowing other writers to queue
        lock.unlock();

        // Write the data
        fwrite(log->data(), 1, log->size(), file);

        // Delete the data
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
log_worker::log_worker(int fd)
    : file(fdopen(fd, "a"))
    , worker_thread([=] { worker(); })
{
    if(!file)
    {
        perror("fdopen() error");
        abort();
    }
}

// Get worker for file descriptor
std::shared_ptr<log_worker> log_worker::get_worker(int fd)
{
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
            return lhs.st_dev < rhs.st_dev || (lhs.st_dev == rhs.st_dev && lhs.st_ino < rhs.st_ino);
        }
    };

    // Map from file_id to a log_worker shared_ptr
    static std::map<file_id_t, std::shared_ptr<log_worker>, file_id_less> map;

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
    auto worker = map.emplace(file_id, nullptr).first;

    // If a new entry was inserted, or an old entry is empty, create worker
    if(!worker->second)
        worker->second = std::make_shared<log_worker>(fd);

    // Return the existing or new worker matching the file
    return worker->second;
}

// Open a file and return a worker
std::shared_ptr<log_worker> log_worker::get_worker(const char* filename)
{
    int fd = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if(fd == -1)
    {
        fprintf(stderr, "Cannot open %s: %m\n", filename);
        abort();
    }
    return get_worker(fd);
}

// Destroy a worker when all references to it are gone
log_worker::~log_worker()
{
    enqueue(nullptr); // Tell the worker thread to exit
    worker_thread.join(); // Wait for the worker thread to exit
    fclose(file); // Close the file
}

/***********************************************************************
 * rocblas_ostream functions                                           *
 ***********************************************************************/

// Flush the output
ROCBLAS_EXPORT void rocblas_ostream::flush()
{
    if(worker)
    {
        worker->enqueue(std::make_shared<std::string>(os.str()));
        os.clear();
        os.str({});
    }
}

// Destroy the rocblas_ostream
ROCBLAS_EXPORT rocblas_ostream::~rocblas_ostream()
{
    // Flush any pending IO
    flush();

    // If we had a worker, we delete its temporary ostringstream
    if(worker)
        delete &os;
}

// Floating-point output
// We use <cstdio> and <cstring> functions for fine-grained control of output
ROCBLAS_EXPORT rocblas_ostream& operator<<(rocblas_ostream& os, double x)
{
    char        s[32] = "";
    const char* out;

    if(std::isnan(x))
        out = ".nan";
    else if(std::isinf(x))
        out = x < 0 ? "-.inf" : ".inf";
    else
    {
        snprintf(s, sizeof(s) - 2, "%.17g", x);

        // If no decimal point or exponent, append .0
        char* end = s + strcspn(s, ".eE");
        if(!*end)
            strcat(end, ".0");

        out = s;
    }
    os.os << out;
    return os;
}
