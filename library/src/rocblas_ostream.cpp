#include "rocblas_ostream.hpp"
#include "rocblas.h"
#include <cerrno>
#include <cstdio>

/***********************************************************************
 * log_worker functions handle logging in a single thread              *
 ***********************************************************************/

// enqueue a string to be written and freed by the worker
void log_worker::enqueue(std::shared_ptr<std::string> str)
{
    { // keep lock for as short as possible
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(std::move(str));
    }
    cond.notify_all();
}

// Worker thread which waits for strings to be logged
void log_worker::worker()
{
    std::unique_lock<std::mutex> lock(mutex);
    while(true)
    {
        // Wait for any data
        cond.wait(lock, [&] { return !queue.empty(); });

        // With the mutex locked, pop data from the front of queue
        std::shared_ptr<std::string> log = queue.front();
        queue.pop_front();

        // A nullptr indicates closing of the file and
        if(!log)
            break;

        // Only write non-zero-length strings
        size_t size = log->size();
        if(size)
        {
            // Temporarily unlock mutex, allowing other writers to queue
            lock.unlock();

            // Write the data to the filehandle until it's all written
            const char* data = log->data();
            ssize_t     written;
            while(size && (written = write(filehandle, data, size)) < size)
            {
                if(written < 0)
                {
                    // All errors except EINTR are fatal
                    if(errno != EINTR)
                    {
                        perror("Error writing log");
                        break;
                    }
                }
                else
                {
                    // Advance the data past the bytes written so far
                    data += written;
                    size -= written;
                }
            }

            // Delete the data
            log.reset();

            // Re-lock the mutex in preparation for cond.wait
            lock.lock();
        }
    }
}

std::shared_ptr<log_worker> log_worker::get_worker(int fh)
{
    struct stat statbuf;

    // Get the device ID and inode, to detect files already open
    if(fstat(fh, &statbuf))
    {
        perror("error executing fstat()");
        return {};
    }

    // Lock the map
    std::lock_guard<std::mutex> lock(map_mutex);

    std::shared_ptr<log_worker> worker;

    // Insert an element if it doesn't already exist
    bool inserted;
    std::tie(worker, inserted) = map.emplace(statbuf, nullptr);

    // If a new entry was inserted, replace its shared_ptr with a new worker
    if(inserted)
        worker = std::make_shared<log_worker>(fh);

    return worker;
}

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
    fsync(filehandle); // Sync the filehandle
    close(filehandle); // Close the filehandle
}

/***********************************************************************
 * rocblas_ostream functions                                           *
 ***********************************************************************/

// Flush the output
void rocblas_ostream::flush()
{
    if(worker)
    {
        auto& str = dynamic_cast<std::ostringstream&>(os);
        worker->enqueue(str.str());
        str.clear();
        str.str({});
    }
    else
    {
        os.flush();
    }
}

// Destroy the rocblas_ostream
rocblas_ostream::~rocblas_ostream()
{
    flush();
    if(worker)
        delete dynamic_cast<std::ostringstream*>(&os);
}

// Floating-point output
rocblas_ostream& operator<<(rocblas_ostream& os, double x)
{
    char        s[32] = "";
    const char* out   = s;

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
    }
    os.os << out;
    return os;
}
