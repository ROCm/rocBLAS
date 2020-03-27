/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../../library/src/include/handle.h"
#include "cblas_interface.hpp"
#include "rocblas.hpp"
#include "rocblas_math.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "utility.hpp"
#include <cstdio>
#include <thread>

constexpr size_t NTIMES  = 100; // Number of times the tests are repeated across files
constexpr size_t NLINES  = 10000; // Number of lines each thread outputs
constexpr size_t MAXLEN  = 2000; // Maximum length of random strings
constexpr size_t NTHREAD = 16; // Number of threads to run simultaneously
constexpr size_t SIGLEN  = 16; // Number of characters in signature

inline void testing_ostream_threadsafety(const Arguments& arg)
{
    rocblas_seedrand();

    for(size_t n = 0; n < NTIMES; ++n)
    {
        // Open a file in /tmp
        char path[] = "/tmp/rocblas-XXXXXX";
        int  fd     = mkstemp(path);
        if(fd == -1)
        {
            FAIL() << "Cannot open temporary file " << path;
            return;
        }

        // Signature for detecting garbled output
        auto sig = [](const std::string& s) {
            char h[SIGLEN + 1];
            snprintf(h, sizeof(h), "%0*zX", int(SIGLEN), tuple_helper::hash(s));
            return std::string(h);
        };

        // Verify that the signature matches the string
        auto check_sig
            = [&sig](const std::string& s) { return s.substr(0, SIGLEN) == sig(s.substr(SIGLEN)); };

        // Each thread writes random strings with signature checksums
        auto thread_func = [&sig](rocblas_ostream os) {
            for(size_t i = 0; i < NLINES; ++i)
            {
                // Random ASCII string
                auto s = random_string(MAXLEN);

                // Write the signature followed by the random string, flushing at the end
                os << sig(s) << s << std::endl;
            }
        };

        std::thread threads[NTHREAD];

        // Launch NTHREAD threads, creating a rocblas_ostream for each thread by duplicating fd
        for(auto& t : threads)
            t = std::thread(thread_func, rocblas_ostream(fd));

        // Close the original file descriptor
        if(close(fd))
            FAIL() << "Could not close filehandle for " << path;

        // Wait for the threads to exit
        for(auto& t : threads)
            t.join();

        // Reopen the file to check its integrity
        std::ifstream is(path);
        if(!is.is_open())
        {
            FAIL() << "Could not open " << path;
            return;
        }

        // For each line in the file, make sure its signature matches.
        // This detects interleaved IO which causes garbled output.
        for(std::string line; std::getline(is, line);)
        {
            if(!check_sig(line))
            {
                FAIL() << " detected garbled output in " << path << ":\n\n" << line << "\n";
                return;
            }
        }

        // If there were no failures, erase the temporary file
        remove(path);
    }
}
