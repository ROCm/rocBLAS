/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_DATA_H_
#define ROCBLAS_DATA_H_

#include "rocblas_arguments.hpp"
#include "test_cleanup.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

// Class used to read Arguments data into the tests
class RocBLAS_TestData
{
    // data filename
    static auto& filename()
    {
        static std::string filename
            = "(Uninitialized data. RocBLAS_TestData::set_filename needs to be called first.)";
        return filename;
    }

    // filter iterator
    class iterator : public std::istream_iterator<Arguments>
    {
        bool (*const filter)(const Arguments&) = nullptr;

        // Skip entries for which filter is false
        void skip_filter()
        {
            if(filter)
                while(*this != std::istream_iterator<Arguments>{} && !filter(**this))
                    ++*static_cast<std::istream_iterator<Arguments>*>(this);
        }

    public:
        // Constructor takes a filter and iterator
        iterator(bool filter(const Arguments&), std::istream_iterator<Arguments> iter)
            : std::istream_iterator<Arguments>(iter)
            , filter(filter)
        {
            skip_filter();
        }

        // Default end iterator and nullptr filter
        iterator() = default;

        // Preincrement iterator operator with filtering
        iterator& operator++()
        {
            ++*static_cast<std::istream_iterator<Arguments>*>(this);
            skip_filter();
            return *this;
        }

        // We do not need a postincrement iterator operator
        // We delete it here so that the base class's isn't silently called
        // To implement it, use "auto old = *this; ++*this; return old;"
        iterator operator++(int) = delete;
    };

public:
    // Initialize filename, optionally removing it at exit
    static void set_filename(std::string name, bool remove_atexit = false)
    {
        filename() = std::move(name);
        if(remove_atexit)
        {
            auto cleanup = [] { remove(filename().c_str()); };
            atexit(cleanup);
            at_quick_exit(cleanup);
        }
    }

    // begin() iterator which accepts an optional filter.
    static iterator begin(bool filter(const Arguments&) = nullptr)
    {
        static std::ifstream* ifs;

        // If this is the first time, or after test_cleanup::cleanup() has been called
        if(!ifs)
        {
            // Allocate a std::ifstream and register it to be deleted during cleanup
            ifs = test_cleanup::allocate(&ifs, filename(), std::ifstream::binary);
            if(!ifs || ifs->fail())
            {
                rocblas_cerr << "Cannot open " << filename() << ": " << strerror(errno)
                             << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // We re-seek the file back to position 0
        ifs->clear();
        ifs->seekg(0);

        // Validate the data file format
        Arguments::validate(*ifs);

        // We create a filter iterator which will choose only the test cases we want right now.
        // This is to preserve Gtest structure while not creating no-op tests which "always pass".
        return iterator(filter, std::istream_iterator<Arguments>(*ifs));
    }

    // end() iterator
    static iterator end()
    {
        return {};
    }
};

#endif
