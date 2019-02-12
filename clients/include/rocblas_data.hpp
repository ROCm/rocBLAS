/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_DATA_H_
#define ROCBLAS_DATA_H_

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <cerrno>
#include <utility>
#include <boost/iterator/filter_iterator.hpp>
#include "rocblas_arguments.hpp"
#include "test_cleanup.hpp"

// Class used to read Arguments data into the tests
class RocBLAS_TestData
{
    // data filename
    static std::string& filename()
    {
        static std::string filename =
            "(Uninitialized data. RocBLAS_TestData<...>::init needs to be called first.)";
        return filename;
    }

    public:
    // filter iterator
    using iterator = boost::filter_iterator<std::function<bool(const Arguments&)>,
                                            std::istream_iterator<Arguments>>;
    // Initialize filename
    static void set_filename(std::string name) { filename() = std::move(name); }

    // begin() iterator which accepts an optional filter.
    static iterator begin(std::function<bool(const Arguments&)> filter = [](auto) { return true; })
    {
        static std::ifstream* ifs;

        // If this is the first time, or after test_cleanup::cleanup() has been called
        if(!ifs)
        {
            // Allocate a std::ifstream and register it to be deleted during cleanup
            ifs = test_cleanup::allocate<std::ifstream>(&ifs, filename(), std::ifstream::binary);
            if(!ifs || ifs->fail())
            {
                std::cerr << "Cannot open " << filename() << ": " << strerror(errno) << std::endl;
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
    static iterator end() { return {}; }
};

#endif
