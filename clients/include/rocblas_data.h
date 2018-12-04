/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_DATA_H_
#define ROCBLAS_DATA_H_

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <functional>
#include <iterator>
#include <boost/iterator/filter_iterator.hpp>
#include "rocblas.h"
#include "rocblas_arguments.h"

enum rocblas_data_class
{
    rocblas_test_data,
    rocblas_perf_data,
};

// Class used to read Arguments data into the tests
template <rocblas_data_class>
struct RocBLAS_Data
{
    // filter iterator
    using iterator = boost::filter_iterator<std::function<bool(const Arguments&)>,
                                            std::istream_iterator<Arguments>>;

    // Initialize class
    static void init(const std::string& file) { datafile = file; }

    // begin() iterator which accepts an optional filter.
    static iterator begin(std::function<bool(const Arguments&)> filter = [](const Arguments&) {
        return true;
    })
    {
        auto& ifs = get().ifs;

        // We re-seek the file back to position 0
        ifs.clear();
        ifs.seekg(0);

        // We create a filter iterator which will choose only those test cases
        // we want right now. This is to preserve Gtest output structure while
        // not creating no-op tests which "always pass".
        return iterator(filter, std::istream_iterator<Arguments>(ifs));
    }

    // end() iterator
    static iterator end() { return iterator(); }

    private:
    // We define this function to generate a single instance of the class on
    // first use so that we don't depend on the static initialization order.
    static RocBLAS_Data& get()
    {
        static RocBLAS_Data singleton;
        return singleton;
    }

    // Private constructor which opens file
    RocBLAS_Data()
    {
        ifs.open(datafile, std::ifstream::binary);
        if(ifs.fail())
        {
            std::cerr << "Cannot open " << datafile << ": " << strerror(errno) << std::endl;
            throw std::ifstream::failure("Cannot open " + datafile);
        }
    }

    static std::string datafile;
    std::ifstream ifs;
};

// The datafile must be initialized by calling RocBLAS_Data<>::init()
template <rocblas_data_class C>
std::string RocBLAS_Data<C>::datafile =
    "(Uninitialized data. RocBLAS_Data<...>::init needs to be called first.)";

// RocBLAS_Data is instantiated once per rocblas_data_class enum
// One is for the correctness tests; one is for the performance tests
using RocBLAS_TestData = RocBLAS_Data<rocblas_test_data>;
using RocBLAS_PerfData = RocBLAS_Data<rocblas_perf_data>;

#endif
