/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_parse_data.hpp"
#include "rocblas_data.hpp"
#include "utility.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/types.h>
#ifdef WIN32
//
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
//
#ifdef __cpp_lib_filesystem
#include <filesystem>
#else
#include <experimental/filesystem>

namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif
#endif // WIN32

// Parse YAML data
static std::string rocblas_parse_yaml(const std::string& yaml)
{
    std::string tmp     = rocblas_tempname();
    auto        exepath = rocblas_exepath();
    auto cmd = exepath + "rocblas_gentest.py --template " + exepath + "rocblas_template.yaml -o "
               + tmp + " " + yaml;
    rocblas_cerr << cmd << std::endl;

#ifdef WIN32
    int status = std::system(cmd.c_str());
#else
    int status = system(cmd.c_str());
    if(status == -1 || !WIFEXITED(status) || WEXITSTATUS(status))
        exit(EXIT_FAILURE);
#endif

    return tmp;
}

// Parse --data and --yaml command-line arguments
bool rocblas_parse_data(int& argc, char** argv, const std::string& default_file)
{
    std::string filename;
    char**      argv_p = argv + 1;
    bool        help = false, yaml = false;

    // Scan, process and remove any --yaml or --data options
    for(int i = 1; argv[i]; ++i)
    {
        if(!strcmp(argv[i], "--data") || !strcmp(argv[i], "--yaml"))
        {
            if(!strcmp(argv[i], "--yaml"))
            {
                yaml = true;
            }

            if(filename != "")
            {
                rocblas_cerr << "Only one of the --yaml and --data options may be specified"
                             << std::endl;
                exit(EXIT_FAILURE);
            }

            if(!argv[i + 1] || !argv[i + 1][0])
            {
                rocblas_cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            filename = argv[++i];
        }
        else
        {
            *argv_p++ = argv[i];
            if(!help && (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")))
            {
                help = true;
                rocblas_cout << "\n"
                             << argv[0] << " [ --data <path> | --yaml <path> ] <options> ...\n"
                             << std::endl;
            }
        }
    }

    // argc and argv contain remaining options and non-option arguments
    *argv_p = nullptr;
    argc    = argv_p - argv;

    if(filename == "-")
        filename = "/dev/stdin";
    else if(filename == "")
        filename = default_file;

    if(yaml)
        filename = rocblas_parse_yaml(filename);

    if(filename != "")
    {
        RocBLAS_TestData::set_filename(filename, yaml);
        return true;
    }

    return false;
}
