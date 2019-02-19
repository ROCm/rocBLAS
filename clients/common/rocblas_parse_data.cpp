#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "utility.hpp"
#include "rocblas_data.hpp"
#include "rocblas_parse_data.hpp"

// Parse YAML data
static std::string rocblas_parse_yaml(const std::string& yaml)
{
    char temp[] = "/tmp/rocblas-XXXXXX";
    int fd      = mkostemp(temp, O_CLOEXEC);
    if(fd == -1)
    {
        perror("Cannot open temporary file");
        exit(1);
    }
    int saved_stdout = fcntl(1, F_DUPFD_CLOEXEC, 0);
    dup2(fd, 1);
    std::string cmd = rocblas_exepath() + "rocblas_gentest.py " + yaml;
    std::cerr << cmd << std::endl;
    int status = system(cmd.c_str());
    dup2(saved_stdout, 1);
    close(saved_stdout);
    if(status == -1 || !WIFEXITED(status) || WEXITSTATUS(status))
        exit(1);
    return temp;
}

// Parse --data and --yaml command-line arguments
bool rocblas_parse_data(int& argc, char** argv, const std::string& default_file)
{
    std::string filename;
    char** argv_p = argv + 1;
    bool help = false, yaml = false;

    // Scan, process and remove any --yaml or --data options
    for(int i = 1; argv[i]; ++i)
    {
        if(!strcmp(argv[i], "--data") || (yaml |= !strcmp(argv[i], "--yaml")))
        {
            if(filename != "")
            {
                std::cerr << "Only one of the --yaml and --data options may be specified"
                          << std::endl;
                exit(1);
            }

            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(1);
            }
            filename = argv[++i];
        }
        else
        {
            *argv_p++ = argv[i];
            if(!help && (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")))
            {
                help = true;
                std::cout << "\n"
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
        RocBLAS_TestData::set_filename(filename);
        return true;
    }

    return false;
}
