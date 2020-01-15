/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_PARSE_DATA_H
#define _ROCBLAS_PARSE_DATA_H

#include <string>

// Parse --data and --yaml command-line arguments
bool rocblas_parse_data(int& argc, char** argv, const std::string& default_file = "");

#endif
