/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_arguments.hpp"
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

//
//

ArgumentModel::ArgumentModel(const std::vector<arg_type>& params)
    : m_args(params)
{
}

bool ArgumentModel::hasParam(arg_type a)
{
    return std::find(m_args.begin(), m_args.end(), a) != m_args.end();
}

void ArgumentModel::log_perf(std::stringstream& name_line,
                             std::stringstream& val_line,
                             const Arguments&   arg,
                             double             gpu_us,
                             double             gflops,
                             double             gpu_bytes,
                             double             cpu_us,
                             double             norm1,
                             double             norm2)
{
    int batch_count = hasParam(e_batch_count) ? arg.batch_count : 1;
    int hot_calls   = arg.iters < 1 ? 1 : arg.iters;

    // per/us to per/sec *10^6
    double rocblas_gflops = gflops * batch_count * hot_calls / gpu_us * 1e6;
    double cblas_gflops   = gflops * batch_count / cpu_us * 1e6;

    // bytes/us to GB/s = 10^6 * 10^-9 = /10^3
    double rocblas_GBps = gpu_bytes * batch_count / gpu_us / 1e3;

    name_line << "rocblas-Gflops,rocblas-GB/s,rocblas-us,";
    val_line << rocblas_gflops << "," << rocblas_GBps << "," << gpu_us << ",";

    if(arg.unit_check || arg.norm_check)
    {
        name_line << "CPU-Gflops,CPU-us,";
        val_line << cblas_gflops << "," << cpu_us << ",";
        if(arg.norm_check)
        {
            name_line << "norm1,norm2";
            val_line << norm1 << "," << norm2 << ",";
        }
    }
}
