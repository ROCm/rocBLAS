/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ARGUMENT_MODEL_HPP_
#define _ARGUMENT_MODEL_HPP_

#include "rocblas_arguments.hpp"

// ArgumentModel template has a variadic list of argument enums
template <rocblas_argument... Args>
class ArgumentModel
{
public:
    void log_perf(rocblas_ostream& name_line,
                  rocblas_ostream& val_line,
                  const Arguments& arg,
                  double           gpu_us,
                  double           gflops,
                  double           gpu_bytes,
                  double           cpu_us,
                  double           norm1,
                  double           norm2)
    {
        constexpr bool has_batch_count = ((Args == e_batch_count) || ...);
        rocblas_int    batch_count     = has_batch_count ? arg.batch_count : 1;
        rocblas_int    hot_calls       = arg.iters < 1 ? 1 : arg.iters;

        // per/us to per/sec *10^6
        double rocblas_gflops = gflops * batch_count * hot_calls / gpu_us * 1e6;
        double cblas_gflops   = gflops * batch_count / cpu_us * 1e6;

        // bytes/us to GB/s = 10^6 * 10^-9 = 10^-3
        double rocblas_GBps = gpu_bytes * batch_count / gpu_us / 1e3;

        name_line << "rocblas-Gflops,rocblas-GB/s,rocblas-us,";
        val_line << rocblas_gflops << "," << rocblas_GBps << "," << gpu_us << ",";

        if(arg.unit_check || arg.norm_check)
        {
            name_line << "CPU-Gflops,CPU-us,";
            val_line << cblas_gflops << "," << cpu_us << ",";
            if(arg.norm_check)
            {
                name_line << "norm_error_host_ptr,norm_error_device_ptr,";
                val_line << norm1 << "," << norm2 << ",";
            }
        }
    }

    template <typename T>
    void log_args(rocblas_ostream& str,
                  const Arguments& arg,
                  double           gpu_us,
                  double           gflops,
                  double           gpu_bytes = 0,
                  double           cpu_us    = 0,
                  double           norm1     = 0,
                  double           norm2     = 0)
    {
        rocblas_ostream name_list;
        rocblas_ostream value_list;

        // Output (name, value) pairs to name_list and value_list
        auto print = [&, delim = ""](const char* name, auto&& value) mutable {
            name_list << delim << name;
            value_list << delim << value;
            delim = ",";
        };

        // Apply the arguments, calling print on each one
        (ArgumentsHelper::apply<Args>(print, arg, T{}), ...);

        if(arg.timing)
            log_perf(name_list, value_list, arg, gpu_us, gflops, gpu_bytes, cpu_us, norm1, norm2);

        str << name_list << "\n" << value_list << std::endl;
    }
};

#endif
