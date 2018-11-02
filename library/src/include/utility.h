/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include <fstream>
#include <string>

// if trace logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
// void log_function(rocblas_handle handle, H head, Ts&... xs)
void log_trace(rocblas_handle handle, H head, Ts&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocblas_layer_mode_log_trace)
        {
            std::string comma_separator = ",";

            std::ostream* os = handle->log_trace_os;
            log_arguments(*os, comma_separator, head, xs...);
        }
    }
}

// if bench logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocblas-bench.
template <typename H, typename... Ts>
void log_bench(rocblas_handle handle, H head, std::string precision, Ts&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocblas_layer_mode_log_bench)
        {
            std::string space_separator = " ";

            std::ostream* os = handle->log_bench_os;
            log_arguments(*os, space_separator, head, precision, xs...);
        }
    }
}

// return letters in place of rocblas enums
std::string rocblas_transpose_letter(rocblas_operation trans);
std::string rocblas_side_letter(rocblas_side side);
std::string rocblas_fill_letter(rocblas_fill fill);
std::string rocblas_diag_letter(rocblas_diagonal diag);
std::string rocblas_datatype_letter(rocblas_datatype type);

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T>
std::string replaceX(std::string input_string)
{
    if(std::is_same<T, float>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 's');
    }
    else if(std::is_same<T, double>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'd');
    }
    else if(std::is_same<T, rocblas_float_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'c');
    }
    else if(std::is_same<T, rocblas_double_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'z');
    }
    else if(std::is_same<T, rocblas_half>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'h');
    }
    return input_string;
}

static inline bool isAligned(const void *pointer, size_t byte_count)
{ 
    return (uintptr_t)pointer % byte_count == 0;
}

#endif
