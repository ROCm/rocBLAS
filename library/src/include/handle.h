/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HANDLE_H
#define HANDLE_H
#include <hip/hip_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <string>
#include <algorithm>

#include "rocblas.h"

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
******************************************************************************/
struct _rocblas_handle
{

    _rocblas_handle();
    ~_rocblas_handle();

    rocblas_status set_stream(hipStream_t stream);
    rocblas_status get_stream(hipStream_t* stream) const;

    rocblas_int device;
    hipDeviceProp_t device_properties;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    rocblas_layer_mode layer_mode;
    std::ofstream log_ofs;
    // need to read environment variable that contains layer_mode
};

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

// The variatic template function each_args applies the functor f
// to each argument in the expansion of the parameter pack xs...
//
// Note that in ((void)f(xs),0) the C/C++ comma operator evaluates
// the first expression (void)f(xs) and discards the output, then
// it evaluates the second expression 0 and returns the output 0.
//
// It thus calls (void)f(xs) on each parameter in xs... as a bye-product of
// building the initializer_list 0,0,0,...0. The initializer_list is discarded
//
template <typename F, typename... Ts>
void each_args(F f, Ts&... xs)
{
    (void)std::initializer_list<int>{((void)f(xs), 0)...};
}

// Workaround for gcc warnings when each_args called with single argument
// and no parameter pack
template <typename F>
void each_args(F)
{
}

// Functor to log single argument to ofs
// The overloaded () in log_arg is the function call operator
// The definition in log_arg says "objects of type log_arg can have
// the function call operator () applied to them with operand x,
// and it will output x to ofs and return void.
struct log_arg
{
    log_arg(std::ofstream& ofs_input) : ofs(ofs_input) {}

    template <typename T>
    void operator()(T& x) const
    {
        ofs << "," << x;
    }

    private:
    std::ofstream& ofs;
};

// if logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_logging) == true
// then
// log_function will log "/nh,x1,x2,x3,...xn" to handle->log_ofs
// Note that xs is a variadic parameter pack, and here
// we assume the expansion xs... is x1,x2,x3,...xn.
template <typename H, typename... Ts>
void log_function(rocblas_handle handle, H h, Ts&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocblas_layer_mode_logging)
        {
            std::ofstream& ofs = handle->log_ofs;
            // output newline followed by first argument with no comma
            ofs << "\n" << h;
            // repetitively output: comma then next argument in xs...
            each_args(log_arg{ofs}, xs...);
        }
    }
}

#endif
