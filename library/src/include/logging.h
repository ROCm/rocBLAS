/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef LOGGING_H
#define LOGGING_H
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/param.h>

// open std::ofstream for file in pathname contained in environment_variable_name.
// If this is not successful then open std::ofstream for filename in current 
// working directory. Return std::ofstream. If std::ofstream could not be opened
// then the returned std::ofstream is not opened
inline std::ofstream open_logfile(std::string environment_variable_name, std::string filename)
{
    std::ofstream logfile_ofs;
    std::string logfile_pathname;

    // first try to open filepath in environment_variable_name
    char const* tmp = getenv(environment_variable_name.c_str());
    if(tmp != NULL)
    {
        logfile_pathname = (std::string)tmp;
        logfile_ofs.open(logfile_pathname);
    }

    // second option: open file in current working directory
    if(logfile_ofs.is_open() == false)
    {
        char temp[MAXPATHLEN];
        std::string curr_work_dir = (getcwd(temp, MAXPATHLEN) ? std::string(temp) : std::string(""));
        logfile_pathname = curr_work_dir + "/" + filename;
        logfile_ofs.open(logfile_pathname);
    }

    return logfile_ofs;
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
    log_arg(std::ofstream& ofs, std::string& separator) : ofs_(ofs), separator_(separator) {}

    // generic operator
    template <typename T>
    void operator()(T& x) const
    {
        ofs_ << separator_ << x;
    }

    // operator for rocblas_float_complex
    void operator()(const rocblas_float_complex complex_value) const
    {
        ofs_ << separator_ << complex_value.x << separator_ << complex_value.y;
    }

    // operator for rocblas_double_complex
    void operator()(const rocblas_double_complex complex_value) const
    {
        ofs_ << separator_ << complex_value.x << separator_ << complex_value.y;
    }

    private:
    std::ofstream& ofs_;
    std::string& separator_;
};

// log_function will log "/nh|x1|x2|x3|...xn" to ofs where | is
// replaced by separator. h is the first argument, and it is preceded
// by a new line, not a separator. Each argument x1, x2, x3,   xn
// is preceded by separator. Typically separator will be a comma
// or a space. 
// Note that xs is a variadic parameter pack, and in this comment
// we assume the expansion xs... is x1,x2,x3,...xn.
template <typename H, typename... Ts>
void log_arguments(std::ofstream& ofs, std::string& separator, H head, Ts&... xs)
{
    // output newline followed by first argument with no comma
    ofs << "\n" << head;
    // repetitively output: comma then next argument in xs...
    each_args(log_arg{ofs, separator}, xs...);
}

template <typename H>
void log_argument(std::ofstream& ofs, std::string& separator, H head)
{
    // output newline followed by first argument with no comma
    ofs << "\n" << head;
}

template <typename H>
void log_argument(std::ofstream& ofs, H head)
{
    // output newline followed by first argument with no comma
    ofs << "\n" << head;
}

#endif
